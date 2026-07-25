"""
Shared plate-scan core: turn an uploaded image into a parking event.

WHY THIS MODULE EXISTS:
  The gate kiosk (apps.public) is the only uploader now — the old staff upload
  page was removed. But the hard part of that flow is the *security* of accepting
  an untrusted image (size/format/dimension checks, private storage, decode
  safety) plus wiring the CV pipeline to the billing services. That logic is
  identical regardless of who triggers it, so it lives here once, auth-free, and
  the caller layers on its own authorization + response shaping.

  `run_plate_scan()` holds NO business logic of its own — billing, orphan
  handling, and plate matching all live in apps.parking.services. It only does the
  boundary work: input validation, file handling, invoking CV, and dispatching to
  the entry/exit service. It returns a structured ScanOutcome; the caller renders
  it (the kiosk deliberately renders a privacy-reduced view — no image URL, no
  owner identity, no balance).

SECURITY (unchanged from the original staff endpoint):
  - content_type is attacker-controlled; the real bytes are verified with Pillow
    BEFORE anything is written to disk (no polyglot/web-shell persistence).
  - dimensions are checked to reject decompression bombs.
  - the on-disk size is re-measured after save (client framing can understate it).
  - files are stored under a random UUID name in private storage.
"""

import logging
import math
import os
import stat
import tempfile
import uuid
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from django.conf import settings
from django.core.files.storage import default_storage
from django.db import transaction
from django.http import HttpRequest
from PIL import Image, UnidentifiedImageError

from apps.cv.pipeline import PipelineResult, get_pipeline
from apps.cv.preprocessing import MAX_IMAGE_PIXELS, UnsafeImagePathError
from apps.parking.models import (
    LotSettings,
    ParkingLot,
    ParkingSession,
    PlateDetectionEvent,
)
from apps.parking.services import handle_entry, handle_exit

logger = logging.getLogger("apps.dashboard")

# Upload guard rails. WHY enforce here, before saving: load_image() in the CV
# layer also validates size/format, but we reject obvious abuse at the boundary
# so a hostile client can never get a huge or non-image payload written to disk.
MAX_UPLOAD_BYTES = settings.PARKING_UPLOAD_MAX_BYTES

# Content-type allowlist mapped to the on-disk extension we save under. WebP is
# intentionally excluded here (even though load_image accepts it) to keep the
# upload surface to the two formats a parking camera realistically produces.
ALLOWED_CONTENT_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
}

# Pillow format names we accept — used to verify the real bytes, independent of
# the client-supplied content_type.
ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG"}

VALID_EVENT_TYPES = {"entry", "exit"}


class UploadImageError(ValueError):
    """Represent a client-supplied image validation failure and its HTTP status."""

    def __init__(self, message: str, status: int):
        super().__init__(message)
        self.message = message
        self.status = status


class InvalidPipelineResult(RuntimeError):
    """Raised when the internal CV pipeline violates its documented result contract."""


@dataclass
class ScanOutcome:
    """
    Structured result of a scan, for the caller to render however it needs.

    `outcome` is one of:
      "error"           — validation/config failure (see error/status)
      "entry"           — an active session was opened (session, event set)
      "unreadable_entry"— entry image unreadable; queued sessionless event (status 422)
      "exit_matched"    — a session was closed and billed (session, event set)
      "exit_unmatched"  — exit with no active session; queued for review (event set)
    """

    outcome: str
    event_type: str | None = None
    result: PipelineResult | None = None
    session: ParkingSession | None = None
    event: PlateDetectionEvent | None = None
    is_low_confidence: bool = False
    error: str | None = None
    status: int = 200


def _inspect_uploaded_image(upload_file, declared_extension: str) -> str:
    """
    Validate image bytes and return their trusted on-disk extension.

    WHY dimensions are checked here: Pillow can identify a compressed image with
    enormous decoded dimensions before CV processing. Rejecting it before storage
    prevents decompression-bomb inputs from becoming server errors or persisted
    files. The detected format must also match the declared MIME type so a JPEG is
    never stored with a misleading PNG extension.
    """
    try:
        upload_file.seek(0)
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(upload_file) as img:
                detected = img.format
                width, height = img.size
                if width * height > MAX_IMAGE_PIXELS:
                    raise UploadImageError(
                        "Image dimensions exceed the 12 MP size limit.",
                        413,
                    )
                img.verify()
    except UploadImageError:
        raise
    except (Image.DecompressionBombWarning, Image.DecompressionBombError):
        raise UploadImageError(
            "Image dimensions exceed the 12 MP size limit.",
            413,
        ) from None
    except (UnidentifiedImageError, OSError, ValueError):
        raise UploadImageError(
            "File content is not a valid JPEG or PNG image.",
            415,
        ) from None
    finally:
        upload_file.seek(0)

    format_extensions = {"JPEG": ".jpg", "PNG": ".png"}
    detected_extension = format_extensions.get(detected)
    if detected_extension is None or detected not in ALLOWED_IMAGE_FORMATS:
        raise UploadImageError(
            "File content is not a valid JPEG or PNG image.",
            415,
        )
    if detected_extension != declared_extension:
        raise UploadImageError(
            "File content does not match its declared image type.",
            415,
        )
    return detected_extension


def _resolve_lot(request: HttpRequest) -> ParkingLot | None:
    """
    Resolve the target ParkingLot from the request.

    A `lot` form field (the lot name) selects a specific lot. If omitted, we
    fall back to the sole lot when exactly one exists — the common single-lot
    kiosk shouldn't have to name it on every scan. Returns None when the caller
    is ambiguous (no name given and multiple lots) or names an unknown lot.
    """
    lot_name = (request.POST.get("lot") or "").strip()
    if lot_name:
        return (
            ParkingLot.objects.select_related("settings").filter(name=lot_name).first()
        )
    lots = ParkingLot.objects.select_related("settings").all()[:2]
    return lots[0] if len(lots) == 1 else None


def _validate_pipeline_result(result) -> PipelineResult:
    """
    Validate and normalize the CV result before any database write occurs.

    WHY: response serialization happens after the parking services commit their
    records. Converting every response value up front prevents malformed model
    output from committing an event and then failing while JSON is constructed.
    """
    if not isinstance(result, dict):
        raise InvalidPipelineResult("CV pipeline returned a non-dict result")
    required = {"plate_text", "confidence", "bounding_box", "is_low_confidence"}
    if not required.issubset(result):
        raise InvalidPipelineResult("CV pipeline result is missing required fields")
    if not isinstance(result["plate_text"], str):
        raise InvalidPipelineResult("CV pipeline plate_text must be a string")
    if len(result["plate_text"]) > 20:
        raise InvalidPipelineResult("CV pipeline plate_text exceeds database limits")
    if not isinstance(result["is_low_confidence"], bool):
        raise InvalidPipelineResult("CV pipeline low-confidence flag must be boolean")

    try:
        confidence = float(result["confidence"])
        bounding_box = [float(value) for value in result["bounding_box"]]
    except (TypeError, ValueError):
        raise InvalidPipelineResult("CV pipeline returned non-numeric values") from None
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise InvalidPipelineResult("CV pipeline confidence is outside [0, 1]")
    if len(bounding_box) != 4 or not all(
        math.isfinite(value) and 0.0 <= value <= 1.0 for value in bounding_box
    ):
        raise InvalidPipelineResult(
            "CV pipeline bounding_box must contain four normalized numbers"
        )

    return {
        "plate_text": result["plate_text"],
        "confidence": confidence,
        "bounding_box": bounding_box,
        "is_low_confidence": result["is_low_confidence"],
    }


@contextmanager
def _processing_path(storage, stored_name: str):
    """
    Yield a local path for CV while retaining support for remote storage.

    FileSystemStorage exposes a direct path, which avoids a redundant copy on the
    normal deployment. Remote storage backends do not; for those, stream a bounded
    temporary copy under the private CV scratch root so path containment still
    applies without assuming MEDIA_ROOT is locally writable.
    """
    try:
        local_path = storage.path(stored_name)
    except NotImplementedError:
        local_path = None
    if local_path is not None:
        yield local_path
        return

    temp_root = _private_processing_root()
    temp_path = None
    try:
        suffix = Path(stored_name).suffix
        with storage.open(stored_name, "rb") as source:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                suffix=suffix,
                dir=temp_root,
                delete=False,
            ) as destination:
                temp_path = destination.name
                copied = 0
                for chunk in iter(lambda: source.read(64 * 1024), b""):
                    copied += len(chunk)
                    if copied > MAX_UPLOAD_BYTES:
                        raise UploadImageError(
                            "Image exceeds the 10 MB size limit.",
                            413,
                        )
                    destination.write(chunk)
        yield temp_path
    finally:
        if temp_path:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass
            except OSError:
                logger.exception("Failed to remove temporary CV upload copy")


def _private_processing_root() -> Path:
    """Create or verify the owner-only non-symlink root used for CV scratch data."""
    root = Path(settings.CV_PROCESSING_TEMP_ROOT)
    try:
        root.mkdir(mode=0o700, parents=True, exist_ok=False)
    except FileExistsError:
        pass

    try:
        root_stat = root.lstat()
    except OSError as exc:
        raise UnsafeImagePathError(
            "CV processing directory could not be inspected"
        ) from exc
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise UnsafeImagePathError(
            "CV processing directory must be a real directory"
        )
    if root_stat.st_uid != os.geteuid():
        raise UnsafeImagePathError(
            "CV processing directory must be owned by the application user"
        )
    try:
        root.chmod(0o700)
    except OSError as exc:
        raise UnsafeImagePathError(
            "CV processing directory permissions could not be secured"
        ) from exc
    return root


def _delete_upload(storage, stored_name: str | None) -> None:
    """
    Delete a rejected upload without masking the original request outcome.

    Storage cleanup failures are operationally important but should not replace a
    useful error with a second exception, so they are logged explicitly.
    """
    if not stored_name:
        return
    try:
        storage.delete(stored_name)
    except Exception:
        logger.exception("Failed to delete rejected upload")


def _queue_unreadable_entry(
    result: PipelineResult,
    stored_name: str,
    lot: ParkingLot,
) -> PlateDetectionEvent:
    """
    Persist an unreadable entry for review without opening an invalid session.

    WHY the event is sessionless and forced low-confidence: an empty plate cannot
    safely participate in active-session matching, but deleting the image would
    prevent an operator from recovering the real plate.
    """
    return PlateDetectionEvent.objects.create(
        session=None,
        lot=lot,
        image=stored_name,
        raw_plate_text=result["plate_text"],
        confidence_score=result["confidence"],
        event_type="entry",
        is_low_confidence=True,
        bounding_box=result["bounding_box"],
    )


def run_plate_scan(request: HttpRequest) -> ScanOutcome:
    """
    Validate an uploaded plate image, run CV, and open/close a parking session.

    Auth-free by design: the caller (a public kiosk view or a staff tool) applies
    its own authorization and renders the ScanOutcome. Expects multipart fields:
      image       — plate photo (required; JPEG or PNG, <= 10 MB)
      event_type  — "entry" or "exit" (required)
      lot         — lot name (optional when exactly one lot exists)

    Never raises for expected client errors — it returns ScanOutcome(outcome="error")
    with an HTTP status hint. It owns the uploaded file's lifecycle: a file is only
    retained when it becomes referenced by a persisted event/session; otherwise it
    is deleted before returning.
    """
    event_type = (request.POST.get("event_type") or "").strip().lower()
    if event_type not in VALID_EVENT_TYPES:
        return ScanOutcome(
            "error", error="event_type must be 'entry' or 'exit'.", status=400
        )

    lot = _resolve_lot(request)
    if lot is None:
        return ScanOutcome("error", error="Unknown or unspecified lot.", status=400)
    try:
        lot_settings = lot.settings
    except LotSettings.DoesNotExist:
        logger.error("Lot %s has no billing settings configured", lot.pk)
        return ScanOutcome(
            "error", error="Lot billing settings are not configured.", status=503
        )

    upload_file = request.FILES.get("image")
    if upload_file is None:
        return ScanOutcome("error", error="No image file provided.", status=400)
    if upload_file.size > MAX_UPLOAD_BYTES:
        return ScanOutcome(
            "error", error="Image exceeds the 10 MB size limit.", status=413
        )
    declared_extension = ALLOWED_CONTENT_TYPES.get(upload_file.content_type)
    if declared_extension is None:
        return ScanOutcome(
            "error", error="Unsupported image type; use JPEG or PNG.", status=415
        )

    # Verify the actual bytes are a JPEG/PNG BEFORE writing anything to disk.
    try:
        extension = _inspect_uploaded_image(upload_file, declared_extension)
    except UploadImageError as exc:
        return ScanOutcome("error", error=exc.message, status=exc.status)

    stored_name = None
    keep_file = False
    try:
        stored_name = default_storage.save(
            f"plates/{uuid.uuid4().hex}{extension}", upload_file
        )
        # Re-check the size against the BYTES ON DISK (client framing can understate).
        if default_storage.size(stored_name) > MAX_UPLOAD_BYTES:
            return ScanOutcome(
                "error", error="Image exceeds the 10 MB size limit.", status=413
            )

        try:
            pipeline = get_pipeline(
                settings.CV_DETECTOR_WEIGHTS, settings.CV_RECOGNIZER_WEIGHTS
            )
        except (FileNotFoundError, RuntimeError):
            logger.exception("CV pipeline unavailable for scan")
            return ScanOutcome(
                "error",
                error="Plate recognition is temporarily unavailable.",
                status=503,
            )

        try:
            with _processing_path(default_storage, stored_name) as image_path:
                result = _validate_pipeline_result(pipeline.process(image_path))
        except UnsafeImagePathError:
            logger.exception("Saved upload failed the MEDIA_ROOT safety check")
            return ScanOutcome(
                "error", error="Internal error processing the image.", status=500
            )
        except UploadImageError as exc:
            return ScanOutcome("error", error=exc.message, status=exc.status)
        except FileNotFoundError:
            logger.info("Upload passed header checks but could not be decoded")
            return ScanOutcome(
                "error", error="Image could not be decoded safely.", status=422
            )
        except ValueError:
            logger.info("Upload rejected by CV image validation")
            return ScanOutcome(
                "error", error="Image could not be decoded safely.", status=422
            )

        # The persisted review flag uses the lot-specific threshold, not the
        # pipeline's fixed default threshold.
        is_low_confidence = result["confidence"] < lot_settings.confidence_threshold

        with transaction.atomic():
            if event_type == "entry":
                try:
                    session = handle_entry(
                        plate_text=result["plate_text"],
                        confidence=result["confidence"],
                        bounding_box=result["bounding_box"],
                        image=stored_name,
                        lot=lot,
                    )
                except ValueError:
                    event = _queue_unreadable_entry(result, stored_name, lot)
                    logger.info(
                        "Unreadable entry queued as sessionless event %s", event.pk
                    )
                    outcome = ScanOutcome(
                        "unreadable_entry",
                        event_type="entry",
                        result=result,
                        event=event,
                        is_low_confidence=True,
                        status=422,
                    )
                else:
                    event = session.detection_events.order_by("-pk").first()
                    outcome = ScanOutcome(
                        "entry",
                        event_type="entry",
                        result=result,
                        session=session,
                        event=event,
                        is_low_confidence=is_low_confidence,
                    )
            else:
                session = handle_exit(
                    plate_text=result["plate_text"],
                    confidence=result["confidence"],
                    bounding_box=result["bounding_box"],
                    image=stored_name,
                    lot=lot,
                )
                if session is None:
                    event = (
                        PlateDetectionEvent.objects.filter(
                            lot=lot,
                            session=None,
                            image=stored_name,
                            event_type="exit",
                        )
                        .order_by("-pk")
                        .first()
                    )
                    outcome = ScanOutcome(
                        "exit_unmatched",
                        event_type="exit",
                        result=result,
                        event=event,
                        is_low_confidence=True,
                    )
                else:
                    event = session.detection_events.order_by("-pk").first()
                    outcome = ScanOutcome(
                        "exit_matched",
                        event_type="exit",
                        result=result,
                        session=session,
                        event=event,
                        is_low_confidence=is_low_confidence,
                    )

        # Set ownership only after atomic.__exit__ commits successfully. Marking
        # it inside the block leaks a stored file when commit itself fails.
        keep_file = outcome.event is not None
        return outcome
    except Exception:
        logger.exception("Unexpected error handling scan")
        return ScanOutcome(
            "error", error="Internal error processing the upload.", status=500
        )
    finally:
        if not keep_file:
            _delete_upload(default_storage, stored_name)
