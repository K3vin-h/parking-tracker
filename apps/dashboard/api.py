"""
JSON API endpoints for the dashboard app.

This module is the HTTP entry point that connects the two halves of the system
that were built independently:

  * the CV inference pipeline (apps.cv.pipeline) — reads a plate from an image
  * the session/billing services (apps.parking.services) — open/close sessions

`upload()` is the connector: it accepts an uploaded plate image over HTTP, runs
the CV pipeline on it, and hands the extracted reading to the appropriate
parking service. It deliberately holds NO business logic of its own — billing,
orphan handling, and plate matching all live in apps.parking.services. The view
only does HTTP concerns: auth, input validation, file handling, and shaping the
JSON response.

WHY plain Django (not DRF): the project intentionally has no REST framework
dependency, so we return JsonResponse directly and rely on Django's built-in
CSRF + auth middleware.
"""

import logging
import math
import os
import tempfile
import uuid
import warnings
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.core.exceptions import SuspiciousFileOperation
from django.core.files.storage import default_storage
from django.db import transaction
from django.http import FileResponse, Http404, HttpRequest, JsonResponse
from django.shortcuts import get_object_or_404
from django.views.decorators.csrf import csrf_protect
from django.views.decorators.http import require_GET, require_POST
from PIL import Image, UnidentifiedImageError

from apps.cv.pipeline import PipelineResult, get_pipeline
from apps.cv.preprocessing import MAX_IMAGE_PIXELS, UnsafeImagePathError
from apps.parking.models import LotSettings, ParkingLot, PlateDetectionEvent
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


def _error(message: str, status: int) -> JsonResponse:
    """Return a consistent error envelope. Never leak internal paths/details."""
    return JsonResponse({"error": message}, status=status)


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
    deployment shouldn't have to name it on every upload. Returns None when the
    caller is ambiguous (no name given and multiple lots) or names an unknown
    lot; the view turns that into a 400/404.
    """
    lot_name = (request.POST.get("lot") or "").strip()
    if lot_name:
        return (
            ParkingLot.objects.select_related("settings")
            .filter(name=lot_name)
            .first()
        )
    # No name supplied: only safe to auto-pick if there is exactly one lot.
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

    os.makedirs(settings.CV_PROCESSING_TEMP_ROOT, mode=0o700, exist_ok=True)
    temp_path = None
    try:
        suffix = Path(stored_name).suffix
        with storage.open(stored_name, "rb") as source:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                suffix=suffix,
                dir=settings.CV_PROCESSING_TEMP_ROOT,
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
                # Another cleanup path already removed the ephemeral file.
                pass
            except OSError:
                logger.exception("Failed to remove temporary CV upload copy")


def _delete_upload(storage, stored_name: str | None) -> None:
    """
    Delete a rejected upload without masking the original request outcome.

    Storage cleanup failures are operationally important but should not replace a
    useful 4xx/5xx response with a second exception, so they are logged explicitly.
    """
    if not stored_name:
        return
    try:
        storage.delete(stored_name)
    except Exception:
        logger.exception("Failed to delete rejected upload")


@csrf_protect
@login_required
@require_POST
def upload(request: HttpRequest) -> JsonResponse:
    """
    Accept a plate image, run CV, and open/close a parking session.

    Request (multipart/form-data):
      image       — the plate photo (required; JPEG or PNG, <= 10 MB)
      event_type  — "entry" or "exit" (required; the client/camera states which
                    gate produced the image — see plan decision)
      lot         — lot name (optional when exactly one lot exists)

    Auth: staff only. @login_required handles authentication; the is_staff check
    below handles authorization (the project's single-role model: is_staff = full
    access, matching the correct-event endpoint's requirement).

    Responses:
      200 — entry opened / exit billed / exit unmatched (queued for review)
      400 — missing/invalid event_type, unknown/ambiguous lot, or missing image
      403 — authenticated but not staff
      413 — image exceeds size cap
      415 — unsupported content type
      422 — entry image whose plate could not be read (no session opened)
      503 — CV model weights missing or corrupt (server not ready)
    """
    # ── Authorization ──────────────────────────────────────────────────────
    if not request.user.is_staff:
        return _error("Staff access required.", 403)

    # ── Validate event_type ────────────────────────────────────────────────
    event_type = (request.POST.get("event_type") or "").strip().lower()
    if event_type not in VALID_EVENT_TYPES:
        return _error("event_type must be 'entry' or 'exit'.", 400)

    # ── Validate lot ───────────────────────────────────────────────────────
    lot = _resolve_lot(request)
    if lot is None:
        return _error("Unknown or unspecified lot.", 400)
    try:
        lot_settings = lot.settings
    except LotSettings.DoesNotExist:
        logger.error("Lot %s has no billing settings configured", lot.pk)
        return _error("Lot billing settings are not configured.", 503)

    # ── Validate the uploaded file ─────────────────────────────────────────
    upload_file = request.FILES.get("image")
    if upload_file is None:
        return _error("No image file provided.", 400)
    if upload_file.size > MAX_UPLOAD_BYTES:
        return _error("Image exceeds the 10 MB size limit.", 413)
    declared_extension = ALLOWED_CONTENT_TYPES.get(upload_file.content_type)
    if declared_extension is None:
        return _error("Unsupported image type; use JPEG or PNG.", 415)

    # Verify the actual bytes are a JPEG/PNG BEFORE writing anything to disk.
    # WHY: content_type is the attacker-controlled request header. Without this
    # check a hostile client could pass the allowlist with arbitrary bytes (a web
    # shell or polyglot) and have them persisted before load_image() rejects them.
    # Pillow's header inspection here means only genuine images are ever stored.
    try:
        extension = _inspect_uploaded_image(upload_file, declared_extension)
    except UploadImageError as exc:
        return _error(exc.message, exc.status)

    # ── Save once in private storage ─────────────────────────────────────────
    # The parking services persist the same storage name on the detection event.
    # Local storage takes the zero-copy path into CV; remote storage makes one
    # bounded scratch copy. The random UUID name prevents predictable collisions.
    stored_name = None
    keep_file = False
    try:
        stored_name = default_storage.save(
            f"plates/{uuid.uuid4().hex}{extension}", upload_file
        )
        # Re-check the size against the BYTES ON DISK. WHY: upload_file.size is
        # derived from client-supplied framing (Content-Length / multipart) and
        # can understate a chunked-encoding upload, so the pre-save check above
        # is bypassable. default_storage.size() measures what was actually
        # written — the only authoritative size.
        if default_storage.size(stored_name) > MAX_UPLOAD_BYTES:
            return _error("Image exceeds the 10 MB size limit.", 413)

        try:
            pipeline = get_pipeline(
                settings.CV_DETECTOR_WEIGHTS, settings.CV_RECOGNIZER_WEIGHTS
            )
        except (FileNotFoundError, RuntimeError):
            # Weights missing/corrupt is an operational/config problem, not the
            # caller's fault. get_pipeline already logged the path-stripped cause.
            logger.exception("CV pipeline unavailable for upload")
            return _error("Plate recognition is temporarily unavailable.", 503)

        try:
            with _processing_path(default_storage, stored_name) as image_path:
                result = _validate_pipeline_result(pipeline.process(image_path))
        except UnsafeImagePathError:
            # We control the path, so this should be unreachable; treat as a bug.
            logger.exception("Saved upload failed the MEDIA_ROOT safety check")
            return _error("Internal error processing the image.", 500)
        except UploadImageError as exc:
            return _error(exc.message, exc.status)
        except FileNotFoundError:
            logger.info("Upload passed header checks but could not be decoded")
            return _error("Image could not be decoded safely.", 422)
        except ValueError:
            logger.info("Upload rejected by CV image validation")
            return _error("Image could not be decoded safely.", 422)

        # Build all shared response values before the service transaction. The
        # persisted review flag uses the lot-specific threshold, not the pipeline's
        # fixed default threshold.
        is_low_confidence = result["confidence"] < lot_settings.confidence_threshold

        # An outer transaction makes service writes and response construction one
        # unit. If shaping/serialization fails, the nested service transaction rolls
        # back and the finally block removes the now-unreferenced image.
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
                    logger.info("Entry upload rejected: plate unreadable")
                    return _error("Plate could not be read; no session opened.", 422)
                response = _entry_response(result, session, is_low_confidence)
            else:
                session = handle_exit(
                    plate_text=result["plate_text"],
                    confidence=result["confidence"],
                    bounding_box=result["bounding_box"],
                    image=stored_name,
                    lot=lot,
                )
                response = _exit_response(
                    result,
                    session,
                    is_low_confidence=is_low_confidence if session else True,
                )

        keep_file = True
        return response

    except Exception:
        # Any unexpected failure: don't leave an orphaned file on disk, and never
        # surface internals to the client. The traceback is logged server-side.
        logger.exception("Unexpected error handling upload")
        return _error("Internal error processing the upload.", 500)
    finally:
        if not keep_file:
            _delete_upload(default_storage, stored_name)


def _cv_fields(result: PipelineResult, is_low_confidence: bool) -> dict:
    """Shared CV portion of the JSON envelope."""
    return {
        "plate_text": result["plate_text"],
        "confidence": round(float(result["confidence"]), 4),
        "is_low_confidence": is_low_confidence,
        "bounding_box": result["bounding_box"],
    }


def _entry_response(
    result: PipelineResult, session, is_low_confidence: bool
) -> JsonResponse:
    """Shape the 200 response for a successful entry."""
    payload = _cv_fields(result, is_low_confidence)
    payload.update(
        {
            "event_type": "entry",
            "session_id": session.pk,
            "status": session.status,
            "has_duplicate_warning": session.has_duplicate_warning,
        }
    )
    return JsonResponse(payload, status=200)


def _exit_response(
    result: PipelineResult, session, is_low_confidence: bool
) -> JsonResponse:
    """Shape the 200 response for an exit (matched or queued for review)."""
    payload = _cv_fields(result, is_low_confidence)
    payload["event_type"] = "exit"
    if session is None:
        # Exit-without-entry: handle_exit already logged a review event.
        payload.update({"matched": False, "queued_for_review": True})
    else:
        payload.update(
            {
                "matched": True,
                "session_id": session.pk,
                "status": session.status,
                # Decimal -> str preserves exact cents (never float in JSON).
                "charge_amount": str(session.charge_amount),
                "duration_seconds": session.duration_seconds,
            }
        )
    return JsonResponse(payload, status=200)


@login_required
@require_GET
def event_image(request: HttpRequest, event_id: int) -> FileResponse:
    """
    Serve a plate image only to authenticated staff.

    Images are deliberately absent from public MEDIA_URL routing. Opening through
    Django storage also works for local and remote backends, while ``private,
    no-store`` prevents browsers and shared proxies from retaining plate data.
    """
    if not request.user.is_staff:
        return _error("Staff access required.", 403)

    event = get_object_or_404(PlateDetectionEvent, pk=event_id)
    stored_name = event.image.name
    if not stored_name:
        raise Http404
    storage_path = PurePosixPath(stored_name)
    if (
        storage_path.is_absolute()
        or ".." in storage_path.parts
        or not storage_path.parts
        or storage_path.parts[0] != "plates"
    ):
        logger.error("Event %s references an unsafe image name", event.pk)
        raise Http404

    content_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
    }
    content_type = content_types.get(Path(stored_name).suffix.lower())
    if content_type is None:
        logger.error("Event %s references an unsupported image extension", event.pk)
        raise Http404

    try:
        image_file = default_storage.open(stored_name, "rb")
    except (FileNotFoundError, OSError, SuspiciousFileOperation):
        logger.warning("Stored image missing for event %s", event.pk)
        raise Http404 from None

    response = FileResponse(
        image_file,
        content_type=content_type,
        as_attachment=False,
        filename=f"plate-event-{event.pk}{Path(stored_name).suffix.lower()}",
    )
    response["Cache-Control"] = "private, no-store"
    return response
