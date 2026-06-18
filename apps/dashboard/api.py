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
import uuid

from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.core.files.storage import default_storage
from django.http import HttpRequest, JsonResponse
from django.views.decorators.csrf import csrf_protect
from django.views.decorators.http import require_POST
from PIL import Image, UnidentifiedImageError

from apps.cv.pipeline import get_pipeline
from apps.cv.preprocessing import UnsafeImagePathError
from apps.parking.models import LotSettings, ParkingLot
from apps.parking.services import handle_entry, handle_exit

logger = logging.getLogger("apps.dashboard")

# Upload guard rails. WHY enforce here, before saving: load_image() in the CV
# layer also validates size/format, but we reject obvious abuse at the boundary
# so a hostile client can never get a huge or non-image payload written to disk.
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB — generous for a phone photo.

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


def _error(message: str, status: int) -> JsonResponse:
    """Return a consistent error envelope. Never leak internal paths/details."""
    return JsonResponse({"error": message}, status=status)


def _detect_image_format(upload_file) -> str | None:
    """
    Return the real Pillow format ("JPEG"/"PNG") of an uploaded file, or None.

    WHY: validates the magic bytes in-memory so we never persist a file whose
    content does not match a real image. Uses verify() (header-only, cheap) and
    rewinds the file so the subsequent save reads it from the start.
    """
    try:
        upload_file.seek(0)
        with Image.open(upload_file) as img:
            detected = img.format
            img.verify()  # structural check without a full decode
        return detected
    except (UnidentifiedImageError, OSError, ValueError):
        # Not a decodable image (truncated, wrong type, or corrupt header).
        return None
    finally:
        upload_file.seek(0)


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
        return ParkingLot.objects.filter(name=lot_name).first()
    # No name supplied: only safe to auto-pick if there is exactly one lot.
    lots = ParkingLot.objects.all()[:2]
    return lots[0] if len(lots) == 1 else None


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

    # ── Validate the uploaded file ─────────────────────────────────────────
    upload_file = request.FILES.get("image")
    if upload_file is None:
        return _error("No image file provided.", 400)
    if upload_file.size > MAX_UPLOAD_BYTES:
        return _error("Image exceeds the 10 MB size limit.", 413)
    extension = ALLOWED_CONTENT_TYPES.get(upload_file.content_type)
    if extension is None:
        return _error("Unsupported image type; use JPEG or PNG.", 415)

    # Verify the actual bytes are a JPEG/PNG BEFORE writing anything to disk.
    # WHY: content_type is the attacker-controlled request header. Without this
    # check a hostile client could pass the allowlist with arbitrary bytes (a web
    # shell, a polyglot) and have them land under MEDIA_ROOT — which may be served
    # directly by nginx — for the brief window before load_image() rejects them.
    # Pillow's header inspection here means only genuine images are ever stored.
    if _detect_image_format(upload_file) not in ALLOWED_IMAGE_FORMATS:
        return _error("File content is not a valid JPEG or PNG image.", 415)

    # ── Save once, under MEDIA_ROOT ────────────────────────────────────────
    # The CV pipeline requires a path inside MEDIA_ROOT (load_image enforces it),
    # AND the parking services persist the same image on the detection event.
    # We save exactly once here with a random UUID name (prevents enumeration),
    # run CV on its absolute path, then pass the RELATIVE NAME to the service so
    # the ImageField stores a reference to this file rather than copying it again.
    stored_name = default_storage.save(
        f"plates/{uuid.uuid4().hex}{extension}", upload_file
    )

    try:
        # Re-check the size against the BYTES ON DISK. WHY: upload_file.size is
        # derived from client-supplied framing (Content-Length / multipart) and
        # can understate a chunked-encoding upload, so the pre-save check above
        # is bypassable. default_storage.size() measures what was actually
        # written — the only authoritative size.
        if default_storage.size(stored_name) > MAX_UPLOAD_BYTES:
            default_storage.delete(stored_name)
            return _error("Image exceeds the 10 MB size limit.", 413)

        try:
            pipeline = get_pipeline(
                settings.CV_DETECTOR_WEIGHTS, settings.CV_RECOGNIZER_WEIGHTS
            )
        except (FileNotFoundError, RuntimeError):
            # Weights missing/corrupt is an operational/config problem, not the
            # caller's fault. get_pipeline already logged the path-stripped cause.
            logger.exception("CV pipeline unavailable for upload")
            default_storage.delete(stored_name)
            return _error("Plate recognition is temporarily unavailable.", 503)

        try:
            result = pipeline.process(default_storage.path(stored_name))
        except UnsafeImagePathError:
            # We control the path, so this should be unreachable; treat as a bug.
            logger.exception("Saved upload failed the MEDIA_ROOT safety check")
            default_storage.delete(stored_name)
            return _error("Internal error processing the image.", 500)

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
                # Empty/garbage plate on an ENTRY: per the service contract we do
                # NOT open a session keyed on an unreadable plate. Drop the file
                # and tell the client to retry (or use the manual flow later).
                logger.info("Entry upload rejected: plate unreadable")
                default_storage.delete(stored_name)
                return _error("Plate could not be read; no session opened.", 422)
            return _entry_response(result, session)

        # event_type == "exit"
        session = handle_exit(
            plate_text=result["plate_text"],
            confidence=result["confidence"],
            bounding_box=result["bounding_box"],
            image=stored_name,
            lot=lot,
        )
        return _exit_response(result, session)

    except LotSettings.DoesNotExist:
        # The lot exists but has no billing settings — a configuration bug, not a
        # client error. Surface it as "not ready" rather than a generic 500.
        logger.error("Lot %s has no billing settings configured", lot.pk)
        default_storage.delete(stored_name)
        return _error("Lot billing settings are not configured.", 503)
    except Exception:
        # Any unexpected failure: don't leave an orphaned file on disk, and never
        # surface internals to the client. The traceback is logged server-side.
        logger.exception("Unexpected error handling upload")
        default_storage.delete(stored_name)
        return _error("Internal error processing the upload.", 500)


def _cv_fields(result: dict) -> dict:
    """Shared CV portion of the JSON envelope."""
    return {
        "plate_text": result["plate_text"],
        "confidence": round(float(result["confidence"]), 4),
        "is_low_confidence": result["is_low_confidence"],
        "bounding_box": result["bounding_box"],
    }


def _entry_response(result: dict, session) -> JsonResponse:
    """Shape the 200 response for a successful entry."""
    payload = _cv_fields(result)
    payload.update(
        {
            "event_type": "entry",
            "session_id": session.pk,
            "status": session.status,
            "has_duplicate_warning": session.has_duplicate_warning,
        }
    )
    return JsonResponse(payload, status=200)


def _exit_response(result: dict, session) -> JsonResponse:
    """Shape the 200 response for an exit (matched or queued for review)."""
    payload = _cv_fields(result)
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
