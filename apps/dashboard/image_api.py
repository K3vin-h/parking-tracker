"""Private authenticated plate-image serving endpoint."""

import logging
from pathlib import Path, PurePosixPath

from django.core.exceptions import SuspiciousFileOperation
from django.core.files.storage import default_storage
from django.http import FileResponse, Http404, HttpRequest
from django.shortcuts import get_object_or_404
from django.views.decorators.http import require_GET

from apps.parking.models import PlateDetectionEvent

from .api import staff_required

logger = logging.getLogger("apps.dashboard")


@staff_required
@require_GET
def event_image(request: HttpRequest, event_id: int) -> FileResponse:
    """
    Serve a plate image only to authenticated staff.

    Images are deliberately absent from public MEDIA_URL routing. Opening through
    Django storage also works for local and remote backends, while ``private,
    no-store`` prevents browsers and shared proxies from retaining plate data.
    """
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
