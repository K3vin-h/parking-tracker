"""
Streaming upload limits for image-bearing dashboard requests.

The view-level ``UploadedFile.size`` check happens after Django has parsed the
multipart body. This module adds an earlier boundary: bytes are counted while
the multipart parser streams them, before the temporary-file handler can keep
writing an oversized request to disk.
"""

from django.conf import settings
from django.core.files.uploadhandler import FileUploadHandler, StopUpload
from django.http import JsonResponse


class UploadBodyTooLarge(Exception):
    """Raised before multipart parsing when Content-Length exceeds the hard cap."""


class BoundedUploadHandler(FileUploadHandler):
    """
    Stop multipart parsing once aggregate uploaded file bytes exceed the cap.

    WHY aggregate bytes rather than per-file bytes: accepting ten individually
    valid 10 MB files would still permit a 100 MB request to consume temporary
    disk. The upload API expects one image, so one shared request budget is the
    correct resource boundary.
    """

    def __init__(self, request=None):
        super().__init__(request)
        self._received_bytes = 0

    def handle_raw_input(
        self, input_data, META, content_length, boundary, encoding=None
    ):
        """
        Reject known-oversized bodies before reading their first byte.

        Multipart boundaries and the two short text fields add a small amount of
        framing beyond the image itself, so the request cap includes configurable
        overhead while the streamed file-byte cap remains exact.
        """
        request_limit = (
            settings.PARKING_UPLOAD_MAX_BYTES
            + settings.UPLOAD_FORM_OVERHEAD_BYTES
        )
        if content_length is not None and content_length > request_limit:
            raise UploadBodyTooLarge

    def receive_data_chunk(self, raw_data, start):
        """
        Count actual streamed bytes and stop before forwarding an excess chunk.

        Returning the chunk lets Django's normal memory/temporary-file handlers
        store valid uploads. On overflow, ``StopUpload`` closes partial temporary
        files and prevents the remaining body from being consumed.
        """
        self._received_bytes += len(raw_data)
        if self._received_bytes > settings.PARKING_UPLOAD_MAX_BYTES:
            self.request.upload_too_large = True
            raise StopUpload(connection_reset=True)
        return raw_data

    def file_complete(self, file_size):
        """This guard validates only; later handlers create the UploadedFile."""
        return None


class UploadLimitMiddleware:
    """
    Convert early upload-handler aborts into a stable JSON HTTP 413 response.

    WHY middleware: CSRF middleware may trigger multipart parsing before the
    upload view runs. This middleware is placed before CSRF so it can catch a
    Content-Length rejection and can replace a response after a streamed
    ``StopUpload`` marked the request.
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        try:
            response = self.get_response(request)
        except UploadBodyTooLarge:
            return self._response()
        if getattr(request, "upload_too_large", False):
            return self._response()
        return response

    @staticmethod
    def _response() -> JsonResponse:
        """Return the same generic envelope used by the upload API."""
        return JsonResponse(
            {"error": "Image exceeds the 10 MB size limit."},
            status=413,
        )
