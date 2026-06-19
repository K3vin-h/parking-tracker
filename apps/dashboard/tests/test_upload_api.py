"""
Tests for the POST /api/upload/ endpoint (apps.dashboard.api.upload).

The CV pipeline is mocked end-to-end (no weights, no torch, no disk I/O for the
image): we patch `get_pipeline` so `.process()` returns a canned PipelineResult,
and we patch `default_storage` so no real file is written. The parking SERVICES
(handle_entry/handle_exit) run for real against the test DB — that is the
integration boundary we actually want to exercise from the view.

WHY mock storage rather than write files: matches the project convention of
keeping image data as path strings (see apps/parking/tests/test_services.py) and
keeps the suite free of filesystem side effects.
"""

import io
import os
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest
from django.contrib.auth import get_user_model
from django.core.files.storage import default_storage
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client, override_settings
from django.urls import reverse
from PIL import Image

from apps.parking.models import (
    LotSettings,
    ParkingLot,
    ParkingSession,
    PlateDetectionEvent,
)

User = get_user_model()

UPLOAD_URL = reverse("dashboard:api_upload")
STORED_NAME = "plates/deadbeef.jpg"


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def staff_user(db):
    return User.objects.create_user(
        username="attendant",
        email="attendant@example.com",
        password="testpass123",
        is_staff=True,
    )


@pytest.fixture
def regular_user(db):
    return User.objects.create_user(
        username="driver",
        email="driver@example.com",
        password="testpass123",
        is_staff=False,
    )


@pytest.fixture
def parking_lot(db):
    return ParkingLot.objects.create(name="Test Lot")


@pytest.fixture
def lot_settings(parking_lot):
    return LotSettings.objects.create(
        lot=parking_lot,
        rate=Decimal("5.00"),
        billing_unit="hour",
        grace_period_minutes=15,
        confidence_threshold=0.6,
    )


def _result(plate_text="ABC123", confidence=0.95, low=False):
    """Build a canned PipelineResult dict matching pipeline.process() output."""
    return {
        "plate_text": plate_text,
        "confidence": confidence,
        "bounding_box": [0.1, 0.2, 0.3, 0.4],
        "is_low_confidence": low,
    }


def _real_image_bytes(fmt="JPEG"):
    """Produce genuine image bytes so the view's Pillow header check passes."""
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), color=(120, 120, 120)).save(buffer, format=fmt)
    return buffer.getvalue()


def _image(content=None, content_type="image/jpeg", name="plate.jpg"):
    if content is None:
        content = _real_image_bytes("JPEG")
    return SimpleUploadedFile(name, content, content_type=content_type)


def _patched(process_result=None, process_side_effect=None, pipeline_factory=None):
    """
    Build mocks for the view's CV pipeline factory and storage.

    Returns a (storage, factory) tuple; callers feed them to `patch(...)` for
    `apps.dashboard.api.default_storage` and `apps.dashboard.api.get_pipeline`.
    """
    storage = MagicMock()
    storage.save.return_value = STORED_NAME
    storage.path.return_value = f"/srv/media/{STORED_NAME}"
    storage.size.return_value = 1024  # on-disk size re-check stays under the cap

    pipeline = MagicMock()
    if process_side_effect is not None:
        pipeline.process.side_effect = process_side_effect
    else:
        pipeline.process.return_value = process_result or _result()

    factory = pipeline_factory or MagicMock(return_value=pipeline)

    return storage, factory


# ── Authorization ─────────────────────────────────────────────────────────


@pytest.mark.django_db
class TestUploadAuth:
    def test_anonymous_is_redirected_to_login(self, client):
        resp = client.post(UPLOAD_URL)
        # @login_required redirects unauthenticated users to LOGIN_URL.
        assert resp.status_code == 302
        assert "/login/" in resp.url

    def test_non_staff_user_is_redirected_to_login(self, client, regular_user):
        client.force_login(regular_user)
        resp = client.post(UPLOAD_URL, {"event_type": "entry", "image": _image()})
        assert resp.status_code == 302
        assert "/login/" in resp.url

    def test_staff_request_without_csrf_token_is_forbidden(
        self, staff_user, lot_settings
    ):
        csrf_client = Client(enforce_csrf_checks=True)
        csrf_client.force_login(staff_user)
        resp = csrf_client.post(
            UPLOAD_URL,
            {"event_type": "entry", "image": _image()},
        )
        assert resp.status_code == 403


# ── Input validation ────────────────────────────────────────────────────────


@pytest.mark.django_db
class TestUploadValidation:
    def test_missing_event_type_is_400(self, client, staff_user, lot_settings):
        client.force_login(staff_user)
        resp = client.post(UPLOAD_URL, {"image": _image()})
        assert resp.status_code == 400

    def test_invalid_event_type_is_400(self, client, staff_user, lot_settings):
        client.force_login(staff_user)
        resp = client.post(UPLOAD_URL, {"event_type": "sideways", "image": _image()})
        assert resp.status_code == 400

    def test_missing_image_is_400(self, client, staff_user, lot_settings):
        client.force_login(staff_user)
        resp = client.post(UPLOAD_URL, {"event_type": "entry"})
        assert resp.status_code == 400

    def test_unsupported_content_type_is_415(self, client, staff_user, lot_settings):
        client.force_login(staff_user)
        resp = client.post(
            UPLOAD_URL,
            {"event_type": "entry", "image": _image(content_type="image/gif")},
        )
        assert resp.status_code == 415

    def test_oversize_image_is_413(self, client, staff_user, lot_settings):
        client.force_login(staff_user)
        with patch("apps.dashboard.api.MAX_UPLOAD_BYTES", 4):
            resp = client.post(
                UPLOAD_URL,
                {"event_type": "entry", "image": _image(content=b"too-big")},
            )
        assert resp.status_code == 413

    @override_settings(PARKING_UPLOAD_MAX_BYTES=4)
    def test_streaming_handler_rejects_before_storage_save(
        self, client, staff_user, lot_settings
    ):
        client.force_login(staff_user)
        storage = MagicMock()
        with patch("apps.dashboard.api.default_storage", storage):
            resp = client.post(
                UPLOAD_URL,
                {"event_type": "entry", "image": _image()},
            )
        assert resp.status_code == 413
        storage.save.assert_not_called()

    def test_ambiguous_lot_is_400(self, client, staff_user, parking_lot):
        # Two lots exist and the request names none → ambiguous → 400. Rejected
        # before any file I/O, so no storage/pipeline patching is needed.
        ParkingLot.objects.create(name="Second Lot")
        client.force_login(staff_user)
        resp = client.post(UPLOAD_URL, {"event_type": "entry", "image": _image()})
        assert resp.status_code == 400

    def test_unknown_named_lot_is_400(self, client, staff_user, lot_settings):
        client.force_login(staff_user)
        resp = client.post(
            UPLOAD_URL,
            {"event_type": "entry", "lot": "Nope", "image": _image()},
        )
        assert resp.status_code == 400

    def test_spoofed_content_type_is_415_and_not_saved(
        self, client, staff_user, lot_settings
    ):
        # Non-image bytes labelled image/jpeg must be rejected BEFORE any save,
        # by the in-memory Pillow header check.
        client.force_login(staff_user)
        storage = MagicMock()
        bogus = SimpleUploadedFile(
            "plate.jpg", b"definitely not an image", content_type="image/jpeg"
        )
        with patch("apps.dashboard.api.default_storage", storage):
            resp = client.post(UPLOAD_URL, {"event_type": "entry", "image": bogus})
        assert resp.status_code == 415
        storage.save.assert_not_called()

    def test_declared_type_must_match_detected_format(
        self, client, staff_user, lot_settings
    ):
        client.force_login(staff_user)
        storage = MagicMock()
        mislabeled = _image(
            content=_real_image_bytes("PNG"),
            content_type="image/jpeg",
        )
        with patch("apps.dashboard.api.default_storage", storage):
            resp = client.post(
                UPLOAD_URL,
                {"event_type": "entry", "image": mislabeled},
            )
        assert resp.status_code == 415
        storage.save.assert_not_called()

    def test_decompression_bomb_is_413_and_not_saved(
        self, client, staff_user, lot_settings
    ):
        client.force_login(staff_user)
        storage = MagicMock()
        with (
            patch("apps.dashboard.api.default_storage", storage),
            patch(
                "apps.dashboard.api.Image.open",
                side_effect=Image.DecompressionBombError("too many pixels"),
            ),
        ):
            resp = client.post(
                UPLOAD_URL,
                {"event_type": "entry", "image": _image()},
            )
        assert resp.status_code == 413
        storage.save.assert_not_called()

    def test_max_upload_bytes_is_ten_megabytes(self):
        # Guard against a unit slip (bytes vs MB) in the size-cap constant.
        from apps.dashboard.api import MAX_UPLOAD_BYTES

        assert MAX_UPLOAD_BYTES == 10 * 1024 * 1024

    def test_oversize_on_disk_is_413_and_deletes_file(
        self, client, staff_user, lot_settings
    ):
        # Client understates size (bypassing the pre-save check) but the file on
        # disk exceeds the cap → the post-save re-stat rejects it and cleans up.
        from apps.dashboard.api import MAX_UPLOAD_BYTES

        client.force_login(staff_user)
        storage, factory = _patched()
        storage.size.return_value = MAX_UPLOAD_BYTES + 1
        with (
            patch("apps.dashboard.api.default_storage", storage),
            patch("apps.dashboard.api.get_pipeline", factory),
        ):
            resp = client.post(UPLOAD_URL, {"event_type": "entry", "image": _image()})
        assert resp.status_code == 413
        storage.delete.assert_called_once_with(STORED_NAME)
        factory.assert_not_called()  # rejected before the pipeline loads


# ── Entry flow ───────────────────────────────────────────────────────────────


@pytest.mark.django_db
class TestUploadEntry:
    def test_entry_opens_session_and_event(self, client, staff_user, lot_settings):
        client.force_login(staff_user)
        storage, factory = _patched(_result(plate_text="ABC123"))
        with (
            patch("apps.dashboard.api.default_storage", storage),
            patch("apps.dashboard.api.get_pipeline", factory),
        ):
            resp = client.post(UPLOAD_URL, {"event_type": "entry", "image": _image()})

        assert resp.status_code == 200
        body = resp.json()
        assert body["event_type"] == "entry"
        assert body["plate_text"] == "ABC123"
        assert body["status"] == ParkingSession.Status.ACTIVE

        session = ParkingSession.objects.get(pk=body["session_id"])
        assert session.plate_text == "ABC123"
        event = PlateDetectionEvent.objects.get(session=session)
        assert event.event_type == "entry"
        assert event.image.name == STORED_NAME  # referenced, not re-saved
        # Saved exactly once; not deleted on the happy path.
        storage.save.assert_called_once()
        storage.delete.assert_not_called()

    def test_htmx_entry_returns_result_partial(self, client, staff_user, lot_settings):
        """Negotiate HTML for browser swaps while leaving JSON as the default."""
        client.force_login(staff_user)
        storage, factory = _patched(_result(plate_text="HTML01"))
        with (
            patch("apps.dashboard.api.default_storage", storage),
            patch("apps.dashboard.api.get_pipeline", factory),
        ):
            response = client.post(
                UPLOAD_URL,
                {"event_type": "entry", "image": _image()},
                HTTP_HX_REQUEST="true",
            )
        assert response.status_code == 200
        assert response["Content-Type"].startswith("text/html")
        assert b"HTML01" in response.content

    def test_response_uses_lot_specific_confidence_threshold(
        self, client, staff_user, lot_settings
    ):
        lot_settings.confidence_threshold = 0.99
        lot_settings.save(update_fields=["confidence_threshold"])
        client.force_login(staff_user)
        storage, factory = _patched(
            _result(plate_text="THRESH1", confidence=0.95, low=False)
        )
        with (
            patch("apps.dashboard.api.default_storage", storage),
            patch("apps.dashboard.api.get_pipeline", factory),
        ):
            resp = client.post(
                UPLOAD_URL,
                {"event_type": "entry", "image": _image()},
            )

        assert resp.status_code == 200
        assert resp.json()["is_low_confidence"] is True
        assert PlateDetectionEvent.objects.get().is_low_confidence is True

    def test_response_failure_rolls_back_database_and_deletes_file(
        self, client, staff_user, lot_settings
    ):
        client.force_login(staff_user)
        storage, factory = _patched(_result(plate_text="ROLLBK1"))
        with (
            patch("apps.dashboard.api.default_storage", storage),
            patch("apps.dashboard.api.get_pipeline", factory),
            patch(
                "apps.dashboard.api._entry_response",
                side_effect=TypeError("serialization failed"),
            ),
        ):
            resp = client.post(
                UPLOAD_URL,
                {"event_type": "entry", "image": _image()},
            )

        assert resp.status_code == 500
        assert ParkingSession.objects.count() == 0
        assert PlateDetectionEvent.objects.count() == 0
        storage.delete.assert_called_once_with(STORED_NAME)

    def test_unreadable_plate_on_entry_is_queued_without_session(
        self, client, staff_user, lot_settings
    ):
        client.force_login(staff_user)
        # Empty plate → no invalid session, but the private image is retained so
        # an operator can recover the plate from the review queue.
        storage, factory = _patched(_result(plate_text=""))
        with (
            patch("apps.dashboard.api.default_storage", storage),
            patch("apps.dashboard.api.get_pipeline", factory),
        ):
            resp = client.post(UPLOAD_URL, {"event_type": "entry", "image": _image()})
        assert resp.status_code == 422
        assert ParkingSession.objects.count() == 0
        body = resp.json()
        assert body["queued_for_review"] is True
        event = PlateDetectionEvent.objects.get(pk=body["event_id"])
        assert event.session_id is None
        assert event.is_low_confidence is True
        assert event.image.name == STORED_NAME
        storage.delete.assert_not_called()

    def test_htmx_unreadable_entry_returns_swappable_review_result(
        self, client, staff_user, lot_settings
    ):
        """Return HTML 200 so HTMX displays the queued unreadable state."""
        client.force_login(staff_user)
        storage, factory = _patched(_result(plate_text=""))
        with (
            patch("apps.dashboard.api.default_storage", storage),
            patch("apps.dashboard.api.get_pipeline", factory),
        ):
            response = client.post(
                UPLOAD_URL,
                {"event_type": "entry", "image": _image()},
                HTTP_HX_REQUEST="true",
            )
        assert response.status_code == 200
        assert b"Queued for review" in response.content
        assert (
            PlateDetectionEvent.objects.filter(
                session=None, is_low_confidence=True
            ).count()
            == 1
        )


# ── Exit flow ────────────────────────────────────────────────────────────────


@pytest.mark.django_db
class TestUploadExit:
    def test_exit_matches_and_bills_session(self, client, staff_user, lot_settings):
        client.force_login(staff_user)
        # Open a session first via the entry path, then exit the same plate.
        storage, factory = _patched(_result(plate_text="EXIT01"))
        with (
            patch("apps.dashboard.api.default_storage", storage),
            patch("apps.dashboard.api.get_pipeline", factory),
        ):
            client.post(UPLOAD_URL, {"event_type": "entry", "image": _image()})
            resp = client.post(UPLOAD_URL, {"event_type": "exit", "image": _image()})

        assert resp.status_code == 200
        body = resp.json()
        assert body["event_type"] == "exit"
        assert body["matched"] is True
        assert body["status"] == ParkingSession.Status.COMPLETED
        # charge_amount is serialized as a string (Decimal-safe).
        assert isinstance(body["charge_amount"], str)

    def test_exit_without_entry_is_queued_for_review(
        self, client, staff_user, lot_settings
    ):
        client.force_login(staff_user)
        storage, factory = _patched(_result(plate_text="GHOST1"))
        with (
            patch("apps.dashboard.api.default_storage", storage),
            patch("apps.dashboard.api.get_pipeline", factory),
        ):
            resp = client.post(UPLOAD_URL, {"event_type": "exit", "image": _image()})

        assert resp.status_code == 200
        body = resp.json()
        assert body["matched"] is False
        assert body["queued_for_review"] is True
        # A review event with no session was recorded by handle_exit.
        assert PlateDetectionEvent.objects.filter(
            session=None, event_type="exit"
        ).exists()

    def test_unmatched_exit_response_is_always_low_confidence(
        self, client, staff_user, lot_settings
    ):
        client.force_login(staff_user)
        storage, factory = _patched(
            _result(plate_text="GHOST2", confidence=0.99, low=False)
        )
        with (
            patch("apps.dashboard.api.default_storage", storage),
            patch("apps.dashboard.api.get_pipeline", factory),
        ):
            resp = client.post(
                UPLOAD_URL,
                {"event_type": "exit", "image": _image()},
            )

        assert resp.status_code == 200
        assert resp.json()["is_low_confidence"] is True
        assert PlateDetectionEvent.objects.get().is_low_confidence is True


# ── CV availability ──────────────────────────────────────────────────────────


@pytest.mark.django_db
class TestUploadPipelineUnavailable:
    def test_missing_weights_is_503_and_deletes_file(
        self, client, staff_user, lot_settings
    ):
        client.force_login(staff_user)
        storage = MagicMock()
        storage.save.return_value = STORED_NAME
        storage.size.return_value = 1024
        factory = MagicMock(side_effect=FileNotFoundError("weights missing"))
        with (
            patch("apps.dashboard.api.default_storage", storage),
            patch("apps.dashboard.api.get_pipeline", factory),
        ):
            resp = client.post(UPLOAD_URL, {"event_type": "entry", "image": _image()})
        assert resp.status_code == 503
        storage.delete.assert_called_once_with(STORED_NAME)

    def test_unsafe_path_is_500_and_deletes_file(
        self, client, staff_user, lot_settings
    ):
        from apps.cv.preprocessing import UnsafeImagePathError

        client.force_login(staff_user)
        storage, factory = _patched(
            process_side_effect=UnsafeImagePathError("escaped MEDIA_ROOT")
        )
        with (
            patch("apps.dashboard.api.default_storage", storage),
            patch("apps.dashboard.api.get_pipeline", factory),
        ):
            resp = client.post(UPLOAD_URL, {"event_type": "entry", "image": _image()})
        assert resp.status_code == 500
        storage.delete.assert_called_once_with(STORED_NAME)

    def test_decode_failure_is_422_and_deletes_file(
        self, client, staff_user, lot_settings
    ):
        client.force_login(staff_user)
        storage, factory = _patched(
            process_side_effect=FileNotFoundError("decode rejected")
        )
        with (
            patch("apps.dashboard.api.default_storage", storage),
            patch("apps.dashboard.api.get_pipeline", factory),
        ):
            resp = client.post(
                UPLOAD_URL,
                {"event_type": "entry", "image": _image()},
            )
        assert resp.status_code == 422
        storage.delete.assert_called_once_with(STORED_NAME)

    def test_unexpected_error_is_500_and_deletes_file(
        self, client, staff_user, lot_settings
    ):
        client.force_login(staff_user)
        storage, factory = _patched(
            process_side_effect=RuntimeError("boom mid-process")
        )
        with (
            patch("apps.dashboard.api.default_storage", storage),
            patch("apps.dashboard.api.get_pipeline", factory),
        ):
            resp = client.post(UPLOAD_URL, {"event_type": "entry", "image": _image()})
        assert resp.status_code == 500
        storage.delete.assert_called_once_with(STORED_NAME)

    def test_lot_without_settings_is_503_and_deletes_file(
        self, client, staff_user, parking_lot
    ):
        # Reject configuration errors before saving a file or loading CV models.
        client.force_login(staff_user)
        storage, factory = _patched(_result(plate_text="NOCFG1"))
        with (
            patch("apps.dashboard.api.default_storage", storage),
            patch("apps.dashboard.api.get_pipeline", factory),
        ):
            resp = client.post(UPLOAD_URL, {"event_type": "entry", "image": _image()})
        assert resp.status_code == 503
        assert ParkingSession.objects.count() == 0
        storage.save.assert_not_called()
        storage.delete.assert_not_called()
        factory.assert_not_called()

    def test_invalid_pipeline_result_is_500_without_database_write(
        self, client, staff_user, lot_settings
    ):
        client.force_login(staff_user)
        storage, factory = _patched(process_result={"plate_text": "BROKEN"})
        with (
            patch("apps.dashboard.api.default_storage", storage),
            patch("apps.dashboard.api.get_pipeline", factory),
        ):
            resp = client.post(
                UPLOAD_URL,
                {"event_type": "entry", "image": _image()},
            )
        assert resp.status_code == 500
        assert ParkingSession.objects.count() == 0
        assert PlateDetectionEvent.objects.count() == 0
        storage.delete.assert_called_once_with(STORED_NAME)

    def test_remote_storage_uses_bounded_local_processing_copy(
        self, client, staff_user, lot_settings
    ):
        client.force_login(staff_user)
        storage, factory = _patched(_result(plate_text="REMOTE1"))
        storage.path.side_effect = NotImplementedError
        storage.open.return_value = io.BytesIO(_real_image_bytes())
        with (
            patch("apps.dashboard.api.default_storage", storage),
            patch("apps.dashboard.api.get_pipeline", factory),
        ):
            resp = client.post(
                UPLOAD_URL,
                {"event_type": "entry", "image": _image()},
            )

        assert resp.status_code == 200
        processing_path = factory.return_value.process.call_args.args[0]
        from django.conf import settings

        assert processing_path.startswith(str(settings.CV_PROCESSING_TEMP_ROOT))
        assert not os.path.exists(processing_path)


@pytest.mark.django_db
class TestPrivateEventImage:
    def test_event_image_requires_authentication(self, client, parking_lot):
        event = PlateDetectionEvent.objects.create(
            lot=parking_lot,
            image="plates/private.jpg",
            raw_plate_text="ABC123",
            confidence_score=0.9,
            event_type="exit",
            is_low_confidence=True,
            bounding_box=[],
        )
        resp = client.get(reverse("dashboard:api_event_image", args=[event.pk]))
        assert resp.status_code == 302

    def test_event_image_redirects_non_staff(self, client, regular_user, parking_lot):
        event = PlateDetectionEvent.objects.create(
            lot=parking_lot,
            image="plates/private.jpg",
            raw_plate_text="ABC123",
            confidence_score=0.9,
            event_type="exit",
            is_low_confidence=True,
            bounding_box=[],
        )
        client.force_login(regular_user)
        resp = client.get(reverse("dashboard:api_event_image", args=[event.pk]))
        assert resp.status_code == 302
        assert "/login/" in resp.url

    def test_staff_can_stream_event_image(
        self, client, staff_user, parking_lot, tmp_path
    ):
        with override_settings(MEDIA_ROOT=tmp_path):
            stored_name = default_storage.save(
                "plates/private.jpg",
                SimpleUploadedFile("private.jpg", _real_image_bytes()),
            )
            event = PlateDetectionEvent.objects.create(
                lot=parking_lot,
                image=stored_name,
                raw_plate_text="ABC123",
                confidence_score=0.9,
                event_type="exit",
                is_low_confidence=True,
                bounding_box=[],
            )
            client.force_login(staff_user)
            resp = client.get(reverse("dashboard:api_event_image", args=[event.pk]))
            body = b"".join(resp.streaming_content)

        assert resp.status_code == 200
        assert resp["Content-Type"] == "image/jpeg"
        assert resp["Cache-Control"] == "private, no-store"
        assert body.startswith(b"\xff\xd8")

    def test_media_root_has_no_public_url(self, client):
        resp = client.get("/media/plates/private.jpg")
        assert resp.status_code == 404
