"""
Tests for the PUBLIC gate kiosk scan endpoint (apps.public.scan.kiosk_scan).

The CV pipeline is mocked end-to-end (no weights, no torch, no real file I/O):
we patch `get_pipeline` in apps.dashboard.scan_core so `.process()` returns a
canned PipelineResult, and patch `default_storage` so no file is written. The
parking SERVICES (handle_entry/handle_exit) + wallet deduction run for real
against the test DB — that is the integration boundary we want to exercise.

Focus areas unique to the kiosk:
  - it is PUBLIC (anonymous can scan), unlike the old staff upload endpoint;
  - the response is PRIVACY-REDUCED (no image_url / event_id / owner / balance);
  - registered plates are billed to the wallet; guests see an amount due;
  - the endpoint is rate limited.
"""

import hashlib
import io
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from django.contrib.auth import get_user_model
from django.contrib.sessions.backends.db import SessionStore
from django.core.cache import cache
from django.core.files.uploadedfile import SimpleUploadedFile
from django.db import DatabaseError, close_old_connections, connection
from django.test import Client, RequestFactory
from django.test.utils import CaptureQueriesContext
from django.urls import reverse
from django.utils import timezone
from PIL import Image

from apps.cv.preprocessing import UnsafeImagePathError
from apps.dashboard.scan_core import ScanOutcome, _processing_path
from apps.public.scan import _consume_kiosk_request, _public_payload
from apps.parking.models import (
    KioskDeviceCapability,
    KioskImageReplayDigest,
    LicensePlate,
    LotSettings,
    ParkingLot,
    ParkingSession,
    Wallet,
    WalletTransaction,
)
from apps.parking.wallet import credit_wallet

User = get_user_model()

SCAN_URL = reverse("public:kiosk_scan")
ACTIVATE_URL = "/kiosk/activate/"
STORED_NAME = "plates/deadbeef.jpg"


@pytest.fixture(autouse=True)
def clear_rate_limit_cache():
    """The rate limiter uses the cache; isolate each test from the others."""
    cache.clear()
    yield
    cache.clear()


@pytest.fixture(autouse=True)
def configure_kiosk_token(settings):
    """Tests activate explicitly while production receives the token from env."""
    settings.KIOSK_ACTIVATION_TOKEN = "test-kiosk-secret"
    settings.KIOSK_SESSION_SECONDS = 3600


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
    return {
        "plate_text": plate_text,
        "confidence": confidence,
        "bounding_box": [0.1, 0.2, 0.3, 0.4],
        "is_low_confidence": low,
    }


@pytest.mark.parametrize(
    ("outcome", "expected"),
    [
        (
            ScanOutcome(
                outcome="unreadable_entry",
                event_type="entry",
                result=_result(plate_text="", confidence=0.2, low=True),
                is_low_confidence=True,
                status=422,
            ),
            "low_confidence",
        ),
        (
            ScanOutcome(
                outcome="exit_unmatched",
                event_type="exit",
                result=_result(),
            ),
            "unmatched_exit",
        ),
        (
            ScanOutcome(
                outcome="error",
                event_type="entry",
                error="File content is not a valid JPEG or PNG image.",
                status=415,
            ),
            "invalid_image",
        ),
        (
            ScanOutcome(
                outcome="error",
                event_type="entry",
                error="Plate recognition is temporarily unavailable.",
                status=503,
            ),
            "model_error",
        ),
    ],
)
def test_public_payload_exposes_stable_recovery_state(outcome, expected):
    """A wrong backend outcome branch must not render as a successful scan."""
    assert _public_payload(outcome)["presentation_state"] == expected


def test_public_payload_marks_low_confidence_entry_as_recovery():
    """A committed low-confidence entry must still ask the kiosk user to verify it."""
    session = SimpleNamespace(
        user_id=None,
        entry_time=timezone.now(),
    )
    outcome = ScanOutcome(
        outcome="entry",
        event_type="entry",
        result=_result(confidence=0.45, low=True),
        session=session,
        is_low_confidence=True,
    )

    payload = _public_payload(outcome)

    assert payload["presentation_state"] == "low_confidence"
    assert "confidence" not in payload


def _real_image_bytes(fmt="JPEG"):
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), color=(120, 120, 120)).save(buffer, format=fmt)
    return buffer.getvalue()


def _image(content=None, content_type="image/jpeg", name="plate.jpg"):
    if content is None:
        content = _real_image_bytes("JPEG")
    return SimpleUploadedFile(name, content, content_type=content_type)


def _mocks(process_result=None):
    """Mocked storage + pipeline factory targeting scan_core's names."""
    storage = MagicMock()
    storage.save.return_value = STORED_NAME
    storage.path.return_value = f"/srv/media/{STORED_NAME}"
    storage.size.return_value = 1024

    pipeline = MagicMock()
    pipeline.process.return_value = process_result or _result()
    factory = MagicMock(return_value=pipeline)
    return storage, factory


def _activate(client, event_type="entry", lot=""):
    """Activate one test browser with the same fixed scope as a real kiosk."""
    return client.post(
        ACTIVATE_URL,
        {
            "token": "test-kiosk-secret",
            "event_type": event_type,
            "lot": lot,
        },
    )


def _post_scan(client, process_result=None, **data):
    if "kiosk_token_fingerprint" not in client.session:
        activation = _activate(
            client,
            event_type=data.get("event_type") or "entry",
            lot=data.get("lot", ""),
        )
        assert activation.status_code == 204
    data.setdefault("kiosk_nonce", client.session["kiosk_scan_nonce"])
    storage, factory = _mocks(process_result)
    with (
        patch("apps.dashboard.scan_core.default_storage", storage),
        patch("apps.dashboard.scan_core.get_pipeline", factory),
    ):
        return client.post(SCAN_URL, data)


# ── Kiosk device authorization ────────────────────────────────────────────────


@pytest.mark.django_db
class TestKioskAuthorization:
    def test_unactivated_scan_is_rejected_before_processing(self, client):
        """Missing device activation must stop work before CV or persistence."""
        with patch(
            "apps.public.scan.run_plate_scan",
            return_value=ScanOutcome("error", error="missing image", status=400),
        ) as run_scan:
            resp = client.post(SCAN_URL, {"event_type": "entry"})

        assert resp.status_code == 403
        run_scan.assert_not_called()

    def test_wrong_activation_token_is_rejected(self, client):
        """A guessed activation token must not establish a kiosk capability."""
        resp = client.post(ACTIVATE_URL, {"token": "wrong-token"})

        assert resp.status_code == 403
        assert "kiosk_token_fingerprint" not in client.session

    def test_correct_activation_token_authorizes_scan(
        self,
        client,
        settings,
        parking_lot,
    ):
        """A trusted operator can activate one browser without storing the token."""
        settings.KIOSK_ACTIVATION_TOKEN = "test-kiosk-secret"

        resp = _activate(client, lot=parking_lot.name)

        assert resp.status_code == 204
        fingerprint = client.session["kiosk_token_fingerprint"]
        assert fingerprint
        assert fingerprint != "test-kiosk-secret"

    def test_activation_db_failure_does_not_orphan_session(self, client, parking_lot):
        """A capability write failure must not leave a scan-unable browser session."""
        with patch(
            "apps.public.scan.KioskDeviceCapability.objects.update_or_create",
            side_effect=DatabaseError("capability write failed"),
        ):
            resp = _activate(client, lot=parking_lot.name)

        assert resp.status_code == 503
        assert "kiosk_token_fingerprint" not in client.session
        assert not KioskDeviceCapability.objects.exists()

    def test_token_rotation_revokes_existing_activation(
        self,
        client,
        settings,
        parking_lot,
    ):
        """Changing the deployment token must invalidate previously activated kiosks."""
        _activate(client, lot=parking_lot.name)
        settings.KIOSK_ACTIVATION_TOKEN = "rotated-kiosk-secret"

        with patch("apps.public.scan.run_plate_scan") as run_scan:
            resp = client.post(SCAN_URL, {"event_type": "entry"})

        assert resp.status_code == 403
        run_scan.assert_not_called()

    def test_scope_cannot_change_after_activation(self, client, parking_lot):
        """A compromised lane browser cannot switch direction after activation."""
        _activate(client, event_type="entry", lot=parking_lot.name)

        with patch("apps.public.scan.run_plate_scan") as run_scan:
            response = client.post(
                SCAN_URL,
                {
                    "event_type": "exit",
                    "lot": parking_lot.name,
                    "kiosk_nonce": client.session["kiosk_scan_nonce"],
                },
            )

        assert response.status_code == 403
        run_scan.assert_not_called()

    def test_scan_nonce_cannot_be_replayed(self, client, lot_settings):
        """Captured requests and repeated image bytes must both be rejected."""
        _activate(client, event_type="entry", lot=lot_settings.lot.name)
        nonce = client.session["kiosk_scan_nonce"]
        image_bytes = _real_image_bytes()
        storage, factory = _mocks()
        with (
            patch("apps.dashboard.scan_core.default_storage", storage),
            patch("apps.dashboard.scan_core.get_pipeline", factory),
        ):
            first = client.post(
                SCAN_URL,
                {
                    "event_type": "entry",
                    "lot": lot_settings.lot.name,
                    "kiosk_nonce": nonce,
                    "image": _image(content=image_bytes),
                },
            )
            stale_nonce_replay = client.post(
                SCAN_URL,
                {
                    "event_type": "entry",
                    "lot": lot_settings.lot.name,
                    "kiosk_nonce": nonce,
                    "image": _image(content=image_bytes),
                },
            )
            fresh_nonce_image_replay = client.post(
                SCAN_URL,
                {
                    "event_type": "entry",
                    "lot": lot_settings.lot.name,
                    "kiosk_nonce": client.session["kiosk_scan_nonce"],
                    "image": _image(content=image_bytes),
                },
            )
            different_image = client.post(
                SCAN_URL,
                {
                    "event_type": "entry",
                    "lot": lot_settings.lot.name,
                    "kiosk_nonce": client.session["kiosk_scan_nonce"],
                    "image": _image(
                        content=_real_image_bytes("PNG"),
                        content_type="image/png",
                        name="plate.png",
                    ),
                },
            )
            alternating_replay = client.post(
                SCAN_URL,
                {
                    "event_type": "entry",
                    "lot": lot_settings.lot.name,
                    "kiosk_nonce": client.session["kiosk_scan_nonce"],
                    "image": _image(content=image_bytes),
                },
            )

        assert first.status_code == 200
        assert stale_nonce_replay.status_code == 403
        assert fresh_nonce_image_replay.status_code == 403
        assert different_image.status_code == 200
        assert alternating_replay.status_code == 403

    def test_stale_nonce_response_reissues_nonce_for_retry(
        self,
        client,
        lot_settings,
    ):
        """A lost successful response must not strand later kiosk retries."""
        _activate(client, event_type="entry", lot=lot_settings.lot.name)
        lost_response_nonce = client.session["kiosk_scan_nonce"]
        storage, factory = _mocks()
        with (
            patch("apps.dashboard.scan_core.default_storage", storage),
            patch("apps.dashboard.scan_core.get_pipeline", factory),
        ):
            client.post(
                SCAN_URL,
                {
                    "event_type": "entry",
                    "lot": lot_settings.lot.name,
                    "kiosk_nonce": lost_response_nonce,
                    "image": _image(content=_real_image_bytes()),
                },
            )
            stale_response = client.post(
                SCAN_URL,
                {
                    "event_type": "entry",
                    "lot": lot_settings.lot.name,
                    "kiosk_nonce": lost_response_nonce,
                    "image": _image(
                        content=_real_image_bytes("PNG"),
                        content_type="image/png",
                        name="retry.png",
                    ),
                },
            )
            replacement_nonce = stale_response.headers.get("X-Kiosk-Nonce", "")
            retry = client.post(
                SCAN_URL,
                {
                    "event_type": "entry",
                    "lot": lot_settings.lot.name,
                    "kiosk_nonce": replacement_nonce,
                    "image": _image(
                        content=_real_image_bytes("PNG"),
                        content_type="image/png",
                        name="retry.png",
                    ),
                },
            )

        assert stale_response.status_code == 403
        assert replacement_nonce
        assert replacement_nonce != lost_response_nonce
        assert retry.status_code == 200

    def test_failed_scan_allows_same_image_retry(self, client, lot_settings):
        """A rejected upload must not blacklist the same photo for the replay window."""
        _activate(client, event_type="entry", lot=lot_settings.lot.name)
        image_bytes = _real_image_bytes()
        with patch(
            "apps.public.scan.run_plate_scan",
            side_effect=[
                ScanOutcome(
                    "error",
                    error="Image could not be decoded safely.",
                    status=422,
                ),
                ScanOutcome(
                    "entry",
                    event_type="entry",
                    result=_result(),
                    session=SimpleNamespace(
                        user_id=None,
                        entry_time=timezone.now(),
                    ),
                ),
            ],
        ):
            failed = client.post(
                SCAN_URL,
                {
                    "event_type": "entry",
                    "lot": lot_settings.lot.name,
                    "kiosk_nonce": client.session["kiosk_scan_nonce"],
                    "image": _image(content=image_bytes),
                },
            )
            retry = client.post(
                SCAN_URL,
                {
                    "event_type": "entry",
                    "lot": lot_settings.lot.name,
                    "kiosk_nonce": client.session["kiosk_scan_nonce"],
                    "image": _image(content=image_bytes),
                },
            )

        assert failed.status_code == 422
        assert retry.status_code == 200
        assert retry.json()["outcome"] == "entry"
        digest = hashlib.sha256(image_bytes).hexdigest()
        assert KioskImageReplayDigest.objects.filter(digest=digest).count() == 1

    def test_activation_requires_csrf(self, settings):
        """A remote site must not be able to activate a browser through CSRF."""
        csrf_client = Client(enforce_csrf_checks=True)

        resp = csrf_client.post(
            ACTIVATE_URL,
            {"token": settings.KIOSK_ACTIVATION_TOKEN},
        )

        assert resp.status_code == 403


# ── Activated kiosk access ────────────────────────────────────────────────────


@pytest.mark.django_db
class TestKioskAccess:
    def test_anonymous_can_open_entry(self, client, lot_settings):
        resp = _post_scan(client, event_type="entry", image=_image())
        assert resp.status_code == 200
        data = resp.json()
        assert data["outcome"] == "entry"
        assert data["plate_text"] == "ABC123"
        assert ParkingSession.objects.filter(
            plate_text="ABC123", status="active"
        ).exists()

    def test_get_not_allowed(self, client):
        assert client.get(SCAN_URL).status_code == 405

    def test_activated_shell_loads_capability_once(self, client, parking_lot):
        """The shell must not pay for a duplicate capability lookup on every refresh."""
        _activate(client, lot=parking_lot.name)
        with CaptureQueriesContext(connection) as captured:
            response = client.get("/")

        capability_reads = [
            query["sql"]
            for query in captured.captured_queries
            if "parking_kioskdevicecapability" in query["sql"].lower()
            and query["sql"].lstrip().upper().startswith("SELECT")
            and "FOR UPDATE" not in query["sql"].upper()
        ]
        assert response.status_code == 200
        assert response.context["kiosk_activated"] is True
        assert len(capability_reads) == 1


@pytest.mark.django_db(transaction=True)
def test_scan_nonce_has_one_atomic_concurrent_winner(client, lot_settings):
    """A row lock must let only one parallel request consume a captured nonce."""
    _activate(client, event_type="entry", lot=lot_settings.lot.name)
    session_key = client.session.session_key
    nonce = client.session["kiosk_scan_nonce"]

    def consume():
        """Build an independent request/DB connection for the same browser."""
        close_old_connections()
        try:
            request = RequestFactory().post(
                SCAN_URL,
                {
                    "event_type": "entry",
                    "lot": lot_settings.lot.name,
                    "kiosk_nonce": nonce,
                },
            )
            request.session = SessionStore(session_key=session_key)
            return _consume_kiosk_request(request)
        finally:
            connection.close()

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: consume(), range(2)))

    assert sum(result is not None and result.accepted for result in results) == 1


# ── Privacy: never leak image route / event id / balance ──────────────────────


@pytest.mark.django_db
class TestKioskPrivacy:
    def test_entry_payload_omits_private_fields(self, client, lot_settings):
        data = _post_scan(client, event_type="entry", image=_image()).json()
        for leaked in ("image_url", "event_id", "balance", "user", "owner"):
            assert leaked not in data

    def test_exit_payload_omits_private_fields(self, client, parking_lot, lot_settings):
        ParkingSession.objects.create(
            plate_text="ABC123",
            lot=parking_lot,
            entry_time=timezone.now() - timedelta(minutes=90),
            status="active",
        )
        data = _post_scan(client, event_type="exit", image=_image()).json()
        assert data["outcome"] == "exit_matched"
        for leaked in ("image_url", "event_id", "balance", "user", "owner"):
            assert leaked not in data


# ── Billing: guest vs registered ──────────────────────────────────────────────


@pytest.mark.django_db
class TestKioskBilling:
    def _active(self, lot, user=None, plate_obj=None):
        return ParkingSession.objects.create(
            plate_text="ABC123",
            lot=lot,
            user=user,
            license_plate=plate_obj,
            entry_time=timezone.now() - timedelta(minutes=90),
            status="active",
        )

    def test_guest_exit_shows_amount_due(self, client, parking_lot, lot_settings):
        self._active(parking_lot)
        data = _post_scan(client, event_type="exit", image=_image()).json()
        assert data["charge_amount"] == "10.00"
        assert data["billed_to_account"] is False
        assert not Wallet.objects.exists()

    def test_registered_exit_bills_wallet(self, client, parking_lot, lot_settings):
        user = User.objects.create_user(username="drv", password="x", email="d@e.com")
        plate = LicensePlate.objects.create(user=user, plate_text="ABC123")
        credit_wallet(user, Decimal("20.00"), reference="kiosk-exit-credit")
        self._active(parking_lot, user=user, plate_obj=plate)

        data = _post_scan(client, event_type="exit", image=_image()).json()
        assert data["charge_amount"] == "10.00"
        assert data["billed_to_account"] is True
        wallet = Wallet.objects.get(user=user)
        assert wallet.balance == Decimal("10.00")
        assert wallet.transactions.filter(kind=WalletTransaction.Kind.CHARGE).exists()


# ── Input validation ──────────────────────────────────────────────────────────


@pytest.mark.django_db
class TestKioskValidation:
    def test_missing_event_type_is_400(self, client, lot_settings):
        assert _post_scan(client, image=_image()).status_code == 400

    def test_missing_image_is_400(self, client, lot_settings):
        assert _post_scan(client, event_type="entry").status_code == 400

    def test_unsupported_content_type_is_415(self, client, lot_settings):
        resp = _post_scan(
            client, event_type="entry", image=_image(content_type="image/gif")
        )
        assert resp.status_code == 415

    def test_htmx_error_renders_swappable_html(self, client, lot_settings):
        _activate(client, lot=lot_settings.lot.name)
        storage, factory = _mocks()
        with (
            patch("apps.dashboard.scan_core.default_storage", storage),
            patch("apps.dashboard.scan_core.get_pipeline", factory),
        ):
            resp = client.post(
                SCAN_URL,
                {
                    "event_type": "entry",
                    "lot": lot_settings.lot.name,
                    "kiosk_nonce": client.session["kiosk_scan_nonce"],
                },
                HTTP_HX_REQUEST="true",
            )
        # HTMX does not swap 4xx, so expected errors render as HTML 200.
        assert resp.status_code == 200

    def test_commit_failure_deletes_upload(self, client, lot_settings):
        """A file must not survive when transaction commit rejects its event."""
        storage, factory = _mocks()
        atomic = MagicMock()
        atomic.return_value.__exit__.side_effect = RuntimeError("commit failed")
        _activate(client, lot=lot_settings.lot.name)

        transaction_proxy = MagicMock()
        transaction_proxy.atomic = atomic
        with (
            patch("apps.dashboard.scan_core.default_storage", storage),
            patch("apps.dashboard.scan_core.get_pipeline", factory),
            patch("apps.dashboard.scan_core.transaction", transaction_proxy),
        ):
            response = client.post(
                SCAN_URL,
                {
                    "event_type": "entry",
                    "lot": lot_settings.lot.name,
                    "kiosk_nonce": client.session["kiosk_scan_nonce"],
                    "image": _image(),
                },
            )

        assert response.status_code == 500
        storage.delete.assert_called_once_with(STORED_NAME)


class TestProcessingPath:
    """Keep remote-storage scratch copies inside a private trusted directory."""

    def test_rejects_symlink_temp_root(self, settings, tmp_path):
        """A local attacker must not redirect CV scratch files through a symlink."""
        real_root = tmp_path / "real"
        real_root.mkdir()
        linked_root = tmp_path / "linked"
        linked_root.symlink_to(real_root, target_is_directory=True)
        settings.CV_PROCESSING_TEMP_ROOT = str(linked_root)
        storage = MagicMock()
        storage.path.side_effect = NotImplementedError

        with pytest.raises(UnsafeImagePathError):
            with _processing_path(storage, STORED_NAME):
                pass

        storage.open.assert_not_called()

    def test_repairs_temp_root_permissions(self, settings, tmp_path):
        """Scratch files must not be exposed by an overly broad directory mode."""
        scratch_root = tmp_path / "scratch"
        scratch_root.mkdir(mode=0o777)
        os.chmod(scratch_root, 0o777)
        settings.CV_PROCESSING_TEMP_ROOT = str(scratch_root)
        storage = MagicMock()
        storage.path.side_effect = NotImplementedError
        storage.open.return_value.__enter__.return_value = io.BytesIO(b"image")

        with _processing_path(storage, STORED_NAME) as local_path:
            assert Path(local_path).parent == scratch_root
            assert scratch_root.stat().st_mode & 0o777 == 0o700

        assert not Path(local_path).exists()


# ── Rate limiting ─────────────────────────────────────────────────────────────


@pytest.mark.django_db
class TestKioskRateLimit:
    def test_exceeding_limit_returns_429(self, client, lot_settings):
        # The limiter allows 20/60s per IP; the 21st request is rejected before the
        # view runs. Minimal posts are fine — the decorator counts every hit.
        last = None
        for _ in range(21):
            last = client.post(SCAN_URL, {"event_type": "entry"})
        assert last.status_code == 429
