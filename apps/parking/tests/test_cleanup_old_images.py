"""
Tests for the cleanup_old_images management command.

WHY THIS FILE EXISTS:
  Image retention is part of the privacy boundary for uploaded plate photos.
  The command must preserve unresolved review-queue images until correction,
  because operators need that evidence to verify low-confidence and unmatched
  detections. Once reviewed, those images follow the normal lot retention policy.

  Critically, the command purges image FILES and clears the image FIELD while
  KEEPING the PlateDetectionEvent row — so audit data (plate text, confidence,
  timestamps) survives the cleanup. These tests verify that invariant.
"""

from datetime import timedelta
from io import StringIO

import pytest
from django.core.files.base import ContentFile
from django.core.management import call_command
from django.utils import timezone

from apps.parking.models import (
    LotSettings,
    ParkingLot,
    ParkingSession,
    PlateDetectionEvent,
)


@pytest.mark.django_db
class TestCleanupOldImages:
    """Regression tests for lot-scoped image cleanup selection."""

    def test_dry_run_skips_unresolved_sessionless_lot_events(self):
        """Unmatched events keep their evidence until an operator corrects them."""
        lot = ParkingLot.objects.create(name="Cleanup Lot")
        LotSettings.objects.create(lot=lot, image_retention_days=1)
        event = PlateDetectionEvent.objects.create(
            session=None,
            lot=lot,
            image="plates/sessionless-old.jpg",
            raw_plate_text="NOENTRY",
            confidence_score=0.8,
            event_type="exit",
            is_low_confidence=True,
        )
        PlateDetectionEvent.objects.filter(pk=event.pk).update(
            timestamp=timezone.now() - timedelta(days=2)
        )

        stdout = StringIO()
        call_command("cleanup_old_images", "--dry-run", stdout=stdout)

        assert '"Cleanup Lot": no eligible images older than 1 days.' in (
            stdout.getvalue()
        )
        event.refresh_from_db()
        assert event.image.name == "plates/sessionless-old.jpg"

    def test_cleanup_does_not_delete_unresolved_review_image(self):
        """The live cleanup path must not touch evidence still pending review."""
        from unittest.mock import patch

        lot = ParkingLot.objects.create(name="Pending Review Lot")
        LotSettings.objects.create(lot=lot, image_retention_days=1)
        event = PlateDetectionEvent.objects.create(
            session=None,
            lot=lot,
            image="plates/pending-review-old.jpg",
            raw_plate_text="",
            confidence_score=0.0,
            event_type="entry",
            is_low_confidence=True,
        )
        PlateDetectionEvent.objects.filter(pk=event.pk).update(
            timestamp=timezone.now() - timedelta(days=2)
        )

        with patch(
            "django.core.files.storage.FileSystemStorage.delete"
        ) as delete_image:
            call_command("cleanup_old_images")

        delete_image.assert_not_called()
        event.refresh_from_db()
        assert event.image.name == "plates/pending-review-old.jpg"

    def test_dry_run_counts_corrected_sessionless_lot_events(self):
        """A reviewed sessionless event returns to the normal retention policy."""
        lot = ParkingLot.objects.create(name="Corrected Cleanup Lot")
        LotSettings.objects.create(lot=lot, image_retention_days=1)
        event = PlateDetectionEvent.objects.create(
            session=None,
            lot=lot,
            image="plates/sessionless-corrected-old.jpg",
            raw_plate_text="NOENTRY",
            confidence_score=0.8,
            event_type="exit",
            is_low_confidence=True,
            manually_corrected=True,
            corrected_plate="ABC123",
        )
        PlateDetectionEvent.objects.filter(pk=event.pk).update(
            timestamp=timezone.now() - timedelta(days=2)
        )

        stdout = StringIO()
        call_command("cleanup_old_images", "--dry-run", stdout=stdout)

        assert '"Corrected Cleanup Lot": would clear 1 image(s)' in stdout.getvalue()

    def test_dry_run_skips_unresolved_low_confidence_session_event(self):
        """A linked low-confidence event also stays protected until review."""
        lot = ParkingLot.objects.create(name="Low Confidence Cleanup Lot")
        LotSettings.objects.create(lot=lot, image_retention_days=1)
        session = ParkingSession.objects.create(
            plate_text="LOW123",
            lot=lot,
            entry_time=timezone.now() - timedelta(days=3),
            status="active",
        )
        event = PlateDetectionEvent.objects.create(
            session=session,
            lot=lot,
            image="plates/low-confidence-old.jpg",
            raw_plate_text="LOWI23",
            confidence_score=0.2,
            event_type="entry",
            is_low_confidence=True,
        )
        PlateDetectionEvent.objects.filter(pk=event.pk).update(
            timestamp=timezone.now() - timedelta(days=2)
        )

        stdout = StringIO()
        call_command("cleanup_old_images", "--dry-run", stdout=stdout)

        assert (
            '"Low Confidence Cleanup Lot": no eligible images older than 1 days.'
            in stdout.getvalue()
        )
        event.refresh_from_db()
        assert event.image.name == "plates/low-confidence-old.jpg"

    def test_dry_run_counts_session_event_with_null_lot(self):
        """Legacy session-linked events with lot=None still clean via session.lot."""
        lot = ParkingLot.objects.create(name="Legacy Cleanup Lot")
        LotSettings.objects.create(lot=lot, image_retention_days=1)
        session = ParkingSession.objects.create(
            plate_text="LEGACY1",
            lot=lot,
            entry_time=timezone.now() - timedelta(days=3),
            status="active",
        )
        event = PlateDetectionEvent.objects.create(
            session=session,
            lot=None,
            image="plates/legacy-session-old.jpg",
            raw_plate_text="LEGACY1",
            confidence_score=0.9,
            event_type="entry",
        )
        PlateDetectionEvent.objects.filter(pk=event.pk).update(
            timestamp=timezone.now() - timedelta(days=2)
        )

        stdout = StringIO()
        call_command("cleanup_old_images", "--dry-run", stdout=stdout)

        assert '"Legacy Cleanup Lot": would clear 1 image(s)' in stdout.getvalue()

    def test_old_image_is_cleared_and_record_kept(self):
        """
        The command purges the image file from storage and clears the field,
        but the PlateDetectionEvent row itself must survive as an audit record.

        WHY override_settings(MEDIA_ROOT): the container's /app/media/plates
        directory is owned by root and not writable by the test runner (appuser).
        A temp directory lets us write a real file without needing extra container
        permissions, while still exercising the exact same storage code path.
        """
        import tempfile

        from django.test import override_settings

        with tempfile.TemporaryDirectory() as tmp_media:
            with override_settings(MEDIA_ROOT=tmp_media):
                # Re-import default_storage inside the override so it picks up
                # the new MEDIA_ROOT. Django caches the storage instance on the
                # module, so we access it fresh via the module reference.
                from django.core.files.storage import default_storage as storage

                lot = ParkingLot.objects.create(name="Real File Lot")
                LotSettings.objects.create(lot=lot, image_retention_days=1)

                # Save a real file so we can assert it is physically deleted.
                # ContentFile wraps raw bytes — storage.save() picks a unique
                # path if 'plates/old.jpg' already exists, so capture the name.
                name = storage.save("plates/old.jpg", ContentFile(b"fake-bytes"))

                event = PlateDetectionEvent.objects.create(
                    session=None,
                    lot=lot,
                    image=name,
                    raw_plate_text="REALFILE",
                    confidence_score=0.9,
                    event_type="entry",
                    manually_corrected=True,
                    corrected_plate="REALFILE",
                )
                PlateDetectionEvent.objects.filter(pk=event.pk).update(
                    timestamp=timezone.now() - timedelta(days=2)
                )

                try:
                    call_command("cleanup_old_images")

                    # File must be gone from storage.
                    assert storage.exists(name) is False

                    # Field must be cleared on the DB row.
                    event.refresh_from_db()
                    assert not event.image

                    # The row itself must still exist — audit record preserved.
                    assert (
                        PlateDetectionEvent.objects.filter(pk=event.pk).exists() is True
                    )
                finally:
                    # Safety net: remove the file if the command failed to delete it.
                    if storage.exists(name):
                        storage.delete(name)

    def test_new_image_is_kept(self):
        """Events newer than the retention cutoff must not be touched."""
        lot = ParkingLot.objects.create(name="New Image Lot")
        LotSettings.objects.create(lot=lot, image_retention_days=30)
        event = PlateDetectionEvent.objects.create(
            session=None,
            lot=lot,
            image="plates/new.jpg",
            raw_plate_text="NEWPLATE",
            confidence_score=0.9,
            event_type="entry",
        )
        # 1 day old — well inside the 30-day window.
        PlateDetectionEvent.objects.filter(pk=event.pk).update(
            timestamp=timezone.now() - timedelta(days=1)
        )

        call_command("cleanup_old_images")

        event.refresh_from_db()
        # Field must be unchanged.
        assert event.image.name == "plates/new.jpg"

    def test_null_retention_keeps_everything(self):
        """
        Lots with image_retention_days=None are skipped entirely — the operator
        has opted for indefinite retention and the command must not touch them.
        """
        lot = ParkingLot.objects.create(name="Forever Lot")
        LotSettings.objects.create(lot=lot, image_retention_days=None)
        event = PlateDetectionEvent.objects.create(
            session=None,
            lot=lot,
            image="plates/keep-forever.jpg",
            raw_plate_text="FOREVER",
            confidence_score=0.9,
            event_type="entry",
        )
        # 999 days old — ancient, but retention is null so must survive.
        PlateDetectionEvent.objects.filter(pk=event.pk).update(
            timestamp=timezone.now() - timedelta(days=999)
        )

        stdout = StringIO()
        call_command("cleanup_old_images", stdout=stdout)

        # Command should report nothing to do (no lots with retention configured).
        assert "Nothing to do" in stdout.getvalue()

        # Row and image field must be untouched.
        event.refresh_from_db()
        assert event.image.name == "plates/keep-forever.jpg"
        assert PlateDetectionEvent.objects.filter(pk=event.pk).exists() is True

    def test_dry_run_reports_but_does_not_clear(self):
        """
        --dry-run must report what would be cleared without making any changes
        to files or the image field.
        """
        lot = ParkingLot.objects.create(name="Dry Run Lot")
        LotSettings.objects.create(lot=lot, image_retention_days=1)
        event = PlateDetectionEvent.objects.create(
            session=None,
            lot=lot,
            image="plates/dry.jpg",
            raw_plate_text="DRYRUN",
            confidence_score=0.9,
            event_type="entry",
            manually_corrected=True,
            corrected_plate="DRYRUN",
        )
        PlateDetectionEvent.objects.filter(pk=event.pk).update(
            timestamp=timezone.now() - timedelta(days=2)
        )

        stdout = StringIO()
        call_command("cleanup_old_images", "--dry-run", stdout=stdout)

        # Dry-run output must mention the impending clear.
        assert "would clear 1 image(s)" in stdout.getvalue()

        # Nothing must have actually changed.
        event.refresh_from_db()
        assert event.image.name == "plates/dry.jpg"
        assert PlateDetectionEvent.objects.filter(pk=event.pk).exists() is True

    def test_session_record_remains_after_cleanup(self):
        """
        Clearing an event's image must not cascade to its linked ParkingSession.
        Both the session row and the event row survive; only the image field is emptied.
        """
        lot = ParkingLot.objects.create(name="Session Lot")
        LotSettings.objects.create(lot=lot, image_retention_days=1)
        session = ParkingSession.objects.create(
            plate_text="SESS1",
            lot=lot,
            entry_time=timezone.now() - timedelta(days=3),
            status="active",
        )
        event = PlateDetectionEvent.objects.create(
            session=session,
            lot=lot,
            image="plates/sess.jpg",
            raw_plate_text="SESS1",
            confidence_score=0.9,
            event_type="entry",
        )
        PlateDetectionEvent.objects.filter(pk=event.pk).update(
            timestamp=timezone.now() - timedelta(days=2)
        )

        call_command("cleanup_old_images")

        # Session must survive untouched.
        assert ParkingSession.objects.filter(pk=session.pk).exists() is True

        # Event row must survive with image field cleared.
        assert PlateDetectionEvent.objects.filter(pk=event.pk).exists() is True
        event.refresh_from_db()
        assert not event.image

    def test_storage_delete_failure_keeps_field_for_retry(self):
        """
        If purging a file raises (e.g. a remote-storage outage), the command
        must log and skip that row WITHOUT clearing its field or aborting —
        so the row is retried on the next run rather than orphaning the file.
        """
        from unittest.mock import patch

        lot = ParkingLot.objects.create(name="Flaky Storage Lot")
        LotSettings.objects.create(lot=lot, image_retention_days=1)
        event = PlateDetectionEvent.objects.create(
            session=None,
            lot=lot,
            image="plates/flaky.jpg",
            raw_plate_text="FLAKY",
            confidence_score=0.9,
            event_type="entry",
            manually_corrected=True,
            corrected_plate="FLAKY",
        )
        PlateDetectionEvent.objects.filter(pk=event.pk).update(
            timestamp=timezone.now() - timedelta(days=2)
        )

        # Force the storage delete to fail for this run.
        with patch(
            "django.core.files.storage.FileSystemStorage.delete",
            side_effect=OSError("simulated storage outage"),
        ):
            call_command("cleanup_old_images")  # must NOT raise

        # Field retained (so exclude(image='') re-selects it next run); row kept.
        event.refresh_from_db()
        assert event.image.name == "plates/flaky.jpg"
        assert PlateDetectionEvent.objects.filter(pk=event.pk).exists() is True
