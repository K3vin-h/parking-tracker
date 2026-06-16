"""
Tests for the cleanup_old_images management command.

WHY THIS FILE EXISTS:
  Image retention is part of the privacy boundary for uploaded plate photos.
  The command must include both normal session-linked events and unmatched
  review-queue events, because both store vehicle images under the same lot
  retention policy.
"""

from datetime import timedelta
from io import StringIO

import pytest
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

    def test_dry_run_counts_sessionless_lot_events(self):
        """Unmatched exit events with session=None still follow lot retention."""
        lot = ParkingLot.objects.create(name='Cleanup Lot')
        LotSettings.objects.create(lot=lot, image_retention_days=1)
        event = PlateDetectionEvent.objects.create(
            session=None,
            lot=lot,
            image='plates/sessionless-old.jpg',
            raw_plate_text='NOENTRY',
            confidence_score=0.8,
            event_type='exit',
            is_low_confidence=True,
        )
        PlateDetectionEvent.objects.filter(pk=event.pk).update(
            timestamp=timezone.now() - timedelta(days=2)
        )

        stdout = StringIO()
        call_command('cleanup_old_images', '--dry-run', stdout=stdout)

        assert '"Cleanup Lot": would delete 1 event(s)' in stdout.getvalue()

    def test_dry_run_counts_session_event_with_null_lot(self):
        """Legacy session-linked events with lot=None still clean via session.lot."""
        lot = ParkingLot.objects.create(name='Legacy Cleanup Lot')
        LotSettings.objects.create(lot=lot, image_retention_days=1)
        session = ParkingSession.objects.create(
            plate_text='LEGACY1',
            lot=lot,
            entry_time=timezone.now() - timedelta(days=3),
            status='active',
        )
        event = PlateDetectionEvent.objects.create(
            session=session,
            lot=None,
            image='plates/legacy-session-old.jpg',
            raw_plate_text='LEGACY1',
            confidence_score=0.9,
            event_type='entry',
        )
        PlateDetectionEvent.objects.filter(pk=event.pk).update(
            timestamp=timezone.now() - timedelta(days=2)
        )

        stdout = StringIO()
        call_command('cleanup_old_images', '--dry-run', stdout=stdout)

        assert '"Legacy Cleanup Lot": would delete 1 event(s)' in stdout.getvalue()
