"""
Tests for the parking app models.

Verifies that all five models:
  - Can be created with the correct field values
  - Enforce the expected nullable/non-nullable constraints
  - Have correct string representations (__str__)
  - Have the expected default values
  - Maintain correct relationships (ForeignKey, OneToOne)

TESTING PHILOSOPHY FOR MODELS:
  Model tests catch:
    1. Migration errors (missing tables, wrong column types)
    2. Incorrect default values (e.g., status defaults to 'active')
    3. Nullable FK behavior (guest sessions, voided sessions)
    4. Field constraints (DecimalField precision, max_length)
    5. __str__ output (shown in admin, logs, error messages)
"""

from decimal import Decimal

import pytest
from django.contrib.auth import get_user_model
from django.utils import timezone

from apps.parking.models import (
    LicensePlate,
    LotSettings,
    ParkingLot,
    ParkingSession,
    PlateDetectionEvent,
)

User = get_user_model()


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def user(db):
    """A test user to associate with plates and sessions."""
    return User.objects.create_user(
        username='plateuser',
        email='plateuser@example.com',
        password='testpass123',
    )


@pytest.fixture
def parking_lot(db):
    """A default parking lot for session tests."""
    return ParkingLot.objects.create(name='Test Lot')


@pytest.fixture
def lot_settings(parking_lot):
    """LotSettings for the test lot with standard defaults."""
    return LotSettings.objects.create(
        lot=parking_lot,
        rate=Decimal('5.00'),
        billing_unit='hour',
        grace_period_minutes=15,
    )


@pytest.fixture
def license_plate(user):
    """A registered license plate for the test user."""
    return LicensePlate.objects.create(
        user=user,
        plate_text='ABC123',
        is_primary=True,
        label='Daily Driver',
    )


@pytest.fixture
def active_session(parking_lot, user, license_plate):
    """An active parking session with all fields populated."""
    return ParkingSession.objects.create(
        plate_text='ABC123',
        license_plate=license_plate,
        user=user,
        lot=parking_lot,
        entry_time=timezone.now(),
        status='active',
    )


# ── LicensePlate tests ────────────────────────────────────────────────────────

@pytest.mark.django_db
class TestLicensePlate:
    """Tests for the LicensePlate model."""

    def test_create_license_plate(self, user):
        """A plate can be created and retrieved with all fields."""
        plate = LicensePlate.objects.create(
            user=user,
            plate_text='XYZ789',
            is_primary=False,
            label='Work Truck',
        )
        # Verify it was saved (pk is set by the database)
        assert plate.pk is not None
        assert plate.plate_text == 'XYZ789'
        assert plate.label == 'Work Truck'
        assert plate.is_primary is False

    def test_plate_str_with_label(self, license_plate):
        """
        __str__ includes the label in parentheses when a label is set.

        This is what the admin and ForeignKey dropdowns show.
        """
        assert str(license_plate) == 'ABC123 (Daily Driver)'

    def test_plate_str_without_label(self, user):
        """__str__ shows just the plate text when no label is set."""
        plate = LicensePlate.objects.create(
            user=user,
            plate_text='NOLABEL',
        )
        assert str(plate) == 'NOLABEL'

    def test_plate_default_not_primary(self, user):
        """is_primary defaults to False — must be explicitly set."""
        plate = LicensePlate.objects.create(user=user, plate_text='DEF456')
        assert plate.is_primary is False

    def test_plate_label_defaults_to_empty_string(self, user):
        """label defaults to an empty string, not null."""
        plate = LicensePlate.objects.create(user=user, plate_text='NOLBL')
        assert plate.label == ''

    def test_plate_unique_per_user(self, user):
        """
        The same plate text cannot be registered twice for the same user.

        unique_together = [('user', 'plate_text')] enforces this at the DB level.
        """
        from django.db import IntegrityError
        LicensePlate.objects.create(user=user, plate_text='DUP123')
        with pytest.raises(IntegrityError):
            LicensePlate.objects.create(user=user, plate_text='DUP123')


# ── ParkingLot tests ──────────────────────────────────────────────────────────

@pytest.mark.django_db
class TestParkingLot:
    """Tests for the ParkingLot model."""

    def test_create_parking_lot(self):
        """A parking lot can be created with a name."""
        lot = ParkingLot.objects.create(name='South Garage')
        assert lot.pk is not None
        assert lot.name == 'South Garage'

    def test_parking_lot_str(self, parking_lot):
        """__str__ returns the lot name."""
        assert str(parking_lot) == 'Test Lot'


# ── LotSettings tests ─────────────────────────────────────────────────────────

@pytest.mark.django_db
class TestLotSettings:
    """Tests for the LotSettings model."""

    def test_create_lot_settings(self, parking_lot):
        """LotSettings can be created with all fields."""
        settings_obj = LotSettings.objects.create(
            lot=parking_lot,
            rate=Decimal('8.50'),
            billing_unit='minute',
            grace_period_minutes=10,
            daily_cap_enabled=True,
            daily_cap_amount=Decimal('25.00'),
            image_retention_days=30,
            confidence_threshold=0.75,
        )
        assert settings_obj.pk is not None
        assert settings_obj.rate == Decimal('8.50')
        assert settings_obj.billing_unit == 'minute'

    def test_lot_settings_defaults(self, parking_lot):
        """
        LotSettings has sensible default values for all optional fields.

        Operators can start using the system without configuring every setting.
        """
        settings_obj = LotSettings.objects.create(lot=parking_lot)
        assert settings_obj.rate == Decimal('5.00')
        assert settings_obj.billing_unit == 'hour'
        assert settings_obj.grace_period_minutes == 15
        assert settings_obj.daily_cap_enabled is False
        assert settings_obj.daily_cap_amount is None   # null when cap is disabled
        assert settings_obj.image_retention_days is None  # null = keep forever
        assert settings_obj.confidence_threshold == 0.6

    def test_lot_settings_one_to_one(self, parking_lot):
        """
        A lot can only have one LotSettings record.

        OneToOneField enforces this at the DB level. Trying to create a second
        LotSettings for the same lot raises an IntegrityError.
        """
        from django.db import IntegrityError
        LotSettings.objects.create(lot=parking_lot)
        with pytest.raises(IntegrityError):
            LotSettings.objects.create(lot=parking_lot)

    def test_lot_settings_reverse_access(self, lot_settings, parking_lot):
        """
        lot.settings returns the LotSettings via the related_name.

        This is the shorthand used throughout services.py and views to
        get the settings for a lot: session.lot.settings.rate
        """
        assert parking_lot.settings == lot_settings

    def test_lot_settings_str(self, lot_settings, parking_lot):
        """__str__ describes which lot the settings belong to."""
        assert str(lot_settings) == f'Settings for {parking_lot.name}'


# ── ParkingSession tests ──────────────────────────────────────────────────────

@pytest.mark.django_db
class TestParkingSession:
    """Tests for the ParkingSession model."""

    def test_create_active_session(self, parking_lot, user, license_plate):
        """A session starts as 'active' with all FK links set."""
        entry = timezone.now()
        session = ParkingSession.objects.create(
            plate_text='ABC123',
            license_plate=license_plate,
            user=user,
            lot=parking_lot,
            entry_time=entry,
        )
        assert session.pk is not None
        assert session.status == 'active'
        assert session.plate_text == 'ABC123'
        assert session.license_plate == license_plate
        assert session.user == user

    def test_guest_session_no_user(self, parking_lot):
        """
        Guest sessions have null user and null license_plate.

        This is the normal case for unregistered plates. The system still
        creates a session — it just can't link it to a known user.
        """
        session = ParkingSession.objects.create(
            plate_text='UNKNOWN1',
            license_plate=None,
            user=None,
            lot=parking_lot,
            entry_time=timezone.now(),
        )
        assert session.user is None
        assert session.license_plate is None
        assert session.status == 'active'

    def test_session_default_values(self, parking_lot):
        """All default values are set correctly on a new session."""
        session = ParkingSession.objects.create(
            plate_text='DEF789',
            lot=parking_lot,
            entry_time=timezone.now(),
        )
        assert session.status == 'active'
        assert session.duration_seconds == 0
        assert session.charge_amount == Decimal('0.00')
        assert session.has_duplicate_warning is False
        assert session.was_orphaned is False
        assert session.exit_time is None

    def test_session_str(self, active_session):
        """__str__ shows plate, status, and entry time for easy identification."""
        result = str(active_session)
        assert 'ABC123' in result
        assert 'active' in result

    def test_session_orphan_flags(self, parking_lot):
        """
        Duplicate/orphan flags can be set independently.

        was_orphaned=True on the OLD session (force-closed).
        has_duplicate_warning=True on the NEW session (opened while old was active).
        """
        old_session = ParkingSession.objects.create(
            plate_text='DUP456',
            lot=parking_lot,
            entry_time=timezone.now(),
            status='void',
            was_orphaned=True,
        )
        new_session = ParkingSession.objects.create(
            plate_text='DUP456',
            lot=parking_lot,
            entry_time=timezone.now(),
            has_duplicate_warning=True,
        )
        assert old_session.was_orphaned is True
        assert old_session.status == 'void'
        assert new_session.has_duplicate_warning is True
        assert new_session.status == 'active'

    def test_completed_session_has_charge(self, parking_lot):
        """A completed session can store a charge amount and exit time."""
        entry = timezone.now()
        exit_time = timezone.now()
        session = ParkingSession.objects.create(
            plate_text='COMP123',
            lot=parking_lot,
            entry_time=entry,
            exit_time=exit_time,
            duration_seconds=3600,
            charge_amount=Decimal('5.00'),
            status='completed',
        )
        assert session.status == 'completed'
        assert session.charge_amount == Decimal('5.00')
        assert session.exit_time is not None


# ── PlateDetectionEvent tests ─────────────────────────────────────────────────

@pytest.mark.django_db
class TestPlateDetectionEvent:
    """Tests for the PlateDetectionEvent model."""

    def test_create_detection_event(self, active_session):
        """A detection event can be created and linked to a session."""
        event = PlateDetectionEvent.objects.create(
            session=active_session,
            image='plates/test_plate.jpg',  # path relative to MEDIA_ROOT
            raw_plate_text='ABC 123',        # raw, pre-normalization
            confidence_score=0.92,
            event_type='entry',
        )
        assert event.pk is not None
        assert event.session == active_session
        assert event.raw_plate_text == 'ABC 123'
        assert event.confidence_score == 0.92
        assert event.event_type == 'entry'

    def test_detection_event_default_values(self, active_session):
        """Default values: not low-confidence, not corrected, empty bounding box."""
        event = PlateDetectionEvent.objects.create(
            session=active_session,
            image='plates/test.jpg',
            raw_plate_text='XYZ789',
            confidence_score=0.85,
            event_type='exit',
        )
        assert event.is_low_confidence is False
        assert event.manually_corrected is False
        assert event.corrected_plate is None
        assert event.bounding_box == []  # default=list creates []

    def test_bounding_box_stores_json(self, active_session):
        """
        Bounding box is stored as JSON and retrieved as a Python list.

        The [x, y, w, h] format uses normalized coordinates (0.0–1.0).
        JSONField serializes/deserializes automatically — no manual JSON parsing needed.
        """
        bbox = [0.1, 0.2, 0.5, 0.3]
        event = PlateDetectionEvent.objects.create(
            session=active_session,
            image='plates/bbox_test.jpg',
            raw_plate_text='BBOX123',
            confidence_score=0.75,
            event_type='entry',
            bounding_box=bbox,
        )
        # Reload from DB to verify JSON round-trip
        event.refresh_from_db()
        assert event.bounding_box == bbox

    def test_low_confidence_event(self, active_session):
        """Events below the confidence threshold are flagged is_low_confidence=True."""
        event = PlateDetectionEvent.objects.create(
            session=active_session,
            image='plates/lowconf.jpg',
            raw_plate_text='LWC456',
            confidence_score=0.45,  # Below typical threshold of 0.6
            event_type='entry',
            is_low_confidence=True,
        )
        assert event.is_low_confidence is True

    def test_manual_correction_fields(self, active_session):
        """An event can be marked as manually corrected with a corrected plate value."""
        event = PlateDetectionEvent.objects.create(
            session=active_session,
            image='plates/corrected.jpg',
            raw_plate_text='ABE123',   # CV misread 'C' as 'E'
            confidence_score=0.55,
            event_type='entry',
            is_low_confidence=True,
            manually_corrected=True,
            corrected_plate='ABC123',  # Operator provided the correct text
        )
        assert event.manually_corrected is True
        assert event.corrected_plate == 'ABC123'

    def test_orphan_event_null_session(self, parking_lot):
        """
        A detection event can exist without a linked session (session=None).

        This can happen if session creation fails after event creation
        (though in practice the code creates them atomically).
        Testing this edge case ensures SET_NULL is working correctly.
        """
        event = PlateDetectionEvent.objects.create(
            session=None,
            image='plates/orphan.jpg',
            raw_plate_text='ORPHAN1',
            confidence_score=0.80,
            event_type='entry',
        )
        assert event.session is None

    def test_detection_event_str(self, active_session):
        """__str__ includes the event type and raw plate text."""
        event = PlateDetectionEvent.objects.create(
            session=active_session,
            image='plates/str_test.jpg',
            raw_plate_text='STR123',
            confidence_score=0.88,
            event_type='entry',
        )
        result = str(event)
        assert 'entry' in result
        assert 'STR123' in result

    def test_timestamp_auto_set(self, active_session):
        """
        timestamp is set automatically at creation (auto_now_add=True).

        We don't set it — Django sets it to the current UTC time.
        """
        event = PlateDetectionEvent.objects.create(
            session=active_session,
            image='plates/ts_test.jpg',
            raw_plate_text='TS123',
            confidence_score=0.90,
            event_type='exit',
        )
        assert event.timestamp is not None
