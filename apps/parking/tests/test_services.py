"""
Tests for the parking session & billing services (apps/parking/services.py).

WHAT THIS COVERS:
  - normalize_plate: the canonical matching key (whitespace/case/hyphens).
  - calculate_charge: every billing branch (grace, per-hour, per-minute, cap)
    and the boundary cases where ceil() flips to the next unit.
  - handle_entry: guest vs registered, low-confidence flagging against the
    per-lot threshold, and orphan/duplicate voiding.
  - handle_exit: normal completion + billing, and the exit-without-entry
    review-queue path.
  - correct_plate: operator correction with registration re-linking.
  - DB constraint regressions: voided sessions carry no charge; completed
    sessions have exit_time strictly after entry_time.

STYLE: mirrors test_models.py — @pytest.fixture, class-based
@pytest.mark.django_db, plain assert, Decimal('X.XX') for money.
"""

from datetime import timedelta
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
from apps.parking.services import (
    calculate_charge,
    correct_plate,
    handle_entry,
    handle_exit,
    normalize_plate,
)

User = get_user_model()


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def user(db):
    """A test user to own registered plates."""
    return User.objects.create_user(
        username='svcuser',
        email='svcuser@example.com',
        password='testpass123',
    )


@pytest.fixture
def parking_lot(db):
    """A parking lot for entry/exit tests."""
    return ParkingLot.objects.create(name='Service Lot')


@pytest.fixture
def lot_settings(parking_lot):
    """Standard settings ($5/hr, 15-min grace, 0.6 threshold) for parking_lot."""
    return LotSettings.objects.create(
        lot=parking_lot,
        rate=Decimal('5.00'),
        billing_unit='hour',
        grace_period_minutes=15,
        confidence_threshold=0.6,
    )


@pytest.fixture
def license_plate(user):
    """A registered plate 'ABC123' owned by the test user."""
    return LicensePlate.objects.create(
        user=user,
        plate_text='ABC123',
        is_primary=True,
        label='Daily Driver',
    )


# Image is stored as a path string relative to MEDIA_ROOT — matches the
# convention in test_models.py and avoids writing files during tests.
PLATE_IMAGE = 'plates/svc_test.jpg'


def _settings(**overrides) -> LotSettings:
    """Build an UNSAVED LotSettings for pure calculate_charge tests.

    calculate_charge only reads attributes, so it needs no DB row.
    """
    defaults = dict(
        rate=Decimal('5.00'),
        billing_unit='hour',
        grace_period_minutes=15,
        daily_cap_enabled=False,
        daily_cap_amount=None,
    )
    defaults.update(overrides)
    return LotSettings(**defaults)


# ── normalize_plate ─────────────────────────────────────────────────────────

@pytest.mark.unit
class TestNormalizePlate:
    """Tests for the plate normalization key function (no DB)."""

    def test_strips_spaces_and_uppercases(self):
        """'abc 123' collapses to 'ABC123'."""
        assert normalize_plate('abc 123') == 'ABC123'

    def test_preserves_hyphens(self):
        """Only whitespace is removed — hyphens are kept for exact matching."""
        assert normalize_plate('  ab-12 ') == 'AB-12'

    def test_strips_all_whitespace_kinds(self):
        """Tabs and newlines are whitespace too and get stripped."""
        assert normalize_plate('a\tb c\n') == 'ABC'

    def test_already_normalized_passthrough(self):
        """A clean plate is returned unchanged."""
        assert normalize_plate('XYZ789') == 'XYZ789'

    def test_empty_input_returns_empty(self):
        """None/empty/whitespace-only normalize to '' without raising."""
        assert normalize_plate('') == ''
        assert normalize_plate(None) == ''
        assert normalize_plate('   ') == ''


# ── calculate_charge ─────────────────────────────────────────────────────────

@pytest.mark.unit
class TestCalculateCharge:
    """Tests for the billing math (pure, Decimal-only)."""

    def _span(self, minutes=0, seconds=0):
        """Return (entry, exit) timestamps spanning the given duration."""
        entry = timezone.now()
        return entry, entry + timedelta(minutes=minutes, seconds=seconds)

    def test_under_grace_is_free(self):
        """10 minutes with a 15-minute grace → $0.00."""
        entry, exit_ = self._span(minutes=10)
        assert calculate_charge(entry, exit_, _settings()) == Decimal('0.00')

    def test_grace_boundary_is_free(self):
        """Exactly at the grace boundary (<=) is still free."""
        entry, exit_ = self._span(minutes=15)
        assert calculate_charge(entry, exit_, _settings()) == Decimal('0.00')

    def test_fractional_second_over_grace_is_billed(self):
        """15 min + 0.5 s is outside grace and bills the first full hour."""
        entry = timezone.now()
        exit_ = entry + timedelta(minutes=15, milliseconds=500)
        assert calculate_charge(entry, exit_, _settings()) == Decimal('5.00')

    def test_hour_exact_boundary(self):
        """60 minutes → ceil(1.0h) = 1 unit → $5.00."""
        entry, exit_ = self._span(minutes=60)
        assert calculate_charge(entry, exit_, _settings()) == Decimal('5.00')

    def test_hour_one_second_over_rounds_up(self):
        """60 min + 1 s → ceil(>1h) = 2 units → $10.00 (ceil boundary)."""
        entry, exit_ = self._span(minutes=60, seconds=1)
        assert calculate_charge(entry, exit_, _settings()) == Decimal('10.00')

    def test_hour_partial_rounds_up(self):
        """61 minutes → ceil(1.016h) = 2 units → $10.00."""
        entry, exit_ = self._span(minutes=61)
        assert calculate_charge(entry, exit_, _settings()) == Decimal('10.00')

    def test_per_minute_billing(self):
        """90 minutes at $0.25/min → ceil(90) * 0.25 = $22.50."""
        settings = _settings(billing_unit='minute', rate=Decimal('0.25'))
        entry, exit_ = self._span(minutes=90)
        assert calculate_charge(entry, exit_, settings) == Decimal('22.50')

    def test_per_minute_sub_minute_rounds_up(self):
        """90 min + 1 s → ceil(91) * 0.25 = $22.75."""
        settings = _settings(billing_unit='minute', rate=Decimal('0.25'))
        entry, exit_ = self._span(minutes=90, seconds=1)
        assert calculate_charge(entry, exit_, settings) == Decimal('22.75')

    def test_per_minute_fractional_second_rounds_up(self):
        """60.5 seconds at per-minute billing consumes 2 billed minutes."""
        settings = _settings(
            billing_unit='minute',
            rate=Decimal('0.25'),
            grace_period_minutes=0,
        )
        entry = timezone.now()
        exit_ = entry + timedelta(seconds=60, milliseconds=500)
        assert calculate_charge(entry, exit_, settings) == Decimal('0.50')

    def test_daily_cap_applied(self):
        """8 hours = $40 but capped at $25.00."""
        settings = _settings(daily_cap_enabled=True, daily_cap_amount=Decimal('25.00'))
        entry, exit_ = self._span(minutes=8 * 60)
        assert calculate_charge(entry, exit_, settings) == Decimal('25.00')

    def test_daily_cap_inert_when_under(self):
        """A charge below the cap is unaffected by it."""
        settings = _settings(daily_cap_enabled=True, daily_cap_amount=Decimal('25.00'))
        entry, exit_ = self._span(minutes=90)  # ceil(1.5h)=2 → $10
        assert calculate_charge(entry, exit_, settings) == Decimal('10.00')

    def test_daily_cap_enabled_but_amount_none_skips_cap(self):
        """Cap enabled with no amount set → bill uncapped, no crash."""
        settings = _settings(daily_cap_enabled=True, daily_cap_amount=None)
        entry, exit_ = self._span(minutes=8 * 60)
        assert calculate_charge(entry, exit_, settings) == Decimal('40.00')

    def test_result_is_quantized_to_cents(self):
        """Charge is always 2 decimal places (10.00, not 10)."""
        entry, exit_ = self._span(minutes=90)
        result = calculate_charge(entry, exit_, _settings())
        assert result == Decimal('10.00')
        assert result.as_tuple().exponent == -2

    def test_non_positive_duration_is_free(self):
        """Defensive: exit at or before entry → $0.00, no exception."""
        entry = timezone.now()
        assert calculate_charge(entry, entry, _settings()) == Decimal('0.00')
        assert calculate_charge(entry, entry - timedelta(minutes=5), _settings()) == Decimal('0.00')

    def test_unknown_billing_unit_defaults_to_hour(self):
        """An unrecognized billing_unit falls back to per-hour (logged)."""
        settings = _settings(billing_unit='day')
        entry, exit_ = self._span(minutes=90)  # ceil(1.5h)=2 → $10
        assert calculate_charge(entry, exit_, settings) == Decimal('10.00')

    def test_float_rate_is_coerced_to_decimal(self):
        """A non-Decimal rate is coerced safely without float artifacts."""
        settings = _settings(rate=5.0)  # float on purpose
        entry, exit_ = self._span(minutes=90)
        assert calculate_charge(entry, exit_, settings) == Decimal('10.00')


# ── handle_entry ─────────────────────────────────────────────────────────────

@pytest.mark.django_db
class TestHandleEntry:
    """Tests for opening sessions on entry detections."""

    def test_guest_entry_creates_unlinked_session(self, parking_lot, lot_settings):
        """An unknown plate opens an active guest session (no user/plate)."""
        session = handle_entry('GUEST99', 0.9, [0.1, 0.2, 0.3, 0.4], PLATE_IMAGE, parking_lot)
        assert session.pk is not None
        assert session.status == 'active'
        assert session.user is None
        assert session.license_plate is None
        event = PlateDetectionEvent.objects.get(session=session)
        assert event.event_type == 'entry'
        assert event.lot == parking_lot
        assert event.is_low_confidence is False

    def test_registered_entry_links_user_and_plate(self, parking_lot, lot_settings, license_plate, user):
        """A plate matching a registration links the session to its owner."""
        session = handle_entry('ABC 123', 0.9, [], PLATE_IMAGE, parking_lot)
        assert session.license_plate == license_plate
        assert session.user == user

    def test_entry_normalizes_plate_keeps_raw(self, parking_lot, lot_settings):
        """Session stores the normalized plate; event keeps the raw read."""
        session = handle_entry('abc 123', 0.9, [], PLATE_IMAGE, parking_lot)
        assert session.plate_text == 'ABC123'
        event = PlateDetectionEvent.objects.get(session=session)
        assert event.raw_plate_text == 'abc 123'

    def test_low_confidence_flagged_but_session_created(self, parking_lot, lot_settings):
        """Below-threshold reads are flagged yet still open a session."""
        session = handle_entry('LOWC01', 0.45, [], PLATE_IMAGE, parking_lot)
        assert session.status == 'active'
        event = PlateDetectionEvent.objects.get(session=session)
        assert event.is_low_confidence is True

    def test_confidence_equal_to_threshold_not_low(self, parking_lot, lot_settings):
        """Confidence == threshold is NOT low (strict < comparison)."""
        session = handle_entry('EQ0600', 0.6, [], PLATE_IMAGE, parking_lot)
        event = PlateDetectionEvent.objects.get(session=session)
        assert event.is_low_confidence is False

    def test_per_lot_threshold_is_honored(self, parking_lot):
        """A lot with a 0.9 threshold flags a 0.7 read as low confidence.

        Proves the service uses the per-lot setting, not the CV constant (0.6).
        """
        LotSettings.objects.create(lot=parking_lot, confidence_threshold=0.9)
        session = handle_entry('HIGHTH', 0.7, [], PLATE_IMAGE, parking_lot)
        event = PlateDetectionEvent.objects.get(session=session)
        assert event.is_low_confidence is True

    def test_orphan_session_is_voided(self, parking_lot, lot_settings):
        """Re-entry of an active plate voids the old session and warns the new."""
        first = handle_entry('DUP777', 0.9, [], PLATE_IMAGE, parking_lot)
        second = handle_entry('DUP777', 0.9, [], PLATE_IMAGE, parking_lot)

        first.refresh_from_db()
        assert first.status == 'void'
        assert first.was_orphaned is True
        assert first.charge_amount == Decimal('0.00')
        assert second.status == 'active'
        assert second.has_duplicate_warning is True

    def test_orphan_check_is_scoped_to_lot(self, parking_lot, lot_settings):
        """An active session for the same plate in another lot is untouched."""
        other_lot = ParkingLot.objects.create(name='Other Lot')
        LotSettings.objects.create(lot=other_lot)

        first = handle_entry('CROSS1', 0.9, [], PLATE_IMAGE, parking_lot)
        handle_entry('CROSS1', 0.9, [], PLATE_IMAGE, other_lot)

        first.refresh_from_db()
        assert first.status == 'active'  # not voided by the other-lot entry

    def test_empty_plate_is_rejected(self, parking_lot, lot_settings):
        """A blank read must not open a session keyed on an empty plate."""
        with pytest.raises(ValueError):
            handle_entry('   ', 0.2, [], PLATE_IMAGE, parking_lot)
        assert not ParkingSession.objects.exists()

    def test_overlength_plate_is_rejected(self, parking_lot, lot_settings):
        """Plate input longer than the 20-char column is rejected, not truncated."""
        with pytest.raises(ValueError):
            handle_entry('A' * 21, 0.9, [], PLATE_IMAGE, parking_lot)

    def test_bounding_box_is_sanitized(self, parking_lot, lot_settings):
        """Out-of-range coords are clamped; malformed shapes become []."""
        session = handle_entry('BBOX01', 0.9, [1.5, -0.2, 0.5, 0.5], PLATE_IMAGE, parking_lot)
        event = PlateDetectionEvent.objects.get(session=session)
        assert event.bounding_box == [1.0, 0.0, 0.5, 0.5]

        session2 = handle_entry('BBOX02', 0.9, [0.1, 0.2, 0.3], PLATE_IMAGE, parking_lot)
        event2 = PlateDetectionEvent.objects.get(session=session2)
        assert event2.bounding_box == []  # wrong length → empty

    def test_links_lowest_pk_on_multi_user_plate(self, parking_lot, lot_settings, license_plate, user):
        """When two users register the same plate, link to the lowest-pk one."""
        other = User.objects.create_user(username='other', password='x')
        LicensePlate.objects.create(user=other, plate_text='ABC123')  # higher pk

        session = handle_entry('ABC123', 0.9, [], PLATE_IMAGE, parking_lot)
        assert session.license_plate == license_plate  # the lower-pk registration
        assert session.user == user


# ── handle_exit ──────────────────────────────────────────────────────────────

@pytest.mark.django_db
class TestHandleExit:
    """Tests for closing sessions on exit detections."""

    def _active_session(self, lot, plate='ABC123', minutes_ago=90):
        """Create an active session that entered `minutes_ago` minutes ago."""
        return ParkingSession.objects.create(
            plate_text=plate,
            lot=lot,
            entry_time=timezone.now() - timedelta(minutes=minutes_ago),
            status='active',
        )

    def test_normal_exit_completes_and_bills(self, parking_lot, lot_settings):
        """A 90-min session at $5/hr completes with a $10.00 charge."""
        self._active_session(parking_lot, minutes_ago=90)
        session = handle_exit('ABC123', 0.9, [], PLATE_IMAGE, parking_lot)

        assert session is not None
        assert session.status == 'completed'
        assert session.exit_time is not None
        assert session.duration_seconds > 0
        assert session.charge_amount == Decimal('10.00')
        assert PlateDetectionEvent.objects.filter(session=session, event_type='exit').exists()

    def test_exit_within_grace_is_free(self, parking_lot, lot_settings):
        """A 5-minute session completes with a $0.00 charge."""
        self._active_session(parking_lot, minutes_ago=5)
        session = handle_exit('ABC123', 0.9, [], PLATE_IMAGE, parking_lot)
        assert session.status == 'completed'
        assert session.charge_amount == Decimal('0.00')

    def test_exit_without_entry_flags_for_review(self, parking_lot, lot_settings):
        """No active session → None returned, flagged event, no session made."""
        result = handle_exit('NOENTRY', 0.95, [], PLATE_IMAGE, parking_lot)

        assert result is None
        assert not ParkingSession.objects.filter(plate_text='NOENTRY').exists()
        event = PlateDetectionEvent.objects.get(raw_plate_text='NOENTRY')
        assert event.session is None
        assert event.lot == parking_lot
        assert event.event_type == 'exit'
        assert event.is_low_confidence is True  # forced flag despite high confidence

    def test_exit_matches_on_normalized_plate(self, parking_lot, lot_settings):
        """A raw 'abc 123' exit matches the normalized 'ABC123' session."""
        self._active_session(parking_lot, plate='ABC123', minutes_ago=30)
        session = handle_exit('abc 123', 0.9, [], PLATE_IMAGE, parking_lot)
        assert session is not None
        assert session.plate_text == 'ABC123'

    def test_empty_exit_plate_goes_to_review(self, parking_lot, lot_settings):
        """An unreadable exit plate matches nothing → flagged event, None."""
        result = handle_exit('   ', 0.1, [], PLATE_IMAGE, parking_lot)
        assert result is None
        event = PlateDetectionEvent.objects.get(event_type='exit')
        assert event.session is None
        assert event.is_low_confidence is True

    def test_clock_skew_exit_is_bumped_after_entry(self, parking_lot, lot_settings):
        """A future entry_time (clock skew) → exit bumped to entry+1s, valid row."""
        ParkingSession.objects.create(
            plate_text='SKEW01',
            lot=parking_lot,
            entry_time=timezone.now() + timedelta(seconds=60),
            status='active',
        )
        session = handle_exit('SKEW01', 0.9, [], PLATE_IMAGE, parking_lot)
        assert session.status == 'completed'
        assert session.exit_time > session.entry_time
        assert session.duration_seconds >= 1
        assert session.charge_amount == Decimal('0.00')  # 1s → under grace

    def test_exit_matches_active_session_only(self, parking_lot, lot_settings):
        """A prior completed session for the plate is ignored; active one closes."""
        ParkingSession.objects.create(
            plate_text='ABC123',
            lot=parking_lot,
            entry_time=timezone.now() - timedelta(hours=5),
            exit_time=timezone.now() - timedelta(hours=4),
            duration_seconds=3600,
            charge_amount=Decimal('5.00'),
            status='completed',
        )
        active = self._active_session(parking_lot, plate='ABC123', minutes_ago=30)
        session = handle_exit('ABC123', 0.9, [], PLATE_IMAGE, parking_lot)
        assert session.pk == active.pk


# ── correct_plate ────────────────────────────────────────────────────────────

@pytest.mark.django_db
class TestCorrectPlate:
    """Tests for operator plate corrections and re-linking."""

    def test_correction_sets_flags_and_normalizes(self, parking_lot, lot_settings):
        """Correcting an event marks it corrected with the normalized text."""
        session = handle_entry('WRONG1', 0.4, [], PLATE_IMAGE, parking_lot)
        event = PlateDetectionEvent.objects.get(session=session)

        result = correct_plate(event.pk, 'xyz 789')
        assert result.manually_corrected is True
        assert result.corrected_plate == 'XYZ789'

    def test_correction_updates_session_plate(self, parking_lot, lot_settings):
        """The linked session's plate_text is updated to the corrected value."""
        session = handle_entry('WRONG2', 0.4, [], PLATE_IMAGE, parking_lot)
        event = PlateDetectionEvent.objects.get(session=session)

        correct_plate(event.pk, 'RIGHT2')
        session.refresh_from_db()
        assert session.plate_text == 'RIGHT2'

    def test_entry_correction_voids_duplicate_active_session(self, parking_lot, lot_settings):
        """Correcting an active entry preserves orphan handling for duplicates."""
        existing = handle_entry('RIGHT2', 0.9, [], PLATE_IMAGE, parking_lot)
        corrected = handle_entry('WRONG2', 0.4, [], PLATE_IMAGE, parking_lot)
        event = PlateDetectionEvent.objects.get(session=corrected)

        correct_plate(event.pk, 'RIGHT2')

        existing.refresh_from_db()
        corrected.refresh_from_db()
        assert existing.status == 'void'
        assert existing.was_orphaned is True
        assert existing.charge_amount == Decimal('0.00')
        assert corrected.status == 'active'
        assert corrected.plate_text == 'RIGHT2'
        assert corrected.has_duplicate_warning is True

    def test_correction_relinks_to_registered_plate(self, parking_lot, lot_settings, license_plate, user):
        """Correcting to a registered plate links the session to its owner."""
        session = handle_entry('MISRED', 0.4, [], PLATE_IMAGE, parking_lot)
        assert session.user is None  # started as guest
        event = PlateDetectionEvent.objects.get(session=session)

        correct_plate(event.pk, 'ABC123')  # registered to `user`
        session.refresh_from_db()
        assert session.license_plate == license_plate
        assert session.user == user

    def test_correction_clears_link_for_unregistered(self, parking_lot, lot_settings, license_plate, user):
        """Correcting a linked session to an unregistered plate reverts to guest."""
        session = handle_entry('ABC123', 0.4, [], PLATE_IMAGE, parking_lot)
        assert session.user == user  # started linked
        event = PlateDetectionEvent.objects.get(session=session)

        correct_plate(event.pk, 'UNREG9')
        session.refresh_from_db()
        assert session.license_plate is None
        assert session.user is None

    def test_correction_of_orphan_exit_event_without_match(self, parking_lot, lot_settings):
        """Correcting an unmatched-exit event with no active match keeps it queued."""
        handle_exit('GHOST1', 0.4, [], PLATE_IMAGE, parking_lot)
        event = PlateDetectionEvent.objects.get(raw_plate_text='GHOST1')
        assert event.session is None

        result = correct_plate(event.pk, 'FIXED1')
        assert result.manually_corrected is True
        assert result.corrected_plate == 'FIXED1'
        result.refresh_from_db()
        assert result.session is None

    def test_correction_of_orphan_exit_closes_active_session(self, parking_lot, lot_settings):
        """Correcting a misread exit reconciles it with the same-lot active session."""
        active = ParkingSession.objects.create(
            plate_text='ABC123',
            lot=parking_lot,
            entry_time=timezone.now() - timedelta(minutes=90),
            status='active',
        )
        handle_exit('ABC128', 0.4, [], PLATE_IMAGE, parking_lot)
        event = PlateDetectionEvent.objects.get(raw_plate_text='ABC128')
        assert event.session is None
        assert event.lot == parking_lot

        result = correct_plate(event.pk, 'ABC123')

        active.refresh_from_db()
        result.refresh_from_db()
        assert active.status == 'completed'
        assert active.charge_amount == Decimal('10.00')
        assert active.exit_time is not None
        assert result.session == active

    def test_orphan_exit_correction_is_scoped_to_lot(self, parking_lot, lot_settings):
        """Corrected exits close only active sessions in the event's lot."""
        other_lot = ParkingLot.objects.create(name='Other Lot')
        LotSettings.objects.create(lot=other_lot, rate=Decimal('5.00'))
        other_active = ParkingSession.objects.create(
            plate_text='ABC123',
            lot=other_lot,
            entry_time=timezone.now() - timedelta(minutes=90),
            status='active',
        )
        handle_exit('ABC128', 0.4, [], PLATE_IMAGE, parking_lot)
        event = PlateDetectionEvent.objects.get(raw_plate_text='ABC128')

        result = correct_plate(event.pk, 'ABC123')

        other_active.refresh_from_db()
        result.refresh_from_db()
        assert other_active.status == 'active'
        assert result.session is None

    def test_correction_invalid_id_raises(self, db):
        """An unknown event id raises DoesNotExist (explicit failure)."""
        with pytest.raises(PlateDetectionEvent.DoesNotExist):
            correct_plate(999999, 'ABC123')

    def test_empty_correction_is_rejected(self, parking_lot, lot_settings):
        """Blanking a plate via correction is rejected, not silently applied."""
        session = handle_entry('WRONG3', 0.4, [], PLATE_IMAGE, parking_lot)
        event = PlateDetectionEvent.objects.get(session=session)
        with pytest.raises(ValueError):
            correct_plate(event.pk, '   ')


# ── Constraint regressions ────────────────────────────────────────────────────

@pytest.mark.django_db
class TestConstraintRegressions:
    """Guards that service writes never violate the DB CheckConstraints."""

    def test_voided_orphan_persists_with_zero_charge(self, parking_lot, lot_settings):
        """The voided orphan row commits (charge 0 satisfies session_void_no_charge)."""
        first = handle_entry('VOID01', 0.9, [], PLATE_IMAGE, parking_lot)
        handle_entry('VOID01', 0.9, [], PLATE_IMAGE, parking_lot)
        # Re-fetch from DB: a constraint violation would have raised on save.
        stored = ParkingSession.objects.get(pk=first.pk)
        assert stored.status == 'void'
        assert stored.charge_amount == Decimal('0.00')

    def test_completed_exit_has_exit_after_entry(self, parking_lot, lot_settings):
        """Completed session row has exit_time strictly after entry_time."""
        ParkingSession.objects.create(
            plate_text='AFTER1',
            lot=parking_lot,
            entry_time=timezone.now() - timedelta(minutes=45),
            status='active',
        )
        session = handle_exit('AFTER1', 0.9, [], PLATE_IMAGE, parking_lot)
        stored = ParkingSession.objects.get(pk=session.pk)
        assert stored.exit_time > stored.entry_time
