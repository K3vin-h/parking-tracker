"""
Session & billing business logic for the parking app.

WHY THIS LAYER EXISTS:
  The CV pipeline (apps/cv) turns an uploaded image into a plate reading. The
  models (apps/parking/models.py) store sessions and detection events. This
  module is the glue in between: it decides what a detection *means* for the
  business — open a session, close one, charge for it, void a duplicate, or
  flag a bad read for an operator to fix.

  Keeping this logic in one place (rather than scattered across views) means:
    1. The upload API, the admin, and tests all call the same code path.
    2. Every money calculation goes through one audited function.
    3. The rules (grace period, orphan handling, low-confidence) live together.

DESIGN CONTRACT:
  - This module is PURE business logic. It never loads CV model weights or
    calls the pipeline — callers pass in the already-extracted detection data
    (plate_text, confidence, bounding_box, image). That keeps it fast and
    trivially unit-testable.
  - All money is Decimal, never float. This is the cardinal rule of billing.
  - All timestamps are UTC via django.utils.timezone.now().
  - No silent failures: every branch logs, returns an explicit value, or raises.

CONCURRENCY:
  Two images for the same plate can be uploaded near-simultaneously. The
  mutating functions wrap their reads+writes in transaction.atomic(). handle_exit
  and correct_plate lock the session/event row with select_for_update() (which
  only locks inside an open transaction) so a concurrent exit can't double-bill;
  handle_entry voids orphans with a single atomic UPDATE statement.
"""

import logging
import math
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_CEILING, ROUND_HALF_UP
from hashlib import sha256

from django.core.files.uploadedfile import UploadedFile
from django.db import transaction
from django.utils import timezone

from apps.parking.models import (
    LicensePlate,
    LotSettings,
    ParkingLot,
    ParkingSession,
    PlateDetectionEvent,
)

logger = logging.getLogger(__name__)

# Charges are stored to the cent; this is the quantum every Decimal result is
# rounded to before it touches the DB (DecimalField(decimal_places=2)).
_MONEY_QUANTUM = Decimal("0.01")

# Plate columns (ParkingSession.plate_text, PlateDetectionEvent.raw_plate_text /
# corrected_plate) are CharField(max_length=20). CharField does NOT enforce
# length on .create() — Postgres raises DataError mid-transaction instead. We
# validate at this boundary so a hostile/garbled over-length read fails fast
# with a clear ValueError rather than an opaque DB error.
MAX_PLATE_LEN = 20


def _plate_log_token(plate_text: str) -> str:
    """
    Return a stable, non-reversible token for plate values written to logs.

    WHY: license plates are vehicle movement PII. Operators still need to
    correlate repeated log messages for the same plate while debugging, but
    INFO/WARNING logs must not expose the raw normalized plate to centralized
    logging systems.
    """
    if not plate_text:
        return "empty"
    return sha256(plate_text.encode("utf-8")).hexdigest()[:12]


def _duration_seconds_decimal(start: datetime, end: datetime) -> Decimal:
    """
    Return an exact Decimal duration in seconds, preserving microseconds.

    WHY: timedelta.total_seconds() returns float. Truncating or using binary
    float would under-bill boundary cases such as 60.5 seconds with per-minute
    billing. This helper keeps billing comparisons and ceil() inputs exact.
    """
    delta = end - start
    whole_seconds = (delta.days * 24 * 60 * 60) + delta.seconds
    return Decimal(whole_seconds) + (Decimal(delta.microseconds) / Decimal(1_000_000))


def _ceil_duration_seconds(start: datetime, end: datetime) -> int:
    """
    Convert an exact duration to stored integer seconds by rounding up.

    WHY: duration_seconds is an IntegerField used for analytics, while billing
    charges any fractional occupied second. Rounding up preserves that same
    boundary behavior in the stored duration.
    """
    seconds = _duration_seconds_decimal(start, end)
    if seconds <= 0:
        return 0
    return int(seconds.to_integral_value(rounding=ROUND_CEILING))


def _require_plate_within_limits(plate_text: str) -> None:
    """
    Reject plate input that would overflow the 20-char plate columns.

    WHY RAISE (not truncate): a truncated plate is a silently wrong matching
    key, which would mis-bill the wrong vehicle. Failing fast is safer.
    """
    if plate_text and len(plate_text) > MAX_PLATE_LEN:
        logger.error("Plate text too long (%d chars); rejecting", len(plate_text))
        raise ValueError(f"plate text exceeds {MAX_PLATE_LEN} characters")


def _sanitize_bounding_box(bounding_box) -> list:
    """
    Coerce an untrusted bounding box into a safe 4-float list in [0, 1].

    WHY: bounding_box is a JSONField and this is a system boundary. The CV
    pipeline emits ``[x, y, w, h]`` normalised to [0, 1], but any caller could
    pass a malformed/oversized/non-numeric value that would poison the dashboard
    overlay later. Anything that is not exactly four numbers becomes [] (the
    model default); valid coordinates are clamped to [0, 1].
    """
    if not isinstance(bounding_box, list) or len(bounding_box) != 4:
        if bounding_box:  # non-empty but malformed — worth a warning
            logger.warning("Invalid bounding_box %r; storing empty list", bounding_box)
        return []
    try:
        return [min(1.0, max(0.0, float(value))) for value in bounding_box]
    except (TypeError, ValueError):
        logger.warning("Non-numeric bounding_box %r; storing empty list", bounding_box)
        return []


def normalize_plate(raw_text: str) -> str:
    """
    Normalize a raw plate reading into the canonical matching key.

    WHY: CV output (and human input) varies in spacing and case — 'abc 123',
    'ABC 123', and ' abc123 ' all mean the same plate. We collapse those to a
    single key ('ABC123') so entry/exit matching is reliable. Only whitespace
    is removed; hyphens and other characters are preserved deliberately, so
    'ABC-123' stays distinct from 'ABC123' (exact-match policy from PLAN.md).

    Returns '' for None/empty/whitespace-only input — the caller decides what an
    empty plate means, but we never crash on bad CV output.
    """
    if not raw_text:
        # Empty or None: nothing to normalize. Log so a flood of blank reads is
        # visible, but return a defined value rather than raising.
        logger.warning("normalize_plate received empty/None input")
        return ""

    # str.split() with no args splits on ANY run of whitespace (spaces, tabs,
    # newlines) and drops empties; rejoining with '' strips all of it.
    normalized = "".join(raw_text.split()).upper()
    if not normalized:
        logger.warning("normalize_plate received whitespace-only input")
    return normalized


def calculate_charge(
    entry_time: datetime, exit_time: datetime, lot_settings: LotSettings
) -> Decimal:
    """
    Calculate the parking charge for a session, in dollars, as a Decimal.

    WHY A DEDICATED FUNCTION: billing is the one place a bug costs real money,
    so it is isolated, pure (no DB writes), and unit-tested against every
    boundary. handle_exit delegates here.

    Rules (from PLAN.md / LotSettings):
      1. duration <= grace_period_minutes        -> $0.00 (free)
      2. billing_unit == 'minute'                -> ceil(total_minutes) * rate
      3. billing_unit == 'hour'                  -> ceil(total_hours)   * rate
      4. daily_cap_enabled and charge > cap      -> charge = cap

    All arithmetic stays in Decimal. We preserve fractional seconds so grace
    periods and ceil() billing never under-bill just-over-boundary sessions.
    """
    total_seconds = _duration_seconds_decimal(entry_time, exit_time)
    if total_seconds <= 0:
        # Defensive: exit should always be after entry (handle_exit guarantees
        # it), but a non-positive duration must never produce a charge.
        logger.error(
            "calculate_charge got non-positive duration (%ss); charging $0.00",
            total_seconds,
        )
        return Decimal("0.00")

    total_minutes = Decimal(total_seconds) / Decimal(60)

    # Rule 1: grace period. Sessions at or under the grace window are free.
    if total_minutes <= Decimal(lot_settings.grace_period_minutes):
        return Decimal("0.00")

    # Coerce via str so a stray int/float rate (e.g. from a hand-built settings
    # object) can never inject binary-float noise into the money math.
    rate = Decimal(str(lot_settings.rate))

    # Rules 2 & 3: round the billed quantity UP to the next whole unit, because
    # a partial hour/minute of parking still occupies the spot for that unit.
    # math.ceil on a Decimal returns an int (Decimal.__ceil__), staying exact.
    if lot_settings.billing_unit == "minute":
        units = math.ceil(total_minutes)
    elif lot_settings.billing_unit == "hour":
        units = math.ceil(total_minutes / Decimal(60))
    else:
        # billing_unit has model-level choices but those are not DB-enforced. An
        # unrecognised value is a config bug — bill per hour but log loudly so
        # it never passes silently.
        logger.error(
            "Unknown billing_unit %r; defaulting to per-hour billing",
            lot_settings.billing_unit,
        )
        units = math.ceil(total_minutes / Decimal(60))

    charge = Decimal(units) * rate

    # Rule 4: optional daily cap. Only applies when enabled AND configured.
    if lot_settings.daily_cap_enabled:
        if lot_settings.daily_cap_amount is None:
            # Misconfiguration: cap turned on but no amount set. Don't silently
            # cap to zero — log and bill the uncapped charge.
            logger.warning(
                "Lot settings have daily_cap_enabled but daily_cap_amount is None; "
                "skipping cap"
            )
        elif charge > lot_settings.daily_cap_amount:
            charge = lot_settings.daily_cap_amount

    return charge.quantize(_MONEY_QUANTUM, rounding=ROUND_HALF_UP)


def _clamp_confidence(confidence: float) -> float:
    """
    Clamp a confidence score into the DB-allowed [0.0, 1.0] range.

    WHY: PlateDetectionEvent.confidence_score has a CheckConstraint of
    0.0 <= score <= 1.0. The CV pipeline should always produce values in range,
    but the service is a system boundary — a stray out-of-range value would make
    the INSERT raise IntegrityError. Clamp (and log) so a bad input degrades
    gracefully instead of crashing the request.
    """
    if confidence < 0.0 or confidence > 1.0:
        logger.warning("confidence %.4f out of [0,1]; clamping", confidence)
    return min(1.0, max(0.0, confidence))


def _match_registered_plate(normalized_plate: str) -> LicensePlate | None:
    """
    Find a registered LicensePlate for a normalized plate string.

    WHY order_by('pk').first(): a plate string is unique per user, but the SAME
    string can be registered by two different users (the model only enforces
    uniqueness within a user). For a detection we have no user yet, so we link to
    the lowest-pk registration — an explicit, stable tie-break (a bare .first()
    would return DB-storage order, which is not guaranteed stable). A collision
    is logged so a wrong-user link is auditable rather than invisible.
    Returns the LicensePlate or None (guest).
    """
    matches = list(
        LicensePlate.objects.filter(plate_text=normalized_plate).order_by("pk")[:2]
    )
    if len(matches) > 1:
        logger.warning(
            "Plate hash %s matches multiple registrations; linking to lowest pk %s",
            _plate_log_token(normalized_plate),
            matches[0].pk,
        )
    return matches[0] if matches else None


@transaction.atomic
def handle_entry(
    plate_text: str,
    confidence: float,
    bounding_box: list[float],
    image: UploadedFile | None,
    lot: ParkingLot,
) -> ParkingSession:
    """
    Open a parking session for an entry detection and record the event.

    WHY ATOMIC: opening an entry may void a prior orphaned session AND create a
    new session AND create a detection event. Those must commit together — a
    partial write would corrupt the active-session invariant or revenue records.

    Flow (PLAN.md):
      1. Normalize plate; load lot settings.
      2. Flag low-confidence against the lot's OWN threshold (configurable).
      3. Orphan handling: if the plate already has an active session in this lot,
         void it (charge 0, was_orphaned=True) and mark the new one with a
         duplicate warning.
      4. Link to a registered plate/user if one matches; else guest (nulls).
      5. Create the active session and the linked 'entry' detection event.

    Raises:
      ValueError: if plate_text is empty or over-length after normalization (an
        unreadable read should be routed to manual review by the caller, never
        opened as a session keyed on an empty/garbage plate).
      LotSettings.DoesNotExist: if the lot has no settings (configuration bug).
    """
    _require_plate_within_limits(plate_text)
    normalized = normalize_plate(plate_text)
    if not normalized:
        # An empty matching key would collide across every blank read and corrupt
        # orphan/billing logic. Reject; the caller queues unreadable reads.
        logger.error("handle_entry: empty plate after normalization; rejecting")
        raise ValueError("plate_text is empty after normalization")
    settings = _get_lot_settings(lot)

    # Confidence is judged against the per-lot threshold, not a global constant,
    # so operators can tune sensitivity per lot.
    is_low_conf = confidence < settings.confidence_threshold

    # Orphan handling: a single atomic UPDATE voids any active session for this
    # plate in this lot. UPDATE ... WHERE status='active' locks and transitions
    # the matched rows in one statement (no read-then-write window), and the DB
    # CheckConstraints still apply. Two CONCURRENT entries are caught by the
    # session_one_active_per_lot_plate partial-unique constraint: the second
    # INSERT below raises IntegrityError instead of opening a duplicate active
    # session (this UPDATE alone only voids what it can already see).
    voided = ParkingSession.objects.filter(
        lot=lot, plate_text=normalized, status=ParkingSession.Status.ACTIVE
    ).update(
        # Void sessions MUST carry no charge (session_void_no_charge constraint).
        status=ParkingSession.Status.VOID,
        charge_amount=Decimal("0.00"),
        was_orphaned=True,
    )
    has_duplicate = voided > 0
    if has_duplicate:
        logger.info(
            "Voided %d orphaned session(s) for plate_hash=%s in lot %s",
            voided,
            _plate_log_token(normalized),
            lot.pk,
        )

    # Registration match links the session to a known user; absence = guest.
    registered = _match_registered_plate(normalized)

    session = ParkingSession.objects.create(
        plate_text=normalized,
        license_plate=registered,
        user=registered.user if registered else None,
        lot=lot,
        entry_time=timezone.now(),
        status=ParkingSession.Status.ACTIVE,
        has_duplicate_warning=has_duplicate,
    )

    PlateDetectionEvent.objects.create(
        session=session,
        lot=lot,
        image=image,
        raw_plate_text=plate_text,  # store the ORIGINAL read for audit/correction
        confidence_score=_clamp_confidence(confidence),
        event_type="entry",
        is_low_confidence=is_low_conf,
        bounding_box=_sanitize_bounding_box(bounding_box),
    )

    logger.info(
        "Entry: session %s plate_hash=%s lot=%s guest=%s low_conf=%s duplicate=%s",
        session.pk,
        _plate_log_token(normalized),
        lot.pk,
        registered is None,
        is_low_conf,
        has_duplicate,
    )
    return session


@transaction.atomic
def handle_exit(
    plate_text: str,
    confidence: float,
    bounding_box: list[float],
    image: UploadedFile | None,
    lot: ParkingLot,
) -> ParkingSession | None:
    """
    Close the matching active session for an exit detection and bill it.

    WHY ATOMIC: completing a session (status, exit_time, duration, charge) and
    recording the exit event must commit as one unit.

    Exit-without-entry policy (confirmed): if no active session matches, we do
    NOT auto-create one and do NOT raise. We record a flagged exit event with
    session=None so it surfaces in the operator review queue, and return None.
    An empty/unreadable plate naturally takes this same review path (it matches
    no active session) rather than raising.

    Returns the completed ParkingSession, or None for the error-queue path.

    Raises:
      ValueError: if plate_text is over-length (would overflow the plate column).
      LotSettings.DoesNotExist: if the lot has no settings (configuration bug).
    """
    _require_plate_within_limits(plate_text)
    normalized = normalize_plate(plate_text)
    settings = _get_lot_settings(lot)
    is_low_conf = confidence < settings.confidence_threshold

    # Lock the oldest active session for this plate/lot so a concurrent exit
    # can't bill it twice. order_by('entry_time') makes the choice deterministic.
    session = (
        ParkingSession.objects.select_for_update()
        .filter(lot=lot, plate_text=normalized, status=ParkingSession.Status.ACTIVE)
        .order_by("entry_time")
        .first()
    )

    if session is None:
        # Exit-without-entry: flag for review, no session created.
        PlateDetectionEvent.objects.create(
            session=None,
            lot=lot,
            image=image,
            raw_plate_text=plate_text,
            confidence_score=_clamp_confidence(confidence),
            event_type="exit",
            # Force the review flag regardless of confidence: an unmatched exit
            # always needs a human to reconcile it.
            is_low_confidence=True,
            bounding_box=_sanitize_bounding_box(bounding_box),
        )
        logger.warning(
            "Exit with no active session for plate_hash=%s lot=%s; flagged for review",
            _plate_log_token(normalized),
            lot.pk,
        )
        return None

    now = timezone.now()
    # Guard the session_exit_after_entry (strict >) and duration_non_negative
    # constraints against clock skew / sub-second turnarounds: exit_time must be
    # strictly after entry_time.
    if now <= session.entry_time:
        logger.warning(
            "Exit time %s not after entry %s for session %s; bumping +1s",
            now,
            session.entry_time,
            session.pk,
        )
        now = session.entry_time + timedelta(seconds=1)

    _complete_session_for_exit(session, now, settings)

    PlateDetectionEvent.objects.create(
        session=session,
        lot=lot,
        image=image,
        raw_plate_text=plate_text,
        confidence_score=_clamp_confidence(confidence),
        event_type="exit",
        is_low_confidence=is_low_conf,
        bounding_box=_sanitize_bounding_box(bounding_box),
    )

    logger.info(
        "Exit: session %s plate_hash=%s lot=%s duration=%ss low_conf=%s",
        session.pk,
        _plate_log_token(normalized),
        lot.pk,
        session.duration_seconds,
        is_low_conf,
    )
    return session


def _complete_session_for_exit(
    session: ParkingSession,
    exit_time: datetime,
    settings: LotSettings,
) -> Decimal:
    """
    Close an active session at a known exit time and return its final charge.

    WHY: both normal exits and manually corrected unmatched-exit events must
    run exactly the same billing/status transition. Keeping the mutation here
    prevents review corrections from drifting away from handle_exit behavior.

    PRECONDITION: only ever called with an ACTIVE session (handle_exit and
    correct_plate both filter to status='active' before calling). An active
    session is never orphaned, so was_orphaned is left untouched — we do NOT
    revive voided orphans here (that path was a double-billing risk).
    """
    duration_seconds = max(1, _ceil_duration_seconds(session.entry_time, exit_time))
    charge = calculate_charge(session.entry_time, exit_time, settings)

    session.status = ParkingSession.Status.COMPLETED
    session.exit_time = exit_time
    session.duration_seconds = duration_seconds
    session.charge_amount = charge
    session.save(
        update_fields=[
            "status",
            "exit_time",
            "duration_seconds",
            "charge_amount",
        ]
    )
    return charge


def _void_duplicate_active_sessions(
    lot: ParkingLot,
    normalized_plate: str,
    keep_session_id: int,
) -> int:
    """
    Void active sessions that would duplicate a corrected active session.

    WHY: manual correction can change an active session's matching key after it
    was opened. This must preserve the same "one active session per lot/plate"
    invariant as handle_entry's orphan handling.
    """
    voided = (
        ParkingSession.objects.filter(
            lot=lot,
            plate_text=normalized_plate,
            status=ParkingSession.Status.ACTIVE,
        )
        .exclude(pk=keep_session_id)
        .update(
            status=ParkingSession.Status.VOID,
            charge_amount=Decimal("0.00"),
            was_orphaned=True,
        )
    )
    if voided:
        logger.info(
            "Voided %d duplicate active session(s) during correction for "
            "plate_hash=%s in lot %s",
            voided,
            _plate_log_token(normalized_plate),
            lot.pk,
        )
    return voided


@transaction.atomic
def correct_plate(event_id: int, corrected_text: str) -> PlateDetectionEvent:
    """
    Apply an operator's manual plate correction to a detection event.

    WHY: low-confidence reads (and unmatched exits) land in a review queue. When
    an operator fixes the text, we must atomically: mark the event corrected,
    update the linked session's plate, and RE-EVALUATE the registration link —
    the corrected plate might now match a registered user (or no longer match,
    reverting the session to guest).

    AUTHORIZATION: this service performs no access control. The caller (the
    planned PATCH /api/events/<id>/correct/ view) MUST restrict this to staff /
    lot operators — any caller can otherwise rewrite any event and relink any
    session to any registered user.

    Raises:
      PlateDetectionEvent.DoesNotExist: for an unknown event id.
      ValueError: if corrected_text is empty or over-length after normalization.
    """
    _require_plate_within_limits(corrected_text)
    try:
        event = PlateDetectionEvent.objects.select_for_update().get(pk=event_id)
    except PlateDetectionEvent.DoesNotExist:
        logger.error("correct_plate: no detection event with id %s", event_id)
        raise

    normalized = normalize_plate(corrected_text)
    if not normalized:
        # Blanking a plate via "correction" would corrupt the event/session.
        logger.error("correct_plate: empty corrected text for event %s", event_id)
        raise ValueError("corrected plate text is empty after normalization")
    event.manually_corrected = True
    event.corrected_plate = normalized

    if event.session_id is not None:
        # Lock the session row too so the relink can't race a concurrent exit.
        session = ParkingSession.objects.select_for_update().get(pk=event.session_id)
        duplicate_voided = 0
        if session.status == ParkingSession.Status.ACTIVE:
            duplicate_voided = _void_duplicate_active_sessions(
                session.lot,
                normalized,
                session.pk,
            )
        session.plate_text = normalized
        if duplicate_voided:
            session.has_duplicate_warning = True
        registered = _match_registered_plate(normalized)
        session.license_plate = registered
        session.user = registered.user if registered else None
        session.save(
            update_fields=[
                "plate_text",
                "has_duplicate_warning",
                "license_plate",
                "user",
            ]
        )
        # Only write `lot` when we actually backfill it; otherwise the save would
        # rewrite an unchanged column (and clobber it if logic ever diverges).
        event_update_fields = ["manually_corrected", "corrected_plate"]
        if event.lot_id is None:
            event.lot = session.lot
            event_update_fields.append("lot")
        event.save(update_fields=event_update_fields)
        logger.info(
            "Corrected event %s -> plate_hash=%s; session %s relinked "
            "(guest=%s duplicate_voided=%d)",
            event_id,
            _plate_log_token(normalized),
            session.pk,
            registered is None,
            duplicate_voided,
        )
    elif event.event_type == "exit" and event.lot_id is not None:
        # Reconcile a corrected unmatched-exit event to a still-OPEN session only.
        # We deliberately do NOT match voided/orphaned sessions: a session voided
        # because the car re-entered was correctly abandoned, and reviving+billing
        # it here would double-bill the same vehicle (the live session bills too).
        settings = _get_lot_settings(event.lot)
        session = (
            ParkingSession.objects.select_for_update()
            .filter(
                lot=event.lot,
                plate_text=normalized,
                status=ParkingSession.Status.ACTIVE,
                entry_time__lt=event.timestamp,
            )
            .order_by("entry_time")
            .first()
        )
        if session is not None:
            exit_time = event.timestamp
            if exit_time <= session.entry_time:
                logger.warning(
                    "Corrected exit event %s timestamp not after entry for "
                    "session %s; bumping +1s",
                    event_id,
                    session.pk,
                )
                exit_time = session.entry_time + timedelta(seconds=1)
            _complete_session_for_exit(session, exit_time, settings)
            event.session = session
            event.save(
                update_fields=["manually_corrected", "corrected_plate", "session"]
            )
            logger.info(
                "Corrected unmatched exit event %s closed session %s for "
                "plate_hash=%s in lot %s",
                event_id,
                session.pk,
                _plate_log_token(normalized),
                event.lot_id,
            )
        else:
            event.save(update_fields=["manually_corrected", "corrected_plate"])
            logger.info(
                "Corrected unmatched exit event %s found no active session for "
                "plate_hash=%s in lot %s",
                event_id,
                _plate_log_token(normalized),
                event.lot_id,
            )
    else:
        # Orphan event without lot context cannot be reconciled to a session.
        event.save(update_fields=["manually_corrected", "corrected_plate"])
        logger.info(
            "Corrected event %s -> plate_hash=%s; no linked session to relink",
            event_id,
            _plate_log_token(normalized),
        )

    return event


def _get_lot_settings(lot: ParkingLot) -> LotSettings:
    """
    Fetch the LotSettings for a lot, raising explicitly if missing.

    WHY RAISE (not default): a lot with no settings is a configuration bug
    (setup_defaults seeds them). Billing must never run against silent fallback
    rates, so we surface it loudly rather than guess.
    """
    try:
        return lot.settings
    except LotSettings.DoesNotExist:
        logger.error("Lot %s has no LotSettings; cannot process detection", lot.pk)
        raise
