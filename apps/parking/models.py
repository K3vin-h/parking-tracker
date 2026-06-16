"""
Database models for the parking app.

This file defines the complete data schema for all parking operations.
Every table in the database corresponds to a model class in this file.

─────────────────────────────────────────────────────────────────────────────
WHY POSTGRESQL OVER SQLITE FOR THESE MODELS?
─────────────────────────────────────────────────────────────────────────────
  1. DecimalField: PostgreSQL stores it as NUMERIC (exact). SQLite stores it
     as REAL (floating-point). This matters for billing — 0.1 + 0.2 == 0.3
     in NUMERIC, but 0.30000000000000004 in floating-point.
  2. JSONField: PostgreSQL has a native JSONB column type with indexing and
     query operators. We use this for bounding_box storage.
  3. Concurrent writes: the orphan-handling logic requires safe concurrent
     updates. PostgreSQL handles this correctly; SQLite uses file locks.

─────────────────────────────────────────────────────────────────────────────
THE CARDINAL RULE: NEVER USE FLOAT FOR MONEY
─────────────────────────────────────────────────────────────────────────────
  ALL monetary values use DecimalField in the database and Python's Decimal
  type in application code. Never use float for money — it is imprecise.

  Example of the float problem:
    >>> 0.1 + 0.2
    0.30000000000000004        # WRONG
    >>> from decimal import Decimal
    >>> Decimal('0.1') + Decimal('0.2')
    Decimal('0.3')             # CORRECT

  In this project, 'charge_amount', 'rate', 'daily_cap_amount' are always Decimal.

─────────────────────────────────────────────────────────────────────────────
RELATIONSHIP DIAGRAM
─────────────────────────────────────────────────────────────────────────────

  User (accounts.User)
    │
    ├─< LicensePlate (one user → many plates)
    │     │
    │     └─< ParkingSession.license_plate (optional FK)
    │
    └─< ParkingSession.user (optional FK, null = guest)

  ParkingLot
    │
    ├── LotSettings (one lot → one settings record, OneToOne)
    │
    └─< ParkingSession (one lot → many sessions)
          │
          └─< PlateDetectionEvent (one session → many detection events)
"""

from decimal import Decimal

from django.conf import settings
from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models


class LicensePlate(models.Model):
    """
    A registered license plate belonging to a user account.

    REGISTERED vs GUEST plates:
      A "registered" plate has a LicensePlate record. When such a plate is
      detected by the CV pipeline, the system links the parking session to both
      the LicensePlate record and the owning User. This enables per-user session
      history and distinguishes registered from guest traffic in reports.

      A "guest" plate has no LicensePlate record. The session's license_plate and
      user fields are left null, and the session appears in the guest section of
      the dashboard.

    NORMALIZATION:
      plate_text is ALWAYS stored in normalized form: whitespace stripped,
      all characters uppercased. The normalize_plate() function in utils.py
      (Day 7) handles this transformation. This guarantees that plate lookups
      are consistent regardless of how the CV pipeline formats its output.

      Examples:
        "abc 123"  → "ABC123"
        "  AB-12 " → "AB-12"   (hyphens are preserved, only whitespace is stripped)
    """

    # WHY settings.AUTH_USER_MODEL instead of importing User directly?
    #   If we wrote: from apps.accounts.models import User, that would create a
    #   direct import dependency. Django recommends using the string reference
    #   'accounts.User' via settings.AUTH_USER_MODEL to avoid circular imports
    #   and to respect any future change to AUTH_USER_MODEL.
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='plates',
        help_text="The user account that owns this license plate.",
    )

    # Stored normalized: stripped whitespace, uppercase.
    # max_length=20 — the longest realistic US/Canada plate is about 10 characters.
    # 20 gives comfortable headroom without wasting column space.
    plate_text = models.CharField(
        max_length=20,
        help_text="License plate text, normalized: whitespace stripped, uppercase only.",
    )

    # True for the plate the user considers their primary/default vehicle.
    # Useful for UI personalization (e.g., showing the primary plate prominently).
    # A user can have multiple plates but at most one primary (enforced by application
    # logic in services.py, not by a DB constraint — multiple True values are possible
    # in the DB but the UI ensures only one is set at a time).
    is_primary = models.BooleanField(
        default=False,
        help_text="Whether this is the user's primary plate (shown by default in the UI).",
    )

    # Optional human-readable label so users can tell their plates apart.
    # blank=True: the field is optional in Django forms.
    # default='': the DB stores an empty string (not NULL) when no label is given.
    label = models.CharField(
        max_length=100,
        blank=True,
        default='',
        help_text="Optional label to identify this vehicle, e.g. 'Daily Driver' or 'Work Truck'.",
    )

    class Meta:
        verbose_name = 'license plate'
        verbose_name_plural = 'license plates'
        # Prevents one user from registering the same plate twice.
        # Does NOT prevent two DIFFERENT users from registering the same plate_text.
        # DAY 7 NOTE: handle_entry() in services.py must handle the ambiguous case
        # where multiple LicensePlate records share the same plate_text across users.
        # Resolution strategy: match via session.plate_text directly; use .first()
        # on the LicensePlate lookup and document that cross-user plate conflicts
        # are a known edge case (two people share a plate — unlikely in practice).
        #
        # WHY UniqueConstraint (not unique_together): unique_together is
        # deprecated since Django 4.2; the named-constraint form also allows
        # adding condition/deferrable options later without restructuring.
        constraints = [
            models.UniqueConstraint(
                fields=['user', 'plate_text'],
                name='licenseplate_user_plate_unique',
            ),
        ]

    def __str__(self):
        # "ABC123 (Work Truck)" or just "ABC123" if no label.
        label_part = f' ({self.label})' if self.label else ''
        return f'{self.plate_text}{label_part}'


class ParkingLot(models.Model):
    """
    Represents a physical parking lot managed by this system.

    MULTI-LOT DESIGN:
      The schema supports multiple parking lots from day one. Each ParkingSession
      and LotSettings record is FK-linked to a specific lot, so revenue, occupancy,
      and settings can be tracked independently per lot.

      The current UI assumes a single lot ('Main Lot'), created by setup_defaults.
      Extending to multiple lots requires only:
        1. Adding a lot selector to the upload form
        2. Routing sessions to the correct lot — zero schema changes required.

      This is the "multi-lot ready" design referenced in PLAN.md's Notes section.
    """

    # unique=True: setup_defaults uses get_or_create(name='Main Lot') — without
    # uniqueness two rows could share a name and get_or_create would silently
    # return whichever the database happens to order first.
    name = models.CharField(
        max_length=200,
        unique=True,
        help_text="Human-readable name for this parking lot, e.g. 'Main Lot' or 'North Garage'.",
    )

    class Meta:
        verbose_name = 'parking lot'
        verbose_name_plural = 'parking lots'

    def __str__(self):
        return self.name


class LotSettings(models.Model):
    """
    Configuration settings for a specific parking lot.

    WHY SEPARATE FROM ParkingLot?
      Separation of concerns. ParkingLot is a stable identity record (just a name
      and an ID that's referenced by thousands of session records). LotSettings is
      operational configuration that changes regularly (operators adjust rates,
      grace periods, confidence thresholds as they tune the system).

      Keeping them separate means:
        - Changing a rate doesn't touch the lot record referenced by existing sessions.
        - Settings can be audited or versioned independently.
        - A lot record can exist before settings are configured.

    BILLING LOGIC (implemented in services.py, Day 7):
      1. duration ≤ grace_period_minutes → charge = $0.00 (free short stays)
      2. billing_unit='hour' → ceil(total_hours) × rate
      3. billing_unit='minute' → ceil(total_minutes) × rate
      4. daily_cap_enabled=True and charge > daily_cap_amount → charge = daily_cap_amount

    CONFIDENCE THRESHOLD:
      CV pipeline readings below confidence_threshold are flagged as low-confidence.
      They still create a session (best-guess plate text), but the event appears in
      the /errors/ queue so an operator can manually verify and correct the plate.
    """

    # OneToOneField enforces exactly one settings record per lot at the DB level.
    # related_name='settings' enables the shorthand: lot.settings (reverse access).
    lot = models.OneToOneField(
        ParkingLot,
        on_delete=models.CASCADE,
        related_name='settings',
        help_text="The parking lot these settings apply to.",
    )

    # ── Billing ───────────────────────────────────────────────────────────────

    # Rate per billing unit (hour or minute) in dollars.
    # max_digits=8, decimal_places=2 supports values up to $999,999.99.
    # WHY DecimalField? See module docstring — float arithmetic is imprecise for money.
    rate = models.DecimalField(
        max_digits=8,
        decimal_places=2,
        default=Decimal('5.00'),
        help_text="Rate per billing unit (in dollars). Always stored as Decimal, never float.",
    )

    BILLING_UNIT_CHOICES = [
        ('hour', 'Per Hour'),       # ceil(hours) × rate
        ('minute', 'Per Minute'),   # ceil(minutes) × rate
    ]
    billing_unit = models.CharField(
        max_length=10,
        choices=BILLING_UNIT_CHOICES,
        default='hour',
        help_text="Whether to charge per full hour or per full minute.",
    )

    # Free parking window at the start of a session.
    # Cars parked for ≤ grace_period_minutes are not charged.
    # Common use: 15 minutes for quick drop-offs, loading zones, etc.
    grace_period_minutes = models.IntegerField(
        default=15,
        validators=[MinValueValidator(0)],
        help_text="Sessions shorter than this many minutes are free (charged $0.00).",
    )

    # ── Daily Cap ─────────────────────────────────────────────────────────────

    # When True, no session charge can exceed daily_cap_amount.
    # Common at airport lots that offer all-day parking at a flat rate.
    daily_cap_enabled = models.BooleanField(
        default=False,
        help_text="If True, charge per session is capped at daily_cap_amount.",
    )

    # null=True: this field is meaningless when daily_cap_enabled=False.
    # blank=True: optional in forms — operators only fill it when enabling the cap.
    daily_cap_amount = models.DecimalField(
        max_digits=8,
        decimal_places=2,
        null=True,
        blank=True,
        help_text="Maximum charge per session. Only enforced when daily_cap_enabled=True.",
    )

    # ── Image Retention ───────────────────────────────────────────────────────

    # How many days to keep uploaded plate images on disk before cleanup.
    # null=True means "keep forever" — the cleanup command skips deletion.
    # The cleanup_old_images management command reads this setting.
    # MinValueValidator(1): 0 days would delete all images immediately; negative is nonsensical.
    image_retention_days = models.IntegerField(
        null=True,
        blank=True,
        validators=[MinValueValidator(1)],
        help_text="Delete uploaded images older than this many days. Null means keep forever.",
    )

    # ── CV Pipeline ───────────────────────────────────────────────────────────

    # Minimum acceptable confidence score (0.0 to 1.0) from the CV recognizer.
    # Readings below this threshold create a session but are flagged as low-confidence.
    # Default 0.6 means 60% — a reasonable starting threshold for synthetic-trained models.
    confidence_threshold = models.FloatField(
        default=0.6,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text=(
            "CV pipeline confidence threshold (0.0–1.0). "
            "Detections below this score are flagged as low-confidence and queued for review."
        ),
    )

    class Meta:
        verbose_name = 'lot settings'
        verbose_name_plural = 'lot settings'

    def __str__(self):
        return f'Settings for {self.lot.name}'


class ParkingSession(models.Model):
    """
    A single parking event — from when a car enters to when it exits.

    LIFECYCLE STATE MACHINE:
      ┌─────────┐   car exits normally    ┌───────────┐
      │ active  │ ─────────────────────> │ completed │
      └─────────┘                         └───────────┘
           │
           │ same plate enters again
           ▼
      ┌──────┐
      │ void │  (old session voided, new session opened with has_duplicate_warning=True)
      └──────┘

    REGISTERED vs GUEST SESSIONS:
      - Registered: license_plate and user are set (plate matched a LicensePlate record).
      - Guest: license_plate=null, user=null (no registered plate match).
      Both types are visible in the dashboard log.

    PLATE TEXT SOURCE OF TRUTH:
      plate_text stores the normalized plate string directly on the session.
      This is intentional — it remains correct even if the linked LicensePlate
      record is later edited or deleted (both FK fields are SET_NULL).
    """

    # The normalized plate text for this session.
    # This is the primary matching key used by handle_entry/handle_exit (Day 7).
    plate_text = models.CharField(
        max_length=20,
        help_text="Normalized license plate for this session (whitespace stripped, uppercase).",
    )

    # ── Registration Links (both nullable for guest support) ──────────────────

    # Linked registered plate, if the plate was found in the LicensePlate table.
    # SET_NULL: session survives if the plate record is deleted (preserves billing history).
    # null=True + blank=True: required for guest plates (no registered plate found).
    license_plate = models.ForeignKey(
        LicensePlate,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='sessions',
        help_text="Registered plate record linked to this session. Null for guest plates.",
    )

    # Linked user account, derived from license_plate.user.
    # Stored directly on the session for fast access without a JOIN.
    # SET_NULL: billing history survives user account deletion.
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='sessions',
        help_text="User who owns the plate. Null for guest plates.",
    )

    # ── Lot ───────────────────────────────────────────────────────────────────

    # Every session belongs to exactly one lot.
    # PROTECT: sessions are billing records — financial history must survive
    # lot deletion.  Deleting a lot that still has sessions raises
    # ProtectedError, forcing an explicit two-step process (archive or
    # reassign the sessions first).  CASCADE here would silently wipe revenue
    # history along with every linked PlateDetectionEvent.
    lot = models.ForeignKey(
        ParkingLot,
        on_delete=models.PROTECT,
        related_name='sessions',
        help_text="The parking lot where this session occurred.",
    )

    # ── Timing ────────────────────────────────────────────────────────────────

    # Set once when the session is created (handle_entry call).
    # auto_now_add is NOT used here because we want the entry time to reflect
    # when the CAR entered, not when Django processed the image (there may be
    # a small processing delay).
    entry_time = models.DateTimeField(
        help_text="UTC timestamp when the car entered the lot.",
    )

    # Null while the session is still active (car has not exited yet).
    # Set by handle_exit (Day 7) when the exit event is processed.
    exit_time = models.DateTimeField(
        null=True,
        blank=True,
        help_text="UTC timestamp when the car exited. Null if the session is still active.",
    )

    # WHY store duration_seconds?
    #   We could compute it as (exit_time - entry_time).total_seconds(), but:
    #   - Database aggregates (AVG, SUM) on derived values require expressing
    #     the derivation in SQL. Storing it as an integer makes analytics queries trivial.
    #   - The dashboard's "average session duration" stat is a simple ORM aggregate.
    duration_seconds = models.IntegerField(
        default=0,
        help_text="Total session duration in seconds. Calculated and stored at exit.",
    )

    # ── Billing ───────────────────────────────────────────────────────────────

    # Final charge in dollars. 0 during active sessions and for voided sessions.
    # Calculated by calculate_charge() in services.py using LotSettings at exit time.
    # max_digits=10, decimal_places=2 → up to $99,999,999.99 (extreme overkill).
    charge_amount = models.DecimalField(
        max_digits=10,
        decimal_places=2,
        default=Decimal('0.00'),
        validators=[MinValueValidator(Decimal('0.00'))],
        help_text="Final charge for this session in dollars. Always use Decimal, never float.",
    )

    # ── Status ────────────────────────────────────────────────────────────────

    STATUS_CHOICES = [
        ('active', 'Active'),         # Car is currently in the lot
        ('completed', 'Completed'),   # Car has exited normally
        ('void', 'Void'),             # Session was cancelled (orphan handling)
    ]
    # WHY no db_index here: the composite indexes in Meta (plate_text+status,
    # lot+status) already cover every status-filtered query, and a standalone
    # index on a 3-value column has selectivity too poor for PostgreSQL to
    # ever prefer it — it would be pure write overhead.
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default='active',
        help_text="Current lifecycle state of the session.",
    )

    # ── Duplicate / Orphan Flags ──────────────────────────────────────────────

    # Set on the NEW session when a plate enters while already having an active session.
    # Tells operators: "We auto-voided a previous session to open this one."
    has_duplicate_warning = models.BooleanField(
        default=False,
        help_text=(
            "True if this session was opened while the plate already had an active session. "
            "Indicates the previous session was auto-voided."
        ),
    )

    # Set on the OLD session when it's voided because the plate re-entered.
    # Tells operators: "This session was force-closed by a new entry event."
    was_orphaned = models.BooleanField(
        default=False,
        help_text=(
            "True if this session was voided because the same plate entered the lot again "
            "before this session was closed."
        ),
    )

    class Meta:
        verbose_name = 'parking session'
        verbose_name_plural = 'parking sessions'
        indexes = [
            # Used by handle_entry / handle_exit to find active sessions for a plate.
            # Query pattern: ParkingSession.objects.filter(plate_text=X, status='active')
            models.Index(
                fields=['plate_text', 'status'],
                name='session_plate_status_idx',
            ),
            # Used by the dashboard stats endpoint to count active sessions per lot.
            # Query pattern: ParkingSession.objects.filter(lot=lot, status='active').count()
            models.Index(
                fields=['lot', 'status'],
                name='session_lot_status_idx',
            ),
            # ── Partial indexes for the hot path ──────────────────────────────
            # WHY partial: active sessions are a tiny fraction of the table once
            # months of completed sessions accumulate.  These indexes cover only
            # the rows the entry/exit matcher and the 10-second dashboard poll
            # actually touch, so they stay small enough to live in cache while
            # the full composite indexes keep serving historical queries.
            models.Index(
                fields=['plate_text'],
                condition=models.Q(status='active'),
                name='session_active_plate_idx',
            ),
            models.Index(
                fields=['lot'],
                condition=models.Q(status='active'),
                name='session_active_lot_idx',
            ),
        ]
        constraints = [
            # WHY DB-level CheckConstraints (validators are not enough):
            # Django validators only run in full_clean()/ModelForm paths —
            # bulk_create, update(), and raw SQL bypass them entirely.  These
            # invariants protect revenue math, so they belong in the database.
            models.CheckConstraint(
                condition=models.Q(charge_amount__gte=Decimal('0.00')),
                name='session_charge_non_negative',
            ),
            # A car cannot exit before it entered; clock skew or a data-entry
            # bug would otherwise produce negative durations that corrupt the
            # average-duration dashboard stat.
            models.CheckConstraint(
                condition=models.Q(exit_time__isnull=True)
                | models.Q(exit_time__gt=models.F('entry_time')),
                name='session_exit_after_entry',
            ),
            models.CheckConstraint(
                condition=models.Q(duration_seconds__gte=0),
                name='session_duration_non_negative',
            ),
            # Voided sessions are excluded from revenue by definition — a void
            # session with a non-zero charge would silently inflate totals.
            models.CheckConstraint(
                condition=~models.Q(status='void')
                | models.Q(charge_amount=Decimal('0.00')),
                name='session_void_no_charge',
            ),
        ]

    def __str__(self):
        return f'{self.plate_text} — {self.status} (entered {self.entry_time})'


class PlateDetectionEvent(models.Model):
    """
    A single CV pipeline detection event — one uploaded image, one plate read.

    RELATIONSHIP TO ParkingSession:
      Each session normally has at least two events:
        1. An 'entry' event (car arrives, session is created)
        2. An 'exit' event (car leaves, session is completed)

      Low-confidence events may also appear without a clean session linkage,
      and corrections add implicit event-level changes tracked by manually_corrected.
      Those orphan review events still keep lot context so an operator's manual
      correction can safely reconcile them with the right active session.

    WHY STORE THE RAW IMAGE?
      Audit trail and manual correction. When the CV pipeline reads "ABE123" but
      the real plate is "ABC123", an operator needs to see the original image to
      verify the correct text. Images can be purged on a schedule controlled by
      LotSettings.image_retention_days.

    BOUNDING BOX FORMAT:
      [x, y, w, h] where all values are normalized to the range [0, 1] relative
      to the image dimensions. A box covering the full image would be [0, 0, 1, 1].
      Normalization makes the coordinates resolution-independent — the same bounding
      box is valid whether the original image was 640×480 or 1920×1080.
    """

    # FK to the session this detection belongs to.
    # SET_NULL: event records survive even if the session is deleted (rare).
    # null=True: theoretically an event could exist before session creation
    #            (though in practice session creation is atomic with event creation).
    session = models.ForeignKey(
        ParkingSession,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='detection_events',
        help_text="The parking session this detection is linked to.",
    )

    # Stored separately from session because unmatched exit events intentionally
    # have session=None until an operator corrects them. Without the lot, a
    # corrected exit could not safely choose which active same-plate session to
    # close in a multi-lot deployment.
    lot = models.ForeignKey(
        ParkingLot,
        on_delete=models.PROTECT,
        null=True,
        blank=True,
        related_name='detection_events',
        help_text="Parking lot where this detection occurred.",
    )

    # Django's ImageField does two things:
    #   1. Validates the uploaded file is a real image (using Pillow) — rejects .exe etc.
    #   2. Stores the relative path (under MEDIA_ROOT) in the database column.
    # upload_to='plates/' saves files to MEDIA_ROOT/plates/<filename>.
    # The actual file bytes are stored on disk, not in the database.
    #
    # SECURITY — enforce these controls in the /api/upload/ view (Day 8), NOT here:
    #   - File size cap (e.g. 10 MB): reject before Pillow opens the file to prevent
    #     memory exhaustion from a large TIFF or multi-layer PSD.
    #   - MIME type allowlist (JPEG, PNG only): Pillow accepts many formats (SVG via
    #     cairosvg, WebP, etc.) that may render in a browser and carry XSS payloads
    #     when served directly from MEDIA_URL.
    #   - Random filename: use upload_to=<callable> that generates a UUID filename
    #     to prevent filename-guessing enumeration of other users' plate images.
    image = models.ImageField(
        upload_to='plates/',
        help_text="Uploaded plate image. Stored in MEDIA_ROOT/plates/.",
    )

    # The plate text as returned by the CV recognizer, before normalize_plate().
    # We store the raw text to help operators audit what the pipeline originally read
    # versus what was corrected.
    raw_plate_text = models.CharField(
        max_length=20,
        help_text="Plate text as read by the CV pipeline before normalization.",
    )

    # Confidence score from the CV recognizer (0.0 = no confidence, 1.0 = certain).
    # Computed as the average per-character softmax probability from the CRNN output.
    # Scores below LotSettings.confidence_threshold trigger is_low_confidence=True.
    confidence_score = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text="CV recognizer confidence score, 0.0–1.0.",
    )

    EVENT_TYPE_CHOICES = [
        ('entry', 'Entry'),   # Car entering the lot
        ('exit', 'Exit'),     # Car leaving the lot
    ]
    event_type = models.CharField(
        max_length=10,
        choices=EVENT_TYPE_CHOICES,
        help_text="Whether this is an entry or exit detection event.",
    )

    # True when confidence_score < LotSettings.confidence_threshold.
    # These events surface in the /errors/ dashboard page for manual review.
    # A session IS still created with the best-guess plate text — we don't discard
    # low-confidence events, we flag them for human verification.
    is_low_confidence = models.BooleanField(
        default=False,
        help_text="True if confidence_score was below the lot's confidence_threshold.",
    )

    # ── Manual Correction Fields ───────────────────────────────────────────────

    # Set to True when an operator manually reviews and corrects this event.
    # Triggers re-matching of the session's plate_text to registered LicensePlate records.
    manually_corrected = models.BooleanField(
        default=False,
        help_text="True if an operator has manually verified and corrected this detection.",
    )

    # The operator-supplied correct plate text (after correction).
    # null=True: only set after manual correction.
    corrected_plate = models.CharField(
        max_length=20,
        null=True,
        blank=True,
        help_text="Operator-corrected plate text. Set only when manually_corrected=True.",
    )

    # ── Spatial Data ──────────────────────────────────────────────────────────

    # The bounding box [x, y, w, h] output by PlateDetectorCNN, normalized to [0, 1].
    # Stored as JSON so the dashboard can read it directly for canvas overlay rendering.
    # default=list creates a NEW empty list for each record — avoids the Python gotcha
    # where default=[] would share the same list object across all instances.
    bounding_box = models.JSONField(
        default=list,
        help_text="Plate bounding box [x, y, w, h] in normalized 0–1 coordinates.",
    )

    # auto_now_add=True: set once at insert, never updated.
    # WHY not derive timing from the session? Multiple events (entry + exit) each
    # need their own timestamp — they're separate uploads at different times.
    timestamp = models.DateTimeField(
        auto_now_add=True,
        help_text="UTC timestamp when this detection event was created.",
    )

    class Meta:
        verbose_name = 'plate detection event'
        verbose_name_plural = 'plate detection events'
        indexes = [
            # Used by the cleanup_old_images management command (Day 11).
            # Query pattern: PlateDetectionEvent.objects.filter(timestamp__lt=cutoff)
            models.Index(
                fields=['timestamp'],
                name='detection_event_timestamp_idx',
            ),
            # Used when an operator corrects an unmatched exit event and the
            # service must reconcile it with an active session in the same lot.
            models.Index(
                fields=['lot', 'event_type'],
                name='detection_lot_event_idx',
            ),
            # Used by the /errors/ review queue (Day 9).  Partial: only the
            # unreviewed low-confidence rows are indexed, so the queue page
            # stays fast even as total event volume grows.
            # Query pattern: .filter(is_low_confidence=True, manually_corrected=False)
            models.Index(
                fields=['is_low_confidence'],
                condition=models.Q(is_low_confidence=True, manually_corrected=False),
                name='detection_unreviewed_idx',
            ),
        ]
        constraints = [
            # DB-level guarantee that confidence stays in [0, 1] — the field
            # validators above only run in full_clean()/form paths, and this
            # value is written directly by the CV pipeline, not through forms.
            models.CheckConstraint(
                condition=models.Q(confidence_score__gte=0.0)
                & models.Q(confidence_score__lte=1.0),
                name='event_confidence_score_range',
            ),
        ]

    def __str__(self):
        return f'{self.event_type} — {self.raw_plate_text} ({self.timestamp})'
