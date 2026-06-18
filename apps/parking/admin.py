"""
Admin configuration for the parking app.

Registers all five parking models with the Django admin, with
list_display, list_filter, and search_fields configured for
real parking lot operation use cases:

  - Finding a session by plate number
  - Reviewing active vs completed vs voided sessions
  - Inspecting low-confidence detections for manual correction
  - Checking lot settings (rates, grace periods, retention)
  - Managing registered license plates

All models are imported at the top and registered with @admin.register
to keep the code easy to scan.
"""

from django.contrib import admin

from apps.parking.models import (
    LicensePlate,
    LotSettings,
    ParkingLot,
    ParkingSession,
    PlateDetectionEvent,
)


@admin.register(LicensePlate)
class LicensePlateAdmin(admin.ModelAdmin):
    """Admin view for registered license plates."""

    # Show plate text, owner, primary flag, and label at a glance.
    list_display = ['plate_text', 'user', 'is_primary', 'label']
    # WHY list_select_related: rendering 'user' calls str(obj.user) per row —
    # without the JOIN that is one extra query per plate on every list page
    # load (classic N+1).
    list_select_related = ['user']
    list_filter = ['is_primary']
    # Operators can search by plate text or by the owning user's credentials.
    search_fields = ['plate_text', 'user__username', 'user__email']
    ordering = ['plate_text']


@admin.register(ParkingLot)
class ParkingLotAdmin(admin.ModelAdmin):
    """Admin view for parking lots."""

    list_display = ['name']
    search_fields = ['name']


@admin.register(LotSettings)
class LotSettingsAdmin(admin.ModelAdmin):
    """Admin view for lot-level billing and configuration settings."""

    list_display = [
        'lot',
        'rate',
        'billing_unit',
        'grace_period_minutes',
        'daily_cap_enabled',
        'daily_cap_amount',
        'image_retention_days',
        'confidence_threshold',
    ]
    # Avoid one query per row when rendering str(obj.lot) in the list view.
    list_select_related = ['lot']
    list_filter = ['billing_unit', 'daily_cap_enabled']


@admin.register(ParkingSession)
class ParkingSessionAdmin(admin.ModelAdmin):
    """
    Admin view for parking sessions.

    Designed for operators who need to:
      - Find a session by plate number (search_fields)
      - Monitor currently active sessions (status filter)
      - Review duplicate/orphan situations (list_display flags)
      - Audit charge amounts
    """

    list_display = [
        'plate_text',
        'lot',
        'status',
        'entry_time',
        'exit_time',
        'charge_amount',
        'has_duplicate_warning',
        'was_orphaned',
    ]
    # Avoid one query per row when rendering str(obj.lot) in the list view.
    list_select_related = ['lot']
    list_filter = ['status', 'lot', 'has_duplicate_warning', 'was_orphaned']
    search_fields = ['plate_text']
    # Newest sessions first — operators monitoring the lot want recent activity.
    ordering = ['-entry_time']
    # These fields are set programmatically by services.py.
    # Marking them read-only prevents accidental manual edits that would
    # corrupt billing records or session audit trail.
    readonly_fields = ['entry_time', 'exit_time', 'duration_seconds', 'charge_amount']


@admin.register(PlateDetectionEvent)
class PlateDetectionEventAdmin(admin.ModelAdmin):
    """
    Admin view for CV detection events.

    Useful for auditing what the CV pipeline read from each image,
    and for reviewing low-confidence detections before manual correction.
    """

    list_display = [
        'raw_plate_text',
        'lot',
        'event_type',
        'confidence_score',
        'is_low_confidence',
        'manually_corrected',
        'timestamp',
    ]
    # lot is stored directly because unmatched exit events have no session yet.
    list_select_related = ['lot', 'session']
    list_filter = ['lot', 'event_type', 'is_low_confidence', 'manually_corrected']
    search_fields = ['raw_plate_text', 'corrected_plate']
    ordering = ['-timestamp']
    # timestamp is auto-set at creation; bounding_box is set by the CV pipeline.
    # Both should be read-only to prevent accidental corruption.
    readonly_fields = ['timestamp', 'bounding_box']
