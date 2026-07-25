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
from django.urls import reverse
from django.utils.html import format_html

from apps.parking.models import (
    LicensePlate,
    LotSettings,
    ParkingLot,
    ParkingSession,
    PlateDetectionEvent,
    Wallet,
    WalletTransaction,
)


class AuditOnlyAdminMixin:
    """Keep service-owned evidence viewable without permitting admin mutations."""

    def has_add_permission(self, request):
        """Creation must pass through services that enforce lifecycle invariants."""
        return False

    def has_change_permission(self, request, obj=None):
        """Corrections must use the audited application workflow."""
        return False

    def has_delete_permission(self, request, obj=None):
        """Audit evidence must not be removable through the generic admin."""
        return False


@admin.register(LicensePlate)
class LicensePlateAdmin(admin.ModelAdmin):
    """Admin view for registered license plates."""

    # Show plate text, owner, primary flag, and label at a glance.
    list_display = ["plate_text", "user", "is_primary", "label"]
    # WHY list_select_related: rendering 'user' calls str(obj.user) per row —
    # without the JOIN that is one extra query per plate on every list page
    # load (classic N+1).
    list_select_related = ["user"]
    list_filter = ["is_primary"]
    # Operators can search by plate text or by the owning user's credentials.
    search_fields = ["plate_text", "user__username", "user__email"]
    ordering = ["plate_text"]


@admin.register(ParkingLot)
class ParkingLotAdmin(admin.ModelAdmin):
    """Admin view for parking lots."""

    list_display = ["name"]
    search_fields = ["name"]


@admin.register(LotSettings)
class LotSettingsAdmin(admin.ModelAdmin):
    """Admin view for lot-level billing and configuration settings."""

    list_display = [
        "lot",
        "rate",
        "billing_unit",
        "grace_period_minutes",
        "daily_cap_enabled",
        "daily_cap_amount",
        "image_retention_days",
        "confidence_threshold",
    ]
    # Avoid one query per row when rendering str(obj.lot) in the list view.
    list_select_related = ["lot"]
    list_filter = ["billing_unit", "daily_cap_enabled"]


@admin.register(ParkingSession)
class ParkingSessionAdmin(AuditOnlyAdminMixin, admin.ModelAdmin):
    """
    Admin view for parking sessions.

    Designed for operators who need to:
      - Find a session by plate number (search_fields)
      - Monitor currently active sessions (status filter)
      - Review duplicate/orphan situations (list_display flags)
      - Audit charge amounts
    """

    list_display = [
        "plate_text",
        "lot",
        "status",
        "entry_time",
        "exit_time",
        "charge_amount",
        "has_duplicate_warning",
        "was_orphaned",
    ]
    # Avoid one query per row when rendering str(obj.lot) in the list view.
    list_select_related = ["lot"]
    list_filter = ["status", "lot", "has_duplicate_warning", "was_orphaned"]
    search_fields = ["plate_text"]
    # Newest sessions first — operators monitoring the lot want recent activity.
    ordering = ["-entry_time"]
    # These fields are set programmatically by services.py.
    # Marking them read-only prevents accidental manual edits that would
    # corrupt billing records or session audit trail.
    readonly_fields = [
        "id",
        "plate_text",
        "license_plate",
        "user",
        "lot",
        "entry_time",
        "exit_time",
        "duration_seconds",
        "charge_amount",
        "status",
        "has_duplicate_warning",
        "was_orphaned",
    ]


@admin.register(PlateDetectionEvent)
class PlateDetectionEventAdmin(AuditOnlyAdminMixin, admin.ModelAdmin):
    """
    Admin view for CV detection events.

    Useful for auditing what the CV pipeline read from each image,
    and for reviewing low-confidence detections before manual correction.
    """

    list_display = [
        "raw_plate_text",
        "lot",
        "event_type",
        "confidence_score",
        "is_low_confidence",
        "manually_corrected",
        "timestamp",
    ]
    # lot is stored directly because unmatched exit events have no session yet.
    list_select_related = ["lot", "session"]
    list_filter = ["lot", "event_type", "is_low_confidence", "manually_corrected"]
    search_fields = ["raw_plate_text", "corrected_plate"]
    ordering = ["-timestamp"]
    # The original ImageField is excluded so admin uploads cannot bypass the API's
    # size/format/dimension validation. Operators receive a read-only authenticated
    # link instead of Django's default public MEDIA_URL link.
    exclude = ["image"]
    readonly_fields = [
        "id",
        "session",
        "lot",
        "raw_plate_text",
        "confidence_score",
        "event_type",
        "is_low_confidence",
        "manually_corrected",
        "corrected_plate",
        "bounding_box",
        "timestamp",
        "private_image_link",
    ]

    @admin.display(description="Plate image")
    def private_image_link(self, obj):
        """Link through the staff-only image endpoint instead of public media."""
        if not obj.pk or not obj.image:
            return "No image"
        url = reverse("dashboard:api_event_image", args=[obj.pk])
        return format_html(
            '<a href="{}" target="_blank" rel="noopener">View private image</a>',
            url,
        )


class WalletTransactionInline(admin.TabularInline):
    """Read-only ledger rows shown inline on a wallet — the money audit trail."""

    model = WalletTransaction
    extra = 0
    can_delete = False
    # Ledger rows are immutable: never editable or addable through the admin.
    readonly_fields = [
        "amount",
        "kind",
        "session",
        "description",
        "reference",
        "created_at",
    ]

    def has_add_permission(self, request, obj=None):
        return False


@admin.register(Wallet)
class WalletAdmin(admin.ModelAdmin):
    """
    Admin view for prepaid wallets — lets staff see balances and arrears.

    Balance is read-only here: it is a cached total of the immutable ledger and
    must only ever move through apps/parking/wallet.py, never by hand (a manual
    edit would desync it from the transactions and corrupt the audit trail).
    """

    list_display = ["user", "balance", "updated_at"]
    list_select_related = ["user"]
    search_fields = ["user__username", "user__email"]
    ordering = ["balance"]  # arrears (most negative) first
    readonly_fields = ["user", "balance", "created_at", "updated_at"]
    inlines = [WalletTransactionInline]


@admin.register(WalletTransaction)
class WalletTransactionAdmin(admin.ModelAdmin):
    """Read-only audit view of every wallet credit and debit."""

    list_display = ["wallet", "amount", "kind", "session", "reference", "created_at"]
    list_select_related = ["wallet", "wallet__user", "session"]
    list_filter = ["kind"]
    search_fields = ["wallet__user__username", "wallet__user__email", "reference"]
    ordering = ["-created_at"]
    readonly_fields = [
        "wallet",
        "amount",
        "kind",
        "session",
        "description",
        "reference",
        "created_at",
    ]

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        # Viewable, never editable — the ledger is insert-only.
        return False

    def has_delete_permission(self, request, obj=None):
        return False
