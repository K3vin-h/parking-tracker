"""Staff-only HTML page views and shared dashboard query builders."""

from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal

from django.contrib import messages
from django.contrib.auth.mixins import UserPassesTestMixin
from django.contrib.auth.views import redirect_to_login
from django.core.paginator import Paginator
from django.db.models import Avg, Q, Sum
from django.http import Http404, HttpRequest
from django.shortcuts import redirect
from django.urls import reverse
from django.utils import timezone
from django.views.generic import TemplateView

from apps.parking.models import (
    LotSettings,
    ParkingLot,
    ParkingSession,
    PlateDetectionEvent,
)
from apps.parking.services import calculate_charge, normalize_plate

from .forms import LotSettingsForm

SESSION_PAGE_SIZE = 25


class StaffRequiredMixin(UserPassesTestMixin):
    """
    Redirect every unauthorized operator route through the normal login flow.

    WHY redirect authenticated non-staff too: the product has one operator role,
    and exposing a separate 403 page leaks route availability without helping a
    user who cannot gain access from this interface.
    """

    raise_exception = False

    def test_func(self):
        """Permit only authenticated staff accounts to reach operator data."""
        return self.request.user.is_authenticated and self.request.user.is_staff

    def handle_no_permission(self):
        """Redirect non-staff accounts instead of Django's default authenticated 403."""
        return redirect_to_login(
            self.request.get_full_path(),
            self.get_login_url(),
            self.get_redirect_field_name(),
        )


def _selected_lot(request: HttpRequest) -> ParkingLot | None:
    """
    Resolve an optional lot-id filter, with blank or ``all`` meaning all lots.

    WHY reject malformed/unknown ids: silently widening a bad filter to all lots
    could show an operator more location data than the UI indicated.
    """
    raw_lot = (request.GET.get("lot") or request.POST.get("lot") or "").strip()
    if not raw_lot or raw_lot == "all":
        return None
    try:
        lot_id = int(raw_lot)
    except ValueError:
        raise Http404("Unknown parking lot") from None
    try:
        return ParkingLot.objects.get(pk=lot_id)
    except ParkingLot.DoesNotExist:
        raise Http404("Unknown parking lot") from None


def _utc_day_bounds(day: date) -> tuple[datetime, datetime]:
    """Return an inclusive/exclusive UTC day range to avoid local-time drift."""
    start = datetime.combine(day, time.min, tzinfo=UTC)
    return start, start + timedelta(days=1)


def _format_duration(total_seconds: int) -> str:
    """Format durations compactly so live and completed values share one UI shape."""
    total_minutes = max(0, total_seconds) // 60
    hours, minutes = divmod(total_minutes, 60)
    return f"{hours}h {minutes}m" if hours else f"{minutes}m"


def _confidence_band(score: float) -> str:
    """Map confidence to the fixed green/yellow/red bands required by the UI."""
    if score >= 0.8:
        return "good"
    if score >= 0.6:
        return "warning"
    return "error"


def build_dashboard_context(request: HttpRequest) -> dict:
    """
    Build the complete live-dashboard payload from one consistent UTC instant.

    WHY attach running values in Python: active sessions have no stored duration
    or final charge yet, and the existing billing service is the single source
    of truth for grace periods, rounding, and caps.
    """
    selected_lot = _selected_lot(request)
    now_utc = timezone.now().astimezone(UTC)
    today_start, tomorrow_start = _utc_day_bounds(now_utc.date())

    sessions = ParkingSession.objects.select_related("lot", "lot__settings")
    events = PlateDetectionEvent.objects.select_related("lot", "session")
    if selected_lot is not None:
        sessions = sessions.filter(lot=selected_lot)
        events = events.filter(lot=selected_lot)

    active_sessions = list(
        sessions.filter(status=ParkingSession.Status.ACTIVE).order_by("entry_time")
    )
    for session in active_sessions:
        session.running_duration_seconds = max(
            0, int((now_utc - session.entry_time).total_seconds())
        )
        session.running_duration = _format_duration(session.running_duration_seconds)
        try:
            session.estimated_running_charge = calculate_charge(
                session.entry_time,
                now_utc,
                session.lot.settings,
            )
        except LotSettings.DoesNotExist:
            session.estimated_running_charge = Decimal("0.00")
        session.running_cost = session.estimated_running_charge

    recent_events = list(events.order_by("-timestamp", "-pk")[:10])
    for event in recent_events:
        event.confidence_percent = round(event.confidence_score * 100)
        event.confidence_band = _confidence_band(event.confidence_score)

    completed_today = sessions.filter(
        status=ParkingSession.Status.COMPLETED,
        exit_time__gte=today_start,
        exit_time__lt=tomorrow_start,
    )
    revenue_today = completed_today.aggregate(total=Sum("charge_amount"))["total"]
    average_stay = completed_today.aggregate(average=Avg("duration_seconds"))["average"]
    entries_today = events.filter(
        event_type="entry",
        timestamp__gte=today_start,
        timestamp__lt=tomorrow_start,
    ).count()
    exits_today = events.filter(
        event_type="exit",
        timestamp__gte=today_start,
        timestamp__lt=tomorrow_start,
    ).count()
    queue_filter = Q(is_low_confidence=True) | Q(session__isnull=True)
    queue = events.filter(queue_filter, manually_corrected=False)

    return {
        "active_count": len(active_sessions),
        "active_session_count": len(active_sessions),
        "revenue_today": revenue_today or Decimal("0.00"),
        "events_today": entries_today + exits_today,
        "entries_today": entries_today,
        "exits_today": exits_today,
        "average_stay_seconds": int(average_stay or 0),
        "average_stay": _format_duration(int(average_stay or 0)),
        "recent_events": recent_events,
        "active_sessions": active_sessions,
        "registered_active_count": sum(
            session.user_id is not None for session in active_sessions
        ),
        "guest_active_count": sum(
            session.user_id is None for session in active_sessions
        ),
        "queue_count": queue.count(),
        "lots": ParkingLot.objects.order_by("name"),
        "selected_lot": selected_lot,
        "now_utc": now_utc,
    }


def build_session_context(request: HttpRequest) -> dict:
    """
    Apply the log filters and paginate deterministically at 25 rows per page.

    WHY parse dates as UTC day boundaries: database timestamps are UTC and a
    date filter must not shift according to the server or browser locale.
    """
    selected_lot = _selected_lot(request)
    queryset = ParkingSession.objects.select_related(
        "lot", "lot__settings", "user"
    ).order_by("-entry_time", "-pk")
    if selected_lot is not None:
        queryset = queryset.filter(lot=selected_lot)

    status = (request.GET.get("status") or "").strip()
    if status in ParkingSession.Status.values:
        queryset = queryset.filter(status=status)

    raw_plate = request.GET.get("plate") or ""
    plate = normalize_plate(raw_plate) if raw_plate.strip() else ""
    if plate:
        queryset = queryset.filter(plate_text__icontains=plate)

    registration = (request.GET.get("registration") or "").strip()
    if registration == "registered":
        queryset = queryset.filter(user__isnull=False)
    elif registration == "guest":
        queryset = queryset.filter(user__isnull=True)

    date_from = _parse_iso_date(request.GET.get("date_from"))
    date_to = _parse_iso_date(request.GET.get("date_to"))
    if date_from:
        queryset = queryset.filter(
            entry_time__gte=datetime.combine(date_from, time.min, tzinfo=UTC)
        )
    if date_to:
        queryset = queryset.filter(
            entry_time__lt=datetime.combine(
                date_to + timedelta(days=1), time.min, tzinfo=UTC
            )
        )

    paginator = Paginator(queryset, SESSION_PAGE_SIZE)
    page_obj = paginator.get_page(request.GET.get("page"))
    now_utc = timezone.now().astimezone(UTC)
    for session in page_obj:
        if session.status == ParkingSession.Status.ACTIVE:
            running_seconds = max(0, int((now_utc - session.entry_time).total_seconds()))
            session.duration_display = _format_duration(running_seconds)
            try:
                session.display_charge = calculate_charge(
                    session.entry_time,
                    now_utc,
                    session.lot.settings,
                )
            except LotSettings.DoesNotExist:
                session.display_charge = Decimal("0.00")
        else:
            session.duration_display = _format_duration(session.duration_seconds)
            session.display_charge = session.charge_amount
    filters = {
        "status": status,
        "plate": request.GET.get("plate", ""),
        "registration": registration,
        "date_from": date_from.isoformat() if date_from else "",
        "date_to": date_to.isoformat() if date_to else "",
        "lot": str(selected_lot.pk) if selected_lot else "",
    }
    pagination_query = request.GET.copy()
    pagination_query.pop("page", None)
    return {
        "sessions": page_obj,
        "page_obj": page_obj,
        "paginator": paginator,
        "lots": ParkingLot.objects.order_by("name"),
        "selected_lot": selected_lot,
        "filters": filters,
        "pagination_query": pagination_query.urlencode(),
    }


def _parse_iso_date(value: str | None) -> date | None:
    """Ignore invalid optional UI dates instead of turning a log page into a 500."""
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def build_error_queue_context(request: HttpRequest) -> dict:
    """Return a bounded page of unresolved events for human review."""
    selected_lot = _selected_lot(request)
    events = PlateDetectionEvent.objects.select_related("lot", "session").filter(
        Q(is_low_confidence=True) | Q(session__isnull=True),
        manually_corrected=False,
    )
    if selected_lot is not None:
        events = events.filter(lot=selected_lot)
    paginator = Paginator(events.order_by("timestamp", "pk"), SESSION_PAGE_SIZE)
    page_obj = paginator.get_page(request.GET.get("page"))
    for event in page_obj:
        event.confidence_percent = round(event.confidence_score * 100)
        event.confidence_band = _confidence_band(event.confidence_score)
    return {
        "events": page_obj,
        "page_obj": page_obj,
        "paginator": paginator,
        "queue_count": paginator.count,
        "lots": ParkingLot.objects.order_by("name"),
        "selected_lot": selected_lot,
    }


def _unresolved_queue_count() -> int:
    """Keep the shared navigation badge consistent outside the queue page."""
    return PlateDetectionEvent.objects.filter(
        Q(is_low_confidence=True) | Q(session__isnull=True),
        manually_corrected=False,
    ).count()


def _shell_context(active_page: str) -> dict:
    """Provide real shared-shell values instead of prototype-only labels."""
    return {
        "active_page": active_page,
        "online_lot_count": ParkingLot.objects.count(),
        "queue_count": _unresolved_queue_count(),
    }


class DashboardView(StaffRequiredMixin, TemplateView):
    """Render the operator overview; live polling reuses the same context builder."""

    template_name = "dashboard.html"

    def get_context_data(self, **kwargs):
        """Add live statistics while preserving TemplateView's base context."""
        context = super().get_context_data(**kwargs)
        context.update(build_dashboard_context(self.request))
        context.update(_shell_context("dashboard"))
        return context


class UploadView(StaffRequiredMixin, TemplateView):
    """Render upload controls with every configured lot available for routing."""

    template_name = "upload.html"

    def get_context_data(self, **kwargs):
        """Expose lot choices without auto-selecting ambiguously in multi-lot setups."""
        context = super().get_context_data(**kwargs)
        context["lots"] = ParkingLot.objects.order_by("name")
        context["selected_lot"] = _selected_lot(self.request)
        context.update(_shell_context("upload"))
        return context


class LogView(StaffRequiredMixin, TemplateView):
    """Render the session-log shell and its initial filtered page."""

    template_name = "log.html"

    def get_context_data(self, **kwargs):
        """Reuse the API query contract so initial and HTMX results cannot diverge."""
        context = super().get_context_data(**kwargs)
        context.update(build_session_context(self.request))
        context.update(_shell_context("log"))
        return context


class ErrorQueueView(StaffRequiredMixin, TemplateView):
    """Render unresolved detections that require an operator decision."""

    template_name = "errors.html"

    def get_context_data(self, **kwargs):
        """Expose the same queue semantics used after correction."""
        context = super().get_context_data(**kwargs)
        context.update(build_error_queue_context(self.request))
        context.update(_shell_context("errors"))
        return context


class RevenueView(StaffRequiredMixin, TemplateView):
    """Render the analytics shell; Chart.js fetches values from the JSON endpoint."""

    template_name = "revenue.html"

    def get_context_data(self, **kwargs):
        """Provide selector state while keeping chart calculations in the API."""
        context = super().get_context_data(**kwargs)
        context["lots"] = ParkingLot.objects.order_by("name")
        context["selected_lot"] = _selected_lot(self.request)
        context["range_preset"] = self.request.GET.get("range", "30")
        context.update(_shell_context("revenue"))
        return context


class SettingsView(StaffRequiredMixin, TemplateView):
    """Load and save one lot's operational settings with explicit validation."""

    template_name = "settings.html"

    def _settings(self) -> LotSettings:
        """Select an explicit lot or the first configured lot for initial display."""
        selected_lot = _selected_lot(self.request)
        queryset = LotSettings.objects.select_related("lot").order_by("lot__name")
        if selected_lot is not None:
            queryset = queryset.filter(lot=selected_lot)
        settings = queryset.first()
        if settings is None:
            raise Http404("No settings configured for this parking lot")
        return settings

    def get_context_data(self, **kwargs):
        """Expose the bound lot and a percent-aware settings form."""
        context = super().get_context_data(**kwargs)
        lot_settings = kwargs.get("lot_settings") or self._settings()
        context.update(
            {
                "form": kwargs.get("form") or LotSettingsForm(instance=lot_settings),
                "lot_settings": lot_settings,
                "selected_lot": lot_settings.lot,
                "lots": ParkingLot.objects.order_by("name"),
                "confidence_threshold_percent": (
                    Decimal(str(lot_settings.confidence_threshold)) * Decimal("100")
                ),
            }
        )
        context.update(_shell_context("settings"))
        return context

    def post(self, request, *args, **kwargs):
        """Persist valid settings atomically through ModelForm validation."""
        lot_settings = self._settings()
        if request.POST.get("action") == "select_lot":
            # WHY redirect before binding: the lot picker is navigation, not a
            # partial settings submission that should trigger required-field errors.
            return redirect(f"{reverse('dashboard:settings')}?lot={lot_settings.lot_id}")
        form = LotSettingsForm(request.POST, instance=lot_settings)
        if not form.is_valid():
            return self.render_to_response(
                self.get_context_data(form=form, lot_settings=lot_settings)
            )
        form.save()
        messages.success(request, f"Settings saved for {lot_settings.lot.name}.")
        return redirect(f"{reverse('dashboard:settings')}?lot={lot_settings.lot_id}")
