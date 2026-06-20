"""Revenue analytics JSON endpoint for the operator dashboard."""

from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal

from django.db.models import Avg, Count, Sum
from django.db.models.functions import ExtractHour, TruncDate
from django.http import Http404, HttpRequest, JsonResponse
from django.views.decorators.http import require_GET

from apps.parking.models import ParkingLot, ParkingSession

from .api import staff_required


def _parse_revenue_range(request: HttpRequest) -> tuple[str, date, date]:
    """
    Resolve presets/custom dates as inclusive UTC calendar dates.

    WHY raise ValueError: malformed analytics input is a client error and must
    not silently fall back to a different financial reporting period.
    """
    preset = (request.GET.get("range") or "30").strip()
    today = datetime.now(UTC).date()
    if preset in {"7", "30", "90"}:
        days = int(preset)
        return preset, today - timedelta(days=days - 1), today
    if preset != "custom":
        raise ValueError("range must be 7, 30, 90, or custom.")
    try:
        start = date.fromisoformat(request.GET["start"])
        end = date.fromisoformat(request.GET["end"])
    except (KeyError, ValueError):
        raise ValueError("Custom ranges require valid start and end dates.") from None
    if end < start:
        raise ValueError("End date must be on or after start date.")
    if (end - start).days > 365:
        raise ValueError("Custom ranges cannot exceed 366 days.")
    return preset, start, end


def _revenue_lot(request: HttpRequest) -> ParkingLot | None:
    """Resolve the optional analytics lot filter without widening bad input."""
    raw_lot = (request.GET.get("lot") or "").strip()
    if not raw_lot or raw_lot == "all":
        return None
    try:
        return ParkingLot.objects.get(pk=int(raw_lot))
    except (ValueError, ParkingLot.DoesNotExist):
        raise Http404("Unknown parking lot") from None


def _money(value) -> str:
    """Serialize money as an exact two-decimal string for Chart.js consumers."""
    return str((value or Decimal("0.00")).quantize(Decimal("0.01")))


@staff_required
@require_GET
def revenue_data(request: HttpRequest) -> JsonResponse:
    """
    Return UTC revenue summaries plus dense daily, lot, and hourly chart series.

    Zero-filled dates/hours keep chart axes stable even when no sessions closed.
    """
    try:
        preset, start_date, end_date = _parse_revenue_range(request)
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    try:
        selected_lot = _revenue_lot(request)
    except Http404:
        # Keep the JSON contract intact: the Chart.js client cannot parse
        # Django's default HTML 404 page.
        return JsonResponse({"error": "Unknown parking lot."}, status=404)
    start = datetime.combine(start_date, time.min, tzinfo=UTC)
    end_exclusive = datetime.combine(end_date + timedelta(days=1), time.min, tzinfo=UTC)
    queryset = ParkingSession.objects.filter(
        status=ParkingSession.Status.COMPLETED,
        exit_time__gte=start,
        exit_time__lt=end_exclusive,
    )
    if selected_lot is not None:
        queryset = queryset.filter(lot=selected_lot)

    summary = queryset.aggregate(
        total_revenue=Sum("charge_amount"),
        session_count=Count("pk"),
        average_charge=Avg("charge_amount"),
        average_duration_seconds=Avg("duration_seconds"),
    )
    daily_rows = {
        row["day"]: row
        for row in queryset.annotate(day=TruncDate("exit_time", tzinfo=UTC))
        .values("day")
        .annotate(revenue=Sum("charge_amount"), session_count=Count("pk"))
        .order_by("day")
    }
    daily = []
    cursor = start_date
    while cursor <= end_date:
        row = daily_rows.get(cursor, {})
        daily.append(
            {
                "date": cursor.isoformat(),
                "revenue": _money(row.get("revenue")),
                "session_count": row.get("session_count", 0),
            }
        )
        cursor += timedelta(days=1)

    by_lot = [
        {
            "lot_id": row["lot_id"],
            "lot_name": row["lot__name"],
            "revenue": _money(row["revenue"]),
            "session_count": row["session_count"],
        }
        for row in queryset.values("lot_id", "lot__name")
        .annotate(revenue=Sum("charge_amount"), session_count=Count("pk"))
        .order_by("lot__name")
    ]
    hourly_rows = {
        row["hour"]: row
        for row in queryset.annotate(hour=ExtractHour("exit_time", tzinfo=UTC))
        .values("hour")
        .annotate(revenue=Sum("charge_amount"), session_count=Count("pk"))
        .order_by("hour")
    }
    hourly = [
        {
            "hour": hour,
            "revenue": _money(hourly_rows.get(hour, {}).get("revenue")),
            "session_count": hourly_rows.get(hour, {}).get("session_count", 0),
        }
        for hour in range(24)
    ]
    return JsonResponse(
        {
            "range": {
                "preset": preset,
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "timezone": "UTC",
            },
            "lot": (
                {"id": selected_lot.pk, "name": selected_lot.name}
                if selected_lot
                else None
            ),
            "summary": {
                "total_revenue": _money(summary["total_revenue"]),
                "session_count": summary["session_count"],
                "average_charge": _money(summary["average_charge"]),
                "average_duration_seconds": int(
                    summary["average_duration_seconds"] or 0
                ),
            },
            "daily": daily,
            "by_lot": by_lot,
            "hourly": hourly,
        }
    )
