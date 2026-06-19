"""HTMX partial and inline-correction endpoints for the operator dashboard."""

import json

from django.db import transaction
from django.db.models import Q
from django.http import HttpRequest, JsonResponse, QueryDict
from django.shortcuts import render
from django.urls import reverse
from django.views.decorators.csrf import csrf_protect
from django.views.decorators.http import require_GET, require_http_methods

from apps.parking.models import PlateDetectionEvent
from apps.parking.services import correct_plate

from .api import staff_required
from .views import build_dashboard_context, build_session_context


@staff_required
@require_GET
def dashboard_stats(request: HttpRequest):
    """Render the single live region used by the 10-second dashboard poll."""
    return render(
        request,
        "partials/dashboard_stats.html",
        build_dashboard_context(request),
    )


@staff_required
@require_GET
def sessions(request: HttpRequest):
    """Render a filtered, 25-row session table for HTMX replacement."""
    response = render(
        request,
        "partials/session_table.html",
        build_session_context(request),
    )
    # WHY override HTMX history: the API returns only a table fragment, so a
    # refreshable or shareable browser URL must remain on the full log page.
    query_string = request.GET.urlencode()
    log_url = reverse("dashboard:log")
    response["HX-Push-Url"] = f"{log_url}?{query_string}" if query_string else log_url
    return response


@csrf_protect
@staff_required
@require_http_methods(["PATCH"])
def correct_event(request: HttpRequest, event_id: int):
    """
    Apply a manual correction through the transactional parking service.

    WHY parse QueryDict explicitly: Django populates request.POST only for POST,
    while HTMX sends the correction as URL-encoded PATCH per the API contract.
    """
    if request.content_type != "application/x-www-form-urlencoded":
        return JsonResponse({"error": "Use form-encoded PATCH data."}, status=415)
    data = QueryDict(request.body)
    corrected_text = data.get("corrected_plate", "")
    with transaction.atomic():
        event = (
            PlateDetectionEvent.objects.select_for_update().filter(pk=event_id).first()
        )
        if event is None:
            return JsonResponse({"error": "Detection event not found."}, status=404)
        if event.manually_corrected or not (
            event.is_low_confidence or event.session_id is None
        ):
            return JsonResponse(
                {"error": "Detection event is no longer pending review."},
                status=409,
            )
        try:
            event = correct_plate(
                event_id,
                corrected_text,
                locked_event=event,
            )
        except ValueError as exc:
            return JsonResponse({"error": str(exc)}, status=400)
        queue_count = PlateDetectionEvent.objects.filter(
            Q(is_low_confidence=True) | Q(session__isnull=True),
            manually_corrected=False,
        ).count()
    response = render(
        request,
        "partials/queue_corrected.html",
        {"event": event, "corrected": True, "queue_count": queue_count},
    )
    response["HX-Trigger"] = json.dumps({"queueCountChanged": {"count": queue_count}})
    return response
