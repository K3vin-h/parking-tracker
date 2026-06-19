"""Tests for Day 10 dashboard partial and analytics API contracts."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import patch

import pytest
from django.contrib.auth import get_user_model
from django.http import HttpResponse
from django.urls import reverse

from apps.parking.models import (
    LotSettings,
    ParkingLot,
    ParkingSession,
    PlateDetectionEvent,
)

User = get_user_model()


@pytest.fixture
def api_data(db):
    """Create a staff user and two configured lots for API integration tests."""
    staff = User.objects.create_user(
        username="operator", password="testpass123", is_staff=True
    )
    regular = User.objects.create_user(
        username="driver", password="testpass123", is_staff=False
    )
    first = ParkingLot.objects.create(name="Alpha")
    second = ParkingLot.objects.create(name="Beta")
    for lot in (first, second):
        LotSettings.objects.create(lot=lot, rate=Decimal("5.00"))
    return staff, regular, first, second


@pytest.mark.django_db
class TestApiAuthorization:
    """Ensure every new API follows the same staff-only redirect policy."""

    @pytest.mark.parametrize(
        ("method", "route_name", "kwargs"),
        [
            ("get", "api_dashboard_stats", {}),
            ("get", "api_sessions", {}),
            ("get", "api_revenue_data", {}),
            ("patch", "api_correct_event", {"event_id": 1}),
        ],
    )
    def test_non_staff_is_redirected(
        self, client, api_data, method, route_name, kwargs
    ):
        """Redirect authenticated non-staff before reading parking data."""
        _, regular, _, _ = api_data
        client.force_login(regular)
        response = getattr(client, method)(
            reverse(f"dashboard:{route_name}", kwargs=kwargs)
        )
        assert response.status_code == 302
        assert "/login/" in response.url


@pytest.mark.django_db
class TestPartialApis:
    """Verify partial endpoints hand frontend templates the documented contexts."""

    def test_dashboard_stats_uses_live_context(self, client, api_data):
        """Render the dashboard partial with all live statistic keys."""
        staff, _, first, _ = api_data
        client.force_login(staff)
        with patch("apps.dashboard.api.render", return_value=HttpResponse()) as render:
            response = client.get(
                reverse("dashboard:api_dashboard_stats"), {"lot": first.pk}
            )
        assert response.status_code == 200
        context = render.call_args.args[2]
        assert {
            "active_session_count",
            "revenue_today",
            "entries_today",
            "exits_today",
            "recent_events",
            "active_sessions",
        } <= set(context)

    def test_sessions_partial_is_lot_filtered_and_paginated(self, client, api_data):
        """Pass a 25-row Page object and selected lot to the table partial."""
        staff, _, first, second = api_data
        now = datetime.now(UTC)
        wanted = ParkingSession.objects.create(
            plate_text="FIRST1", lot=first, entry_time=now
        )
        ParkingSession.objects.create(plate_text="SECOND", lot=second, entry_time=now)
        client.force_login(staff)
        with patch("apps.dashboard.api.render", return_value=HttpResponse()) as render:
            response = client.get(reverse("dashboard:api_sessions"), {"lot": first.pk})
        assert response.status_code == 200
        context = render.call_args.args[2]
        assert list(context["sessions"]) == [wanted]

    def test_sessions_partial_pushes_shareable_log_url(self, client, api_data):
        """Keep HTMX history on the full page while fetching the table API."""
        staff, _, first, _ = api_data
        client.force_login(staff)

        response = client.get(
            reverse("dashboard:api_sessions"),
            {"lot": first.pk, "status": "active", "plate": "ABC 123"},
            HTTP_HX_REQUEST="true",
        )

        expected_query = f"lot={first.pk}&status=active&plate=ABC+123"
        assert response.status_code == 200
        assert response["HX-Push-Url"] == (
            f"{reverse('dashboard:log')}?{expected_query}"
        )


@pytest.mark.django_db
class TestCorrectionApi:
    """Exercise the real transactional correction service through PATCH."""

    def test_patch_corrects_event_and_emits_queue_count(self, client, api_data):
        """Persist normalization and notify HTMX that the queue badge changed."""
        staff, _, first, _ = api_data
        event = PlateDetectionEvent.objects.create(
            lot=first,
            image="plates/review.jpg",
            raw_plate_text="bad 1",
            confidence_score=0.2,
            event_type="entry",
            is_low_confidence=True,
        )
        client.force_login(staff)
        with patch("apps.dashboard.api.render", return_value=HttpResponse()) as render:
            response = client.patch(
                reverse("dashboard:api_correct_event", args=[event.pk]),
                data="corrected_plate=abc+123",
                content_type="application/x-www-form-urlencoded",
            )
        event.refresh_from_db()
        assert response.status_code == 200
        assert event.corrected_plate == "ABC123"
        assert event.manually_corrected is True
        assert event.session_id is not None
        assert event.session.plate_text == "ABC123"
        assert event.session.status == ParkingSession.Status.ACTIVE
        assert "queueCountChanged" in response["HX-Trigger"]
        assert render.call_args.args[2]["queue_count"] == 0

    def test_patch_rejects_empty_correction(self, client, api_data):
        """Keep an unresolved event unchanged when the operator submits no plate."""
        staff, _, first, _ = api_data
        event = PlateDetectionEvent.objects.create(
            lot=first,
            image="plates/review.jpg",
            raw_plate_text="",
            confidence_score=0.0,
            event_type="entry",
            is_low_confidence=True,
        )
        client.force_login(staff)
        response = client.patch(
            reverse("dashboard:api_correct_event", args=[event.pk]),
            data="corrected_plate=",
            content_type="application/x-www-form-urlencoded",
        )
        assert response.status_code == 400
        event.refresh_from_db()
        assert event.manually_corrected is False

    def test_patch_renders_real_corrected_partial(self, client, api_data):
        """Catch missing or mismatched correction templates outside mocked tests."""
        staff, _, first, _ = api_data
        event = PlateDetectionEvent.objects.create(
            lot=first,
            image="plates/review.jpg",
            raw_plate_text="bad1",
            confidence_score=0.2,
            event_type="entry",
            is_low_confidence=True,
        )
        client.force_login(staff)
        response = client.patch(
            reverse("dashboard:api_correct_event", args=[event.pk]),
            data="corrected_plate=GOOD1",
            content_type="application/x-www-form-urlencoded",
        )
        assert response.status_code == 200
        assert b"Correction saved" in response.content

    def test_patch_rejects_already_corrected_event(self, client, api_data):
        """Prevent stale tabs or retries from rewriting a completed review."""
        staff, _, first, _ = api_data
        event = PlateDetectionEvent.objects.create(
            lot=first,
            image="plates/review.jpg",
            raw_plate_text="DONE1",
            confidence_score=0.2,
            event_type="entry",
            is_low_confidence=True,
            manually_corrected=True,
            corrected_plate="DONE1",
        )
        client.force_login(staff)
        response = client.patch(
            reverse("dashboard:api_correct_event", args=[event.pk]),
            data="corrected_plate=OTHER1",
            content_type="application/x-www-form-urlencoded",
        )
        assert response.status_code == 409
        event.refresh_from_db()
        assert event.corrected_plate == "DONE1"


@pytest.mark.django_db
class TestRevenueApi:
    """Verify exact-money UTC analytics and multi-lot filtering."""

    def test_returns_zero_filled_daily_and_hourly_series(self, client, api_data):
        """Keep both chart axes stable while reporting exact Decimal strings."""
        staff, _, first, second = api_data
        now = datetime.now(UTC)
        ParkingSession.objects.create(
            plate_text="PAID1",
            lot=first,
            entry_time=now - timedelta(hours=2),
            exit_time=now - timedelta(hours=1),
            duration_seconds=3600,
            charge_amount=Decimal("12.50"),
            status=ParkingSession.Status.COMPLETED,
        )
        ParkingSession.objects.create(
            plate_text="OTHER1",
            lot=second,
            entry_time=now - timedelta(hours=2),
            exit_time=now - timedelta(hours=1),
            duration_seconds=3600,
            charge_amount=Decimal("99.00"),
            status=ParkingSession.Status.COMPLETED,
        )
        client.force_login(staff)
        response = client.get(
            reverse("dashboard:api_revenue_data"),
            {"range": "7", "lot": first.pk},
        )
        body = response.json()
        assert response.status_code == 200
        assert body["summary"]["total_revenue"] == "12.50"
        assert body["summary"]["session_count"] == 1
        assert body["summary"]["average_duration_seconds"] == 3600
        assert len(body["daily"]) == 7
        assert len(body["hourly"]) == 24
        assert body["by_lot"] == [
            {
                "lot_id": first.pk,
                "lot_name": "Alpha",
                "revenue": "12.50",
                "session_count": 1,
            }
        ]

    def test_custom_range_validation_is_explicit(self, client, api_data):
        """Reject reversed dates rather than silently changing report boundaries."""
        staff, _, _, _ = api_data
        client.force_login(staff)
        response = client.get(
            reverse("dashboard:api_revenue_data"),
            {"range": "custom", "start": "2026-06-18", "end": "2026-06-01"},
        )
        assert response.status_code == 400
        assert "End date" in response.json()["error"]
