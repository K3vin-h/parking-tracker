"""Tests for staff authorization and shared dashboard page contexts."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from django.contrib.auth import get_user_model
from django.test import RequestFactory
from django.urls import reverse

from apps.dashboard.views import (
    SESSION_PAGE_SIZE,
    build_dashboard_context,
    build_error_queue_context,
    build_session_context,
)
from apps.parking.models import (
    LotSettings,
    ParkingLot,
    ParkingSession,
    PlateDetectionEvent,
)

User = get_user_model()


@pytest.fixture
def users(db):
    """Create both sides of the single operator-role authorization boundary."""
    staff = User.objects.create_user(
        username="operator", password="testpass123", is_staff=True
    )
    regular = User.objects.create_user(
        username="driver", password="testpass123", is_staff=False
    )
    return staff, regular


@pytest.fixture
def lots(db):
    """Create two configured lots for multi-lot filter assertions."""
    first = ParkingLot.objects.create(name="Alpha")
    second = ParkingLot.objects.create(name="Beta")
    for lot in (first, second):
        LotSettings.objects.create(
            lot=lot,
            rate=Decimal("5.00"),
            grace_period_minutes=0,
        )
    return first, second


@pytest.mark.django_db
class TestPageAuthorization:
    """Ensure every operator page redirects anonymous and non-staff users."""

    @pytest.mark.parametrize(
        "route_name",
        ["dashboard", "upload", "log", "errors", "revenue", "settings"],
    )
    def test_anonymous_user_is_redirected(self, client, route_name):
        """Prevent unauthenticated access before any page query or rendering."""
        response = client.get(reverse(f"dashboard:{route_name}"))
        assert response.status_code == 302
        assert "/login/" in response.url

    @pytest.mark.parametrize(
        "route_name",
        ["dashboard", "upload", "log", "errors", "revenue", "settings"],
    )
    def test_non_staff_user_is_redirected(self, client, users, route_name):
        """Apply the requested login redirect consistently to authenticated drivers."""
        _, regular = users
        client.force_login(regular)
        response = client.get(reverse(f"dashboard:{route_name}"))
        assert response.status_code == 302
        assert "/login/" in response.url


@pytest.mark.django_db
class TestDashboardContext:
    """Verify UTC statistics, live charges, and lot scoping."""

    def test_context_filters_lot_and_calculates_live_values(self, lots):
        """Keep one lot's dashboard isolated and calculate active values via billing."""
        first, second = lots
        now = datetime.now(UTC)
        active = ParkingSession.objects.create(
            plate_text="ACTIVE1",
            lot=first,
            entry_time=now - timedelta(hours=1, minutes=30),
        )
        ParkingSession.objects.create(
            plate_text="OTHER1",
            lot=second,
            entry_time=now - timedelta(hours=2),
        )
        PlateDetectionEvent.objects.create(
            session=active,
            lot=first,
            image="plates/active.jpg",
            raw_plate_text="ACTIVE1",
            confidence_score=0.9,
            event_type="entry",
        )
        request = RequestFactory().get("/", {"lot": first.pk})
        context = build_dashboard_context(request)
        assert context["active_session_count"] == 1
        assert context["active_sessions"][0].running_duration_seconds >= 5300
        assert context["active_sessions"][0].running_cost == Decimal("10.00")
        assert context["entries_today"] == 1
        assert context["exits_today"] == 0
        assert context["recent_events"][0].confidence_percent == 90


@pytest.mark.django_db
class TestSessionContext:
    """Verify filters and stable 25-row pagination for the log."""

    def test_filters_by_lot_status_registration_and_plate(self, lots, users):
        """Combine all supported filters rather than applying only the latest one."""
        first, second = lots
        staff, _ = users
        now = datetime.now(UTC)
        wanted = ParkingSession.objects.create(
            plate_text="ABC123",
            lot=first,
            user=staff,
            entry_time=now,
        )
        ParkingSession.objects.create(
            plate_text="ABC999",
            lot=second,
            entry_time=now,
        )
        request = RequestFactory().get(
            "/api/sessions/",
            {
                "lot": first.pk,
                "status": "active",
                "registration": "registered",
                "plate": "abc 123",
            },
        )
        context = build_session_context(request)
        assert list(context["sessions"]) == [wanted]

    def test_paginates_at_twenty_five_rows(self, lots):
        """Bound every HTMX response even when history grows large."""
        first, _ = lots
        now = datetime.now(UTC)
        ParkingSession.objects.bulk_create(
            [
                ParkingSession(
                    plate_text=f"P{index:05d}",
                    lot=first,
                    entry_time=now - timedelta(minutes=index),
                )
                for index in range(26)
            ]
        )
        context = build_session_context(RequestFactory().get("/log/"))
        assert SESSION_PAGE_SIZE == 25
        assert len(context["sessions"]) == 25
        assert context["paginator"].num_pages == 2


@pytest.mark.django_db
class TestErrorQueueContext:
    """Include unresolved low-confidence and unmatched events only."""

    def test_queue_excludes_corrected_events_and_filters_lot(self, lots):
        """Avoid resurfacing completed work or events from another selected lot."""
        first, second = lots
        pending = PlateDetectionEvent.objects.create(
            lot=first,
            image="plates/pending.jpg",
            raw_plate_text="PEND1",
            confidence_score=0.2,
            event_type="entry",
            is_low_confidence=True,
        )
        PlateDetectionEvent.objects.create(
            lot=first,
            image="plates/done.jpg",
            raw_plate_text="DONE1",
            confidence_score=0.2,
            event_type="entry",
            is_low_confidence=True,
            manually_corrected=True,
            corrected_plate="DONE1",
        )
        PlateDetectionEvent.objects.create(
            lot=second,
            image="plates/other.jpg",
            raw_plate_text="OTHER1",
            confidence_score=0.2,
            event_type="exit",
            is_low_confidence=True,
        )
        context = build_error_queue_context(
            RequestFactory().get("/errors/", {"lot": first.pk})
        )
        assert list(context["events"]) == [pending]
        assert context["queue_count"] == 1


@pytest.mark.django_db
class TestSettingsView:
    """Verify valid operator settings persist through the page POST flow."""

    def test_valid_post_updates_selected_lot(self, client, users, lots):
        """Save only the lot named in the query string and redirect after success."""
        staff, _ = users
        first, second = lots
        client.force_login(staff)
        response = client.post(
            f"{reverse('dashboard:settings')}?lot={first.pk}",
            {
                "lot": first.pk,
                "rate": "7.25",
                "billing_unit": "minute",
                "grace_period_minutes": "5",
                "daily_cap_enabled": "on",
                "daily_cap_amount": "25.00",
                "confidence_threshold": "82.5",
                "image_retention_days": "7",
            },
        )
        first.settings.refresh_from_db()
        second.settings.refresh_from_db()
        assert response.status_code == 302
        assert first.settings.rate == Decimal("7.25")
        assert first.settings.confidence_threshold == pytest.approx(0.825)
        assert first.settings.image_retention_days == 7
        assert second.settings.rate == Decimal("5.00")

    def test_lot_picker_redirects_without_validating_settings(
        self, client, users, lots
    ):
        """Treat the multi-lot picker as navigation rather than a partial save."""
        staff, _ = users
        _, second = lots
        client.force_login(staff)
        response = client.post(
            reverse("dashboard:settings"),
            {"lot": second.pk, "action": "select_lot"},
        )
        assert response.status_code == 302
        assert response.url == f"{reverse('dashboard:settings')}?lot={second.pk}"
