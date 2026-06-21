"""Tests for staff authorization and shared dashboard page contexts."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from django.contrib.auth import get_user_model
from django.test import RequestFactory
from django.urls import reverse

from apps.dashboard.utils import confidence_band
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
class TestSecurityHeaders:
    """Verify that security headers are present on authenticated dashboard responses."""

    def test_dashboard_response_sets_content_security_policy(self, client, users):
        """
        Every staff dashboard response must carry a Content-Security-Policy header.

        This catches misconfiguration of CSPMiddleware (e.g. removed from MIDDLEWARE
        or placed before SecurityMiddleware where it would be bypassed).
        The header must restrict script-src to 'self' with no unsafe directives,
        reflecting that all JS is self-hosted and HTMX eval is disabled.
        """
        staff, _ = users
        client.force_login(staff)
        response = client.get(reverse("dashboard:dashboard"))
        assert response.status_code == 200
        csp = response.get("Content-Security-Policy", "")
        assert csp, "Content-Security-Policy header must be present"
        assert "default-src 'self'" in csp
        assert "unsafe-eval" not in csp

        # Assert the script-src directive itself contains no unsafe keyword — a
        # substring check on the whole header would miss a regression that added
        # 'unsafe-inline' to script-src specifically.
        script_src = next(
            d.strip() for d in csp.split(";") if d.strip().startswith("script-src")
        )
        assert script_src == "script-src 'self'"


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
        assert context["recent_events"][0].confidence_band == "good"

    @pytest.mark.parametrize(
        ("score", "band"),
        [
            (0.8, "good"),
            (1.0, "good"),
            (0.6, "warning"),
            (0.79, "warning"),
            (0.59, "error"),
        ],
    )
    def test_confidence_bands_match_plan_thresholds(self, score, band):
        """Keep dashboard, upload, and queue colors on the locked thresholds."""
        assert confidence_band(score) == band


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
        assert context["filtered_queue_count"] == 1


@pytest.mark.django_db
class TestSettingsView:
    """Verify valid operator settings persist through the page POST flow."""

    def test_zero_values_render_without_falling_back_to_defaults(
        self, client, users, lots
    ):
        """Preserve valid zero settings so a later save cannot change them silently."""
        staff, _ = users
        first, _ = lots
        first.settings.confidence_threshold = 0
        first.settings.save(update_fields=["confidence_threshold"])
        client.force_login(staff)

        response = client.get(
            reverse("dashboard:settings"),
            {"lot": first.pk},
        )

        html = response.content.decode()
        grace_input = html.split('id="id_grace_period_minutes"', 1)[1].split(">", 1)[0]
        confidence_input = html.split('id="id_confidence_threshold"', 1)[1].split(
            ">", 1
        )[0]
        assert 'value="0"' in grace_input
        assert 'value="0.0"' in confidence_input
        assert (
            '<output class="range-value mono" for="id_confidence_threshold" '
            "data-confidence-output>0.0%</output>"
        ) in html

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

    def test_day_10_controls_render_as_tabs_slider_and_canvas(
        self, client, users, lots
    ):
        """Guard the exact PLAN controls instead of silently reverting to substitutes."""
        staff, _ = users
        client.force_login(staff)
        log = client.get(reverse("dashboard:log"))
        settings = client.get(reverse("dashboard:settings"))
        assert b'data-registration-tab="guest"' in log.content
        assert b'type="range"' in settings.content

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

    def test_invalid_post_rerenders_form_errors(self, client, users, lots):
        """Reject invalid settings without redirecting away from the form."""
        staff, _ = users
        first, _ = lots
        client.force_login(staff)
        response = client.post(
            f"{reverse('dashboard:settings')}?lot={first.pk}",
            {
                "lot": first.pk,
                "rate": "0",
                "billing_unit": "hour",
                "grace_period_minutes": "5",
                "confidence_threshold": "75",
                "image_retention_days": "30",
            },
        )
        assert response.status_code == 200
        assert b"0.01" in response.content
        first.settings.refresh_from_db()
        assert first.settings.rate == Decimal("5.00")

    def test_upload_page_renders_for_staff(self, client, users, lots):
        """Expose upload controls with lot choices on initial GET."""
        staff, _ = users
        client.force_login(staff)
        response = client.get(reverse("dashboard:upload"))
        assert response.status_code == 200
        assert b"Upload" in response.content

    def test_revenue_page_renders_with_range_preset(self, client, users, lots):
        """Render the analytics shell with the selected preset in context."""
        staff, _ = users
        client.force_login(staff)
        response = client.get(reverse("dashboard:revenue"), {"range": "7"})
        assert response.status_code == 200
        assert b"Revenue" in response.content

    def test_invalid_lot_filter_returns_404(self, client, users):
        """Reject malformed lot filters instead of silently widening scope."""
        staff, _ = users
        client.force_login(staff)
        response = client.get(reverse("dashboard:log"), {"lot": "abc"})
        assert response.status_code == 404
