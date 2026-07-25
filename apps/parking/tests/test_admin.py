"""Security tests for audit-only parking records in Django admin."""

import pytest
from django.contrib.admin.sites import AdminSite
from django.test import RequestFactory

from apps.parking.admin import ParkingSessionAdmin, PlateDetectionEventAdmin
from apps.parking.models import ParkingSession, PlateDetectionEvent


@pytest.fixture
def admin_request():
    """Provide a minimal request because permission hooks accept one by contract."""
    return RequestFactory().get("/admin/")


@pytest.mark.parametrize(
    ("admin_class", "model"),
    [
        (ParkingSessionAdmin, ParkingSession),
        (PlateDetectionEventAdmin, PlateDetectionEvent),
    ],
)
def test_audit_records_cannot_be_added_changed_or_deleted(
    admin_request,
    admin_class,
    model,
):
    """Admin must not bypass service-owned audit-record lifecycle rules."""
    model_admin = admin_class(model, AdminSite())

    assert model_admin.has_add_permission(admin_request) is False
    assert model_admin.has_change_permission(admin_request) is False
    assert model_admin.has_delete_permission(admin_request) is False


def test_session_service_fields_are_read_only(admin_request):
    """Every persisted session field is service-owned and visible only for audit."""
    model_admin = ParkingSessionAdmin(ParkingSession, AdminSite())
    expected_fields = {
        field.name for field in ParkingSession._meta.concrete_fields
    }

    assert expected_fields <= set(model_admin.get_readonly_fields(admin_request))


def test_detection_event_fields_are_read_only(admin_request):
    """Detection evidence must be corrected through the audited correction service."""
    model_admin = PlateDetectionEventAdmin(PlateDetectionEvent, AdminSite())
    expected_fields = {
        field.name
        for field in PlateDetectionEvent._meta.concrete_fields
        if field.name != "image"
    }

    assert expected_fields <= set(model_admin.get_readonly_fields(admin_request))
