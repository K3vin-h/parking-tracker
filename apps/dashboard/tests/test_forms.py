"""Tests for operator-facing lot settings validation and value conversion."""

from decimal import Decimal

import pytest

from apps.dashboard.forms import LotSettingsForm
from apps.parking.models import LotSettings, ParkingLot


@pytest.fixture
def lot_settings(db):
    """Create editable settings with values that exercise every custom field."""
    lot = ParkingLot.objects.create(name="North Lot")
    return LotSettings.objects.create(
        lot=lot,
        rate=Decimal("5.00"),
        billing_unit="hour",
        grace_period_minutes=15,
        daily_cap_enabled=False,
        confidence_threshold=0.6,
        image_retention_days=30,
    )


def _form_data(**overrides):
    """Return a complete valid payload so tests isolate one validation rule."""
    data = {
        "rate": "6.50",
        "billing_unit": "hour",
        "grace_period_minutes": "10",
        "daily_cap_enabled": "",
        "daily_cap_amount": "",
        "confidence_threshold": "75",
        "image_retention_days": "90",
    }
    data.update(overrides)
    return data


@pytest.mark.django_db
class TestLotSettingsForm:
    """Protect billing invariants and translate UI-only settings values."""

    def test_initial_values_are_operator_friendly(self, lot_settings):
        """Display confidence as percent and retention as a supported choice."""
        form = LotSettingsForm(instance=lot_settings)
        assert form.initial["confidence_threshold"] == Decimal("60.0")
        assert form.initial["image_retention_days"] == 30

    def test_save_converts_percent_and_retention(self, lot_settings):
        """Store percent as 0–1 and the selected finite retention as an integer."""
        form = LotSettingsForm(_form_data(), instance=lot_settings)
        assert form.is_valid(), form.errors
        saved = form.save()
        assert saved.confidence_threshold == pytest.approx(0.75)
        assert saved.image_retention_days == 90
        assert saved.daily_cap_amount is None

    def test_forever_retention_stores_null(self, lot_settings):
        """Map the forever choice to NULL for the cleanup command's skip policy."""
        form = LotSettingsForm(
            _form_data(image_retention_days=""),
            instance=lot_settings,
        )
        assert form.is_valid(), form.errors
        assert form.save().image_retention_days is None

    def test_enabled_cap_requires_positive_amount(self, lot_settings):
        """Reject an enabled cap with no amount so billing never guesses."""
        form = LotSettingsForm(
            _form_data(daily_cap_enabled="on", daily_cap_amount=""),
            instance=lot_settings,
        )
        assert not form.is_valid()
        assert "daily_cap_amount" in form.errors

    def test_rejects_invalid_ranges_and_retention(self, lot_settings):
        """Reject unsafe billing, grace, confidence, and retention values."""
        form = LotSettingsForm(
            _form_data(
                rate="0",
                grace_period_minutes="1441",
                confidence_threshold="101",
                image_retention_days="14",
            ),
            instance=lot_settings,
        )
        assert not form.is_valid()
        assert {
            "rate",
            "grace_period_minutes",
            "confidence_threshold",
            "image_retention_days",
        } <= set(form.errors)
