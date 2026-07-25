"""Regression tests for the fail-closed external payment connector seam."""

from decimal import Decimal

import pytest
from django.contrib.auth import get_user_model

from apps.parking.payments import charge_payment_method

User = get_user_model()


@pytest.mark.django_db
def test_placeholder_connector_never_confirms_a_charge():
    """An unconfigured connector must not authorize spendable wallet credit."""
    user = User.objects.create_user(username="payer", password="testpass123")

    result = charge_payment_method(user, Decimal("25.00"))

    assert result.success is False
    assert result.reference == ""
    assert "not configured" in result.message.lower()
