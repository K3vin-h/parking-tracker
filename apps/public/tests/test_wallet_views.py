"""
Tests for the resident wallet views (balance, history, placeholder top-up).
"""

from decimal import Decimal

import pytest
from django.contrib.auth import get_user_model
from django.urls import reverse

from apps.parking.models import Wallet, WalletTransaction
from apps.parking.payments import PaymentResult

User = get_user_model()

WALLET_URL = reverse("public:wallet")
TOPUP_URL = reverse("public:topup")


@pytest.fixture
def user(db):
    return User.objects.create_user(
        username="walletowner", email="w@e.com", password="x"
    )


@pytest.mark.django_db
class TestWalletAuth:
    def test_wallet_requires_login(self, client):
        resp = client.get(WALLET_URL)
        assert resp.status_code == 302
        assert "/login/" in resp.url

    def test_topup_requires_login(self, client):
        resp = client.get(TOPUP_URL)
        assert resp.status_code == 302
        assert "/login/" in resp.url


@pytest.mark.django_db
class TestWalletViews:
    def test_wallet_page_renders_balance(self, client, user):
        client.force_login(user)
        resp = client.get(WALLET_URL)
        assert resp.status_code == 200

    def test_topup_credits_only_after_connector_confirmation(
        self, client, user, monkeypatch
    ):
        """A provider-confirmed charge may become spendable wallet credit."""
        monkeypatch.setattr(
            "apps.public.wallet_views.charge_payment_method",
            lambda _user, _amount: PaymentResult(
                success=True,
                reference="provider-confirmed-123",
            ),
        )
        client.force_login(user)
        resp = client.post(TOPUP_URL, {"amount": "25.00"})
        assert resp.status_code == 302
        assert resp.url == WALLET_URL

        wallet = Wallet.objects.get(user=user)
        assert wallet.balance == Decimal("25.00")
        txn = wallet.transactions.get(kind=WalletTransaction.Kind.TOPUP)
        assert txn.amount == Decimal("25.00")
        assert txn.reference == "provider-confirmed-123"

    def test_unconfigured_connector_does_not_credit_wallet(self, client, user):
        """The placeholder keeps top-up UI available without minting funds."""
        client.force_login(user)

        resp = client.post(TOPUP_URL, {"amount": "25.00"})

        assert resp.status_code == 200
        assert b"not configured" in resp.content.lower()
        assert not WalletTransaction.objects.exists()
        assert not Wallet.objects.filter(
            user=user,
            balance__gt=Decimal("0.00"),
        ).exists()

    def test_connector_success_without_reference_does_not_credit(
        self, client, user, monkeypatch
    ):
        """A boolean alone is not durable proof that an external charge occurred."""
        monkeypatch.setattr(
            "apps.public.wallet_views.charge_payment_method",
            lambda _user, _amount: PaymentResult(success=True, reference=""),
        )
        client.force_login(user)

        resp = client.post(TOPUP_URL, {"amount": "25.00"})

        assert resp.status_code == 200
        assert b"confirmation" in resp.content.lower()
        assert not WalletTransaction.objects.exists()

    def test_topup_rejects_zero(self, client, user):
        client.force_login(user)
        resp = client.post(TOPUP_URL, {"amount": "0"})
        assert resp.status_code == 200  # form error, not a redirect
        assert not WalletTransaction.objects.exists()
