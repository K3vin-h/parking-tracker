"""
Tests for the prepaid wallet: ledger integrity, top-ups, and auto-deduction.

WHAT THIS COVERS:
  - get_or_create_wallet: idempotent provisioning.
  - credit_wallet: credits + a matching ledger row; rejects non-positive top-ups.
  - debit_wallet_for_session: debits, allows negative balances, no-ops for $0 or
    guest sessions.
  - Integration through handle_exit: a registered plate is billed to the wallet;
    a guest plate is not.
  - The cardinal invariant: Wallet.balance == SUM(WalletTransaction.amount).

STYLE: mirrors test_services.py — @pytest.fixture, @pytest.mark.django_db,
plain assert, Decimal('X.XX') for money.
"""

from datetime import timedelta
from decimal import Decimal

import pytest
from django.contrib.auth import get_user_model
from django.db.models import Sum
from django.utils import timezone

from apps.parking.models import (
    LicensePlate,
    LotSettings,
    ParkingLot,
    ParkingSession,
    Wallet,
    WalletTransaction,
)
from apps.parking.services import handle_exit
from apps.parking.wallet import (
    credit_wallet,
    debit_wallet_for_session,
    get_or_create_wallet,
)

User = get_user_model()

PLATE_IMAGE = "plates/wallet_test.jpg"


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def user(db):
    return User.objects.create_user(
        username="walletuser", email="wallet@example.com", password="testpass123"
    )


@pytest.fixture
def parking_lot(db):
    return ParkingLot.objects.create(name="Wallet Lot")


@pytest.fixture
def lot_settings(parking_lot):
    return LotSettings.objects.create(
        lot=parking_lot,
        rate=Decimal("5.00"),
        billing_unit="hour",
        grace_period_minutes=15,
        confidence_threshold=0.6,
    )


@pytest.fixture
def registered_plate(user):
    return LicensePlate.objects.create(user=user, plate_text="ABC123")


def _reconciles(wallet: Wallet) -> bool:
    """The cached balance must equal the sum of the immutable ledger."""
    wallet.refresh_from_db()
    ledger_total = wallet.transactions.aggregate(total=Sum("amount"))[
        "total"
    ] or Decimal("0.00")
    return wallet.balance == ledger_total


# ── get_or_create_wallet ──────────────────────────────────────────────────────


@pytest.mark.django_db
class TestGetOrCreateWallet:
    def test_creates_wallet_with_zero_balance(self, user):
        wallet = get_or_create_wallet(user)
        assert wallet.balance == Decimal("0.00")
        assert Wallet.objects.filter(user=user).count() == 1

    def test_is_idempotent(self, user):
        first = get_or_create_wallet(user)
        second = get_or_create_wallet(user)
        assert first.pk == second.pk
        assert Wallet.objects.filter(user=user).count() == 1


# ── credit_wallet ─────────────────────────────────────────────────────────────


@pytest.mark.django_db
class TestCreditWallet:
    def test_credit_adds_balance_and_ledger_row(self, user):
        wallet = credit_wallet(user, Decimal("25.00"), reference="ref-1")
        assert wallet.balance == Decimal("25.00")
        txn = WalletTransaction.objects.get(wallet=wallet)
        assert txn.amount == Decimal("25.00")
        assert txn.kind == WalletTransaction.Kind.TOPUP
        assert txn.reference == "ref-1"
        assert _reconciles(wallet)

    def test_multiple_credits_accumulate(self, user):
        credit_wallet(user, Decimal("10.00"))
        wallet = credit_wallet(user, Decimal("5.50"))
        assert wallet.balance == Decimal("15.50")
        assert wallet.transactions.count() == 2
        assert _reconciles(wallet)

    def test_zero_topup_rejected(self, user):
        with pytest.raises(ValueError):
            credit_wallet(user, Decimal("0.00"))
        assert not WalletTransaction.objects.exists()

    def test_negative_topup_rejected(self, user):
        with pytest.raises(ValueError):
            credit_wallet(user, Decimal("-5.00"))


# ── debit_wallet_for_session ──────────────────────────────────────────────────


@pytest.mark.django_db
class TestDebitWallet:
    def _completed_session(self, user, lot, charge):
        return ParkingSession.objects.create(
            plate_text="ABC123",
            lot=lot,
            user=user,
            entry_time=timezone.now() - timedelta(hours=1),
            exit_time=timezone.now(),
            duration_seconds=3600,
            status=ParkingSession.Status.COMPLETED,
            charge_amount=charge,
        )

    def test_debit_reduces_balance_and_records_ledger(self, user, parking_lot):
        credit_wallet(user, Decimal("20.00"))
        session = self._completed_session(user, parking_lot, Decimal("5.00"))
        wallet = debit_wallet_for_session(session, Decimal("5.00"))
        assert wallet.balance == Decimal("15.00")
        charge_txn = WalletTransaction.objects.get(kind=WalletTransaction.Kind.CHARGE)
        assert charge_txn.amount == Decimal("-5.00")
        assert charge_txn.session_id == session.pk
        assert _reconciles(wallet)

    def test_low_balance_goes_negative(self, user, parking_lot):
        # No top-up: an underfunded account is allowed to owe (never blocked).
        session = self._completed_session(user, parking_lot, Decimal("8.00"))
        wallet = debit_wallet_for_session(session, Decimal("8.00"))
        assert wallet.balance == Decimal("-8.00")
        assert _reconciles(wallet)

    def test_zero_charge_writes_no_ledger_row(self, user, parking_lot):
        session = self._completed_session(user, parking_lot, Decimal("0.00"))
        result = debit_wallet_for_session(session, Decimal("0.00"))
        assert result is None
        assert not WalletTransaction.objects.filter(
            kind=WalletTransaction.Kind.CHARGE
        ).exists()

    def test_guest_session_is_not_debited(self, parking_lot):
        guest = ParkingSession.objects.create(
            plate_text="GUEST1",
            lot=parking_lot,
            user=None,
            entry_time=timezone.now() - timedelta(hours=1),
            exit_time=timezone.now(),
            duration_seconds=3600,
            status=ParkingSession.Status.COMPLETED,
            charge_amount=Decimal("5.00"),
        )
        assert debit_wallet_for_session(guest, Decimal("5.00")) is None
        assert not WalletTransaction.objects.exists()


# ── Integration through handle_exit ───────────────────────────────────────────


@pytest.mark.django_db
class TestExitDeductsWallet:
    def _active(self, lot, plate="ABC123", user=None, plate_obj=None, minutes_ago=90):
        return ParkingSession.objects.create(
            plate_text=plate,
            lot=lot,
            user=user,
            license_plate=plate_obj,
            entry_time=timezone.now() - timedelta(minutes=minutes_ago),
            status=ParkingSession.Status.ACTIVE,
        )

    def test_registered_exit_deducts_from_wallet(
        self, user, parking_lot, lot_settings, registered_plate
    ):
        credit_wallet(user, Decimal("20.00"))
        self._active(parking_lot, user=user, plate_obj=registered_plate)
        session = handle_exit("ABC123", 0.9, [], PLATE_IMAGE, parking_lot)

        assert session.charge_amount == Decimal("10.00")  # 90 min @ $5/hr → 2h
        wallet = Wallet.objects.get(user=user)
        assert wallet.balance == Decimal("10.00")
        assert wallet.transactions.filter(
            kind=WalletTransaction.Kind.CHARGE, session=session
        ).exists()
        assert _reconciles(wallet)

    def test_registered_exit_can_go_negative(
        self, user, parking_lot, lot_settings, registered_plate
    ):
        # Wallet starts empty; the $10 charge drives it negative (owed).
        self._active(parking_lot, user=user, plate_obj=registered_plate)
        handle_exit("ABC123", 0.9, [], PLATE_IMAGE, parking_lot)
        wallet = Wallet.objects.get(user=user)
        assert wallet.balance == Decimal("-10.00")
        assert _reconciles(wallet)

    def test_guest_exit_creates_no_wallet_or_ledger(self, parking_lot, lot_settings):
        self._active(parking_lot, plate="GUEST9", user=None)
        session = handle_exit("GUEST9", 0.9, [], PLATE_IMAGE, parking_lot)
        assert session.charge_amount == Decimal("10.00")  # still billed (kiosk display)
        assert not Wallet.objects.exists()
        assert not WalletTransaction.objects.exists()

    def test_registered_grace_exit_charges_nothing(
        self, user, parking_lot, lot_settings, registered_plate
    ):
        credit_wallet(user, Decimal("20.00"))
        self._active(parking_lot, user=user, plate_obj=registered_plate, minutes_ago=5)
        session = handle_exit("ABC123", 0.9, [], PLATE_IMAGE, parking_lot)
        assert session.charge_amount == Decimal("0.00")
        wallet = Wallet.objects.get(user=user)
        assert wallet.balance == Decimal("20.00")  # untouched
        assert not wallet.transactions.filter(
            kind=WalletTransaction.Kind.CHARGE
        ).exists()
