"""
Wallet money operations — the only place a balance is ever mutated.

WHY A DEDICATED MODULE:
  Money is the one thing that must never drift. Every credit (top-up) and debit
  (parking charge) flows through one of the three functions here, and each of them
  enforces the same invariant in the same way:

    1. Lock the wallet row (select_for_update) so two concurrent operations can't
       both read the old balance and clobber each other's write.
    2. Write ONE immutable WalletTransaction ledger row (the audit trail).
    3. Update the cached Wallet.balance by exactly the ledger amount.

  Because the cached balance only ever moves by the amount just written to the
  ledger, SUM(transactions.amount) == balance holds for all time. A test asserts
  this reconciliation.

DESIGN CONTRACT:
  - All money is Decimal, never float (the cardinal billing rule).
  - Callers that are already inside a transaction (e.g. handle_exit) get the debit
    folded into that same transaction, so the session completion and the wallet
    deduction commit or roll back together — a charge can never be recorded
    without its matching balance change, and vice versa.
"""

import logging
from decimal import Decimal

from django.db import transaction

from apps.parking.models import ParkingSession, Wallet, WalletTransaction

logger = logging.getLogger(__name__)

# Charges/top-ups are stored to the cent. Callers should already quantize money,
# but we quantize defensively so a stray extra place can never reach the DB.
_MONEY_QUANTUM = Decimal("0.01")


def get_or_create_wallet(user) -> Wallet:
    """
    Return the user's wallet, creating an empty one if it doesn't exist yet.

    WHY DEFENSIVE CREATION: signup creates a wallet up front, but older accounts
    (or fixtures) may predate wallets. Every money path calls this first so a
    missing wallet degrades to a $0.00 balance instead of an AttributeError.
    """
    wallet, _ = Wallet.objects.get_or_create(user=user)
    return wallet


@transaction.atomic
def credit_wallet(
    user, amount: Decimal, *, reference: str = "", description: str = ""
) -> Wallet:
    """
    Add funds to a user's wallet (a top-up) and record the credit in the ledger.

    Rejects non-positive top-ups: a "top-up" that removes or leaves money is a
    caller bug, and allowing it would let the top-up path be abused to forge
    debits. Deductions go through debit_wallet_for_session, not here.

    Returns the refreshed wallet. Raises ValueError on a non-positive amount.
    """
    amount = Decimal(amount).quantize(_MONEY_QUANTUM)
    if amount <= 0:
        raise ValueError("Top-up amount must be positive.")

    # Ensure the row exists, THEN lock it for the read-modify-write. Doing the
    # get_or_create separately (rather than select_for_update().get_or_create())
    # avoids a rare IntegrityError when two first-ever top-ups race to INSERT the
    # same OneToOne wallet — same pattern as debit_wallet_for_session.
    Wallet.objects.get_or_create(user=user)
    wallet = Wallet.objects.select_for_update().get(user=user)
    WalletTransaction.objects.create(
        wallet=wallet,
        amount=amount,  # positive = credit
        kind=WalletTransaction.Kind.TOPUP,
        reference=reference,
        description=description,
    )
    wallet.balance = wallet.balance + amount
    wallet.save(update_fields=["balance", "updated_at"])
    logger.info("Wallet %s credited %s (ref=%s)", wallet.pk, amount, reference or "-")
    return wallet


def debit_wallet_for_session(session: ParkingSession, charge: Decimal) -> Wallet | None:
    """
    Deduct a completed session's charge from the owning user's wallet.

    Called by _complete_session_for_exit INSIDE its open transaction, so the debit
    and the session completion are one atomic unit. Balance is allowed to go
    negative (product decision: an exit is never blocked for insufficient funds).

    A $0.00 charge (e.g. a grace-period stay) writes NO ledger row — there is no
    money movement to record. Returns the wallet, or None when there was nothing
    to charge or the session has no owner.

    PRECONDITION: session.user_id is set (a registered plate). Guests have no
    wallet and are billed at the kiosk instead — the caller must not call this for
    them.
    """
    if session.user_id is None:
        # Defensive: guests are billed at the kiosk, never through a wallet.
        logger.error(
            "debit_wallet_for_session called for a guest session %s", session.pk
        )
        return None

    charge = Decimal(charge).quantize(_MONEY_QUANTUM)
    if charge <= 0:
        # No money moved (free grace-period stay) — nothing to record.
        return None

    # Lock the wallet row within the caller's transaction. get_or_create covers
    # the rare pre-wallet account; the row is then locked for the update below.
    Wallet.objects.get_or_create(user_id=session.user_id)
    wallet = Wallet.objects.select_for_update().get(user_id=session.user_id)

    WalletTransaction.objects.create(
        wallet=wallet,
        amount=-charge,  # negative = debit
        kind=WalletTransaction.Kind.CHARGE,
        session=session,
        description=f"Parking charge for session {session.pk}",
    )
    wallet.balance = wallet.balance - charge
    wallet.save(update_fields=["balance", "updated_at"])
    logger.info(
        "Wallet %s debited %s for session %s (new balance %s)",
        wallet.pk,
        charge,
        session.pk,
        wallet.balance,
    )
    return wallet
