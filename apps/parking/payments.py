"""
Payment gateway seam (PLACEHOLDER).

WHY THIS FILE EXISTS AS A SEAM:
  Top-ups need to take money from a real payment method (WeChat Pay, Alipay,
  Stripe, ...). Integrating a real provider is out of scope, but we must not
  scatter "pretend the payment worked" logic across the views. This module is the
  single, clearly-marked boundary where a real integration will drop in later.

  Everything above this line (wallet ledger, balance math) is production-grade and
  stays. Only `charge_payment_method` is a stub. Replacing it with a real SDK call
  is the entire integration effort.

SECURITY NOTE:
  A real implementation belongs behind server-side secrets (never in the client)
  and must verify the provider's response/signature before crediting a wallet.
  This placeholder holds NO secrets and always "succeeds" — it must never run in a
  deployment that handles real money without being replaced.
"""

import uuid
from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class PaymentResult:
    """Outcome of a charge against an external payment method."""

    success: bool
    reference: str
    message: str = ""


def charge_payment_method(user, amount: Decimal) -> PaymentResult:
    """
    Charge the user's external payment method for a top-up. PLACEHOLDER.

    A real integration would call the provider here, wait for confirmation, and
    return success only after the provider confirms the charge. This stub validates
    the amount and returns a synthetic reference so the wallet-crediting flow (in
    the view) is already correct end-to-end; swapping in the real call is all that
    remains.

    Returns a PaymentResult; the caller credits the wallet only when success=True.
    """
    if amount is None or Decimal(amount) <= 0:
        return PaymentResult(
            success=False, reference="", message="Amount must be positive."
        )

    # Synthetic, obviously-fake reference so it's clear in logs/history this did
    # not touch a real payment network.
    reference = f"placeholder-{uuid.uuid4().hex[:16]}"
    return PaymentResult(success=True, reference=reference)
