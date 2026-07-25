"""
Payment gateway seam (PLACEHOLDER).

WHY THIS FILE EXISTS AS A SEAM:
  Top-ups need to take money from a real payment method (WeChat Pay, Alipay,
  Stripe, ...). Integrating a real provider is out of scope, but we must not
  scatter "pretend the payment worked" logic across the views. This module is the
  single, clearly-marked boundary where a real integration will drop in later.

  Everything above this line (wallet ledger, balance math) is production-grade and
  stays. `PaymentConnector` is the narrow contract a real provider adapter will
  implement later.

SECURITY NOTE:
  A real implementation belongs behind server-side secrets (never in the client)
  and must verify the provider's response/signature before crediting a wallet.
  This placeholder holds NO secrets and deliberately FAILS CLOSED. The top-up page
  remains available, but no spendable credit is created until a real connector
  verifies a provider response and returns success.
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol


@dataclass(frozen=True)
class PaymentResult:
    """Outcome of a charge against an external payment method."""

    success: bool
    reference: str
    message: str = ""


class PaymentConnector(Protocol):
    """Define the provider-confirmation boundary required before wallet credit."""

    def charge(self, user, amount: Decimal) -> PaymentResult:
        """Return success only after the provider confirms the external charge."""
        ...


class PlaceholderPaymentConnector:
    """Keep the integration seam explicit while refusing unverified payments."""

    def charge(self, user, amount: Decimal) -> PaymentResult:
        """Validate the request, then fail because no payment provider is wired."""
        if amount is None or Decimal(amount) <= 0:
            return PaymentResult(
                success=False,
                reference="",
                message="Amount must be positive.",
            )
        return PaymentResult(
            success=False,
            reference="",
            message="Payment connector is not configured. Contact an operator.",
        )


# WHY A MODULE-LEVEL CONNECTOR: the view depends only on charge_payment_method,
# while a future provider adapter has one well-defined object to replace.
_payment_connector: PaymentConnector = PlaceholderPaymentConnector()


def charge_payment_method(user, amount: Decimal) -> PaymentResult:
    """
    Ask the configured connector to charge an external payment method.

    WHY THIS WRAPPER: callers have one stable function while the placeholder can
    later be replaced with a provider adapter that verifies signed confirmation.
    The caller credits the wallet only when the result explicitly succeeds.
    """
    return _payment_connector.charge(user, amount)
