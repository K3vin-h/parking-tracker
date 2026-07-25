"""
Resident wallet: balance, transaction history, and top-ups.

Scoped to request.user throughout — a user only ever sees and funds their own
wallet. Top-ups go through the payment connector and only credit the wallet after
that connector reports provider-confirmed success. The built-in placeholder fails
closed, so swapping in a real provider does not change this view's money flow.
"""

import logging

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import redirect, render

from apps.parking.payments import charge_payment_method
from apps.parking.wallet import credit_wallet, get_or_create_wallet

from .forms import TopupForm
from .ratelimit import rate_limit

logger = logging.getLogger("apps.public")


@login_required
def wallet(request):
    """Show the current user's balance and their ledger history (newest first)."""
    wallet = get_or_create_wallet(request.user)
    return render(
        request,
        "public/wallet.html",
        {
            "wallet": wallet,
            # Model Meta already orders newest-first; cap the page for safety.
            "transactions": wallet.transactions.all()[:100],
        },
    )


@login_required
@rate_limit(scope="topup", limit=20, window_seconds=300)
def topup(request):
    """
    Add funds to the current user's wallet via the payment connector.

    The wallet is credited ONLY after the connector confirms the charge succeeded.
    The placeholder connector rejects the request without changing the ledger.
    """
    if request.method == "POST":
        form = TopupForm(request.POST)
        if form.is_valid():
            amount = form.cleaned_data["amount"]
            payment = charge_payment_method(request.user, amount)
            if payment.success and payment.reference.strip():
                try:
                    credit_wallet(
                        request.user,
                        amount,
                        reference=payment.reference,
                        description="Wallet top-up",
                    )
                except ValueError:
                    # A conflicting provider reference is an integrity failure,
                    # not permission to mint a second credit.
                    logger.exception(
                        "Rejected invalid payment confirmation for user %s",
                        request.user.pk,
                    )
                    messages.error(
                        request,
                        "Payment confirmation could not be reconciled. "
                        "No funds were added.",
                    )
                    return render(request, "public/topup.html", {"form": form})
                messages.success(request, f"Added ${amount} to your balance.")
                return redirect("public:wallet")
            if payment.success:
                # A success boolean without a durable provider reference cannot
                # be reconciled or deduplicated, so it is not payment evidence.
                logger.error(
                    "Payment connector returned success without a confirmation "
                    "reference for user %s",
                    request.user.pk,
                )
                messages.error(
                    request,
                    "Payment confirmation was incomplete. No funds were added.",
                )
                return render(request, "public/topup.html", {"form": form})
            # No silent failure: the gateway declined, tell the user why.
            messages.error(
                request, payment.message or "Payment could not be processed."
            )
    else:
        form = TopupForm()

    return render(request, "public/topup.html", {"form": form})
