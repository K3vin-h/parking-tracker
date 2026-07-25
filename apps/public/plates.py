"""
Resident plate management — strictly scoped to the logged-in user's own plates.

OWNERSHIP IS THE WHOLE POINT HERE: every read and write is filtered by
`user=request.user`, so one account can never see, add to, or delete another
account's plates. Deletes are POST-only + CSRF-protected (state change).
"""

import logging

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.db import IntegrityError, transaction
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_POST

from apps.parking.models import LicensePlate

from .forms import PlateForm

logger = logging.getLogger("apps.public")


@login_required
def plates(request):
    """List the current user's plates and add a new one."""
    if request.method == "POST":
        form = PlateForm(request.POST)
        if form.is_valid():
            try:
                # A savepoint (atomic block) is REQUIRED to catch IntegrityError:
                # without it the failed INSERT poisons the surrounding transaction
                # and every later query (e.g. re-rendering the plate list) raises
                # TransactionManagementError. The savepoint rolls back just the
                # INSERT so the request can continue.
                with transaction.atomic():
                    LicensePlate.objects.create(
                        user=request.user,
                        plate_text=form.cleaned_data["plate_text"],
                        label=form.cleaned_data.get("label", ""),
                    )
                messages.success(request, "Plate added.")
                return redirect("public:plates")
            except IntegrityError:
                # The canonical plate already has an owner. Keep the message generic
                # so the public form never discloses which account owns it.
                form.add_error("plate_text", "This plate is already registered.")
    else:
        form = PlateForm()

    return render(
        request,
        "public/plates.html",
        {
            "form": form,
            # Scoped to the owner — never a global plate list.
            "plates": request.user.plates.order_by("plate_text"),
        },
    )


@login_required
@require_POST
def plate_delete(request, pk: int):
    """Delete one of the current user's own plates (404 if not theirs)."""
    # Filtering by user makes cross-account deletion impossible: another user's pk
    # simply 404s instead of deleting.
    plate = get_object_or_404(LicensePlate, pk=pk, user=request.user)
    plate.delete()
    messages.success(request, "Plate removed.")
    return redirect("public:plates")
