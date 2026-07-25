"""
Public account creation and post-login routing.

Account creation is a security boundary: a self-registered account is always a
plain resident (is_staff=False, is_superuser=False). Nothing here can grant staff
access — the form doesn't expose those fields and the view forces them off.
"""

import logging

from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from django.db import IntegrityError, transaction
from django.shortcuts import redirect, render
from django.urls import reverse

from apps.parking.wallet import get_or_create_wallet

from .forms import SignupForm
from .ratelimit import rate_limit

logger = logging.getLogger("apps.public")


class _SignupIdentityConflict(Exception):
    """Distinguish a raced user identity from unrelated provisioning failures."""


@rate_limit(scope="signup", limit=10, window_seconds=300)
def signup(request):
    """
    Create a resident account, provision an empty wallet, and log the user in.

    GET renders the form; POST validates and creates. On success the new account is
    guaranteed non-staff/non-superuser regardless of anything in the request.
    """
    if request.user.is_authenticated:
        return redirect("public:wallet")

    if request.method == "POST":
        form = SignupForm(request.POST)
        if form.is_valid():
            try:
                # User and required companion record are one unit. A provisioning
                # failure must not leave an account that crashes on first login.
                with transaction.atomic():
                    user = form.save(commit=False)
                    user.email = form.cleaned_data["email"]
                    # Belt-and-suspenders: self-service accounts are never privileged.
                    user.is_staff = False
                    user.is_superuser = False
                    try:
                        user.save()
                    except IntegrityError as exc:
                        # Translate only user-identity conflicts. Provisioning
                        # integrity failures below must remain explicit.
                        raise _SignupIdentityConflict from exc
                    # Keep the existing wallet provisioning behavior unchanged;
                    # only its transaction boundary is strengthened here.
                    get_or_create_wallet(user)
            except _SignupIdentityConflict:
                # Form pre-checks improve UX, while this handles concurrent
                # signups that race the database uniqueness constraint.
                logger.info("Resident signup lost a uniqueness race")
                form.add_error(
                    None,
                    "This account could not be created. Check your details and try again.",
                )
            else:
                # No authenticate() call was made, so name the backend explicitly.
                login(
                    request,
                    user,
                    backend="django.contrib.auth.backends.ModelBackend",
                )
                logger.info("New resident account created: user_id=%s", user.pk)
                return redirect("public:wallet")
    else:
        form = SignupForm()

    return render(request, "public/signup.html", {"form": form})


@login_required
def post_login(request):
    """
    Route users to the right home after login (one shared login page, two roles).

    Staff land on the operator dashboard; residents land on their wallet. This
    exists because LOGIN_REDIRECT_URL is a single global value but we now serve two
    kinds of authenticated user.
    """
    if request.user.is_staff:
        return redirect(reverse("dashboard:dashboard"))
    return redirect("public:wallet")
