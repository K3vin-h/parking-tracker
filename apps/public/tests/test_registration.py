"""
Tests for public account creation and post-login routing.

Security-critical: a self-service account must NEVER be staff, and a resident must
not reach the staff area.
"""

import pytest
from django.contrib.auth import get_user_model
from django.db import IntegrityError
from django.urls import reverse

from apps.parking.models import Wallet

User = get_user_model()

SIGNUP_URL = reverse("public:signup")


def _signup_data(username="resident1", email="resident1@example.com"):
    return {
        "username": username,
        "email": email,
        "password1": "s3cure-pass-99",
        "password2": "s3cure-pass-99",
    }


@pytest.mark.django_db
class TestSignup:
    def test_creates_non_staff_user_with_wallet(self, client):
        resp = client.post(SIGNUP_URL, _signup_data())
        assert resp.status_code == 302
        assert resp.url == reverse("public:wallet")

        user = User.objects.get(username="resident1")
        assert user.is_staff is False
        assert user.is_superuser is False
        assert Wallet.objects.filter(user=user).exists()

    def test_duplicate_email_rejected(self, client):
        User.objects.create_user(
            username="existing", email="dup@example.com", password="x"
        )
        resp = client.post(
            SIGNUP_URL, _signup_data(username="new", email="DUP@example.com")
        )
        assert resp.status_code == 200  # re-rendered form, not a redirect
        assert not User.objects.filter(username="new").exists()

    def test_wallet_provision_failure_rolls_back_user(self, client):
        """Account and its required companion record are one atomic creation."""
        from unittest.mock import patch

        with (
            patch(
                "apps.public.registration.get_or_create_wallet",
                side_effect=RuntimeError("provisioning failed"),
            ),
            pytest.raises(RuntimeError, match="provisioning failed"),
        ):
            client.post(SIGNUP_URL, _signup_data())

        assert not User.objects.filter(username="resident1").exists()

    def test_uniqueness_race_returns_a_safe_form_error(self, client):
        """A concurrent insert must not become a 500 or reveal database details."""
        from unittest.mock import patch

        with patch.object(
            User,
            "save",
            side_effect=IntegrityError("unique_user_email_ci"),
        ):
            response = client.post(SIGNUP_URL, _signup_data())

        assert response.status_code == 200
        assert b"could not be created" in response.content
        assert b"unique_user_email_ci" not in response.content

    def test_wallet_integrity_failure_is_not_mislabeled_as_user_input(self, client):
        """Provisioning invariant failures must remain explicit and roll back."""
        from unittest.mock import patch

        with (
            patch(
                "apps.public.registration.get_or_create_wallet",
                side_effect=IntegrityError("wallet invariant"),
            ),
            pytest.raises(IntegrityError, match="wallet invariant"),
        ):
            client.post(SIGNUP_URL, _signup_data())

        assert not User.objects.filter(username="resident1").exists()

    def test_authenticated_user_redirected_away(self, client):
        user = User.objects.create_user(
            username="already", email="a@e.com", password="x"
        )
        client.force_login(user)
        resp = client.get(SIGNUP_URL)
        assert resp.status_code == 302
        assert resp.url == reverse("public:wallet")


@pytest.mark.django_db
class TestAccessControl:
    def test_resident_cannot_reach_staff_area(self, client):
        client.post(SIGNUP_URL, _signup_data())
        # Logged in as the new resident; the staff dashboard must bounce to login.
        resp = client.get(reverse("dashboard:dashboard"))
        assert resp.status_code == 302
        assert "/login/" in resp.url


@pytest.mark.django_db
class TestPostLogin:
    def test_staff_routed_to_dashboard(self, client):
        staff = User.objects.create_user(
            username="op", email="op@e.com", password="x", is_staff=True
        )
        client.force_login(staff)
        resp = client.get(reverse("public:post_login"))
        assert resp.status_code == 302
        assert resp.url == reverse("dashboard:dashboard")

    def test_resident_routed_to_wallet(self, client):
        user = User.objects.create_user(username="res", email="r@e.com", password="x")
        client.force_login(user)
        resp = client.get(reverse("public:post_login"))
        assert resp.status_code == 302
        assert resp.url == reverse("public:wallet")
