"""Security regression tests for bootstrap administrator identity handling."""

import pytest
from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.core.management.base import CommandError

User = get_user_model()


def _configure_bootstrap(monkeypatch):
    """Supply a strong deterministic credential without placing it in source config."""
    monkeypatch.setenv("DEFAULT_SUPERUSER_EMAIL", "Admin@Example.com")
    monkeypatch.setenv(
        "DEFAULT_SUPERUSER_PASSWORD",
        "Correct-Horse-Battery-Staple-90210",
    )


@pytest.mark.django_db
def test_setup_defaults_rejects_regular_user_email_conflict(monkeypatch):
    """Bootstrap must not report success when the configured admin is non-privileged."""
    _configure_bootstrap(monkeypatch)
    User.objects.create_user(
        username="resident",
        email="admin@example.COM",
        password="resident-password",
    )

    with pytest.raises(CommandError, match="already belongs"):
        call_command("setup_defaults")

    assert not User.objects.filter(is_superuser=True).exists()


@pytest.mark.django_db
def test_setup_defaults_accepts_matching_existing_superuser(monkeypatch):
    """A repeat run should remain idempotent for the same privileged identity."""
    _configure_bootstrap(monkeypatch)
    User.objects.create_superuser(
        username="admin@example.com",
        email="admin@example.com",
        password="existing-secure-password",
    )

    call_command("setup_defaults")

    assert User.objects.filter(
        email="admin@example.com",
        is_staff=True,
        is_superuser=True,
    ).count() == 1
