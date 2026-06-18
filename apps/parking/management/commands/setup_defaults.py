"""
Management command: setup_defaults

Creates the initial data needed to run the parking tracker:
  1. A superuser account (for admin and dashboard access)
  2. A default ParkingLot called 'Main Lot'
  3. LotSettings linked to that lot with safe default values

IDEMPOTENT:
  This command is safe to run multiple times. It uses get_or_create() and
  checks for existing users before creating — running it twice won't create
  duplicates or overwrite existing settings.

USAGE:
  docker-compose exec web python manage.py setup_defaults

CREDENTIALS:
  Superuser credentials are read from environment variables:
    DEFAULT_SUPERUSER_EMAIL    — e.g., admin@example.com
    DEFAULT_SUPERUSER_PASSWORD — must be set in .env

WHY a management command instead of a fixture?
  Fixtures are JSON/YAML files with hardcoded data — they'd need to contain
  a hashed password, which changes every time Django updates its hasher.
  A management command reads credentials from env vars (more secure), generates
  the hash fresh, and provides helpful console output. It's also easier to
  extend with additional setup logic in the future.
"""

import os
from decimal import Decimal

from django.contrib.auth import get_user_model
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from apps.parking.models import LotSettings, ParkingLot

# Minimum length for the bootstrap superuser password. Django's default
# MinimumLengthValidator only requires 8; a privileged admin account warrants
# more, and the shipped .env.example placeholder must never produce a live login.
MIN_SUPERUSER_PASSWORD_LEN = 12


class Command(BaseCommand):
    """
    Create the default superuser, parking lot, and lot settings.

    This is the first command to run after 'manage.py migrate' on a fresh instance.
    """

    help = (
        "Create default superuser and initial ParkingLot + LotSettings. "
        "Safe to run multiple times — skips creation if records already exist."
    )

    def handle(self, *args, **options):
        """
        Main entry point for the management command.

        Django calls handle() when the command is invoked.

        WHY transaction.atomic: the three creation steps are one logical unit.
        Without it, a failure in _create_lot_settings would leave a lot with
        no settings — and because get_or_create makes re-runs report the lot
        as "already exists (skipped)", the missing settings would be masked
        on every subsequent run.  Atomic rollback keeps the database in a
        clean all-or-nothing state.

        The self.stdout.write() and self.style.SUCCESS() calls produce
        colored console output — green for success, yellow for skipped.
        """
        with transaction.atomic():
            self._create_superuser()
            lot = self._create_parking_lot()
            self._create_lot_settings(lot)

        self.stdout.write(self.style.SUCCESS("\nsetup_defaults complete. Ready to go!"))

    def _create_superuser(self):
        """
        Create the initial admin superuser from environment variables.

        WHY read from environment variables?
          Hardcoding credentials in source code is a security vulnerability.
          Using env vars keeps secrets out of the codebase and allows different
          credentials per environment (dev vs staging vs production).
        """
        User = get_user_model()

        email = os.environ.get("DEFAULT_SUPERUSER_EMAIL", "").strip()
        password = os.environ.get("DEFAULT_SUPERUSER_PASSWORD", "").strip()

        # Validate that required env vars are set — fail clearly rather than
        # creating an account with an empty password or empty email.
        if not email:
            raise CommandError(
                "DEFAULT_SUPERUSER_EMAIL environment variable is not set. "
                "Add it to your .env file and try again."
            )
        if not password:
            raise CommandError(
                "DEFAULT_SUPERUSER_PASSWORD environment variable is not set. "
                "Add it to your .env file and try again."
            )

        # Reject weak/placeholder passwords before creating a privileged account.
        # Explicit length floor first (stricter than Django's default), then the
        # project's configured AUTH_PASSWORD_VALIDATORS (common-password list etc.).
        if len(password) < MIN_SUPERUSER_PASSWORD_LEN:
            raise CommandError(
                "DEFAULT_SUPERUSER_PASSWORD is too short "
                f"(minimum {MIN_SUPERUSER_PASSWORD_LEN} characters). "
                "Choose a stronger password in your .env file."
            )
        try:
            validate_password(password)
        except ValidationError as exc:
            raise CommandError(
                "DEFAULT_SUPERUSER_PASSWORD is too weak: " + " ".join(exc.messages)
            )

        # Use the email as the username as well, for simplicity.
        # filter().exists() is preferred over get() because it returns False
        # instead of raising an exception when the user doesn't exist.
        if User.objects.filter(email=email).exists():
            self.stdout.write("  Superuser already exists (skipped)")
            return

        # create_superuser() hashes the password before storing it.
        # NEVER store plain-text passwords — even in development.
        User.objects.create_superuser(
            username=email,
            email=email,
            password=password,
        )
        # WHY no email in the output: this command runs during container
        # initialization, so stdout lands in CI/CD build logs.  The admin
        # email is a username and PII — printing it there invites targeted
        # phishing.
        self.stdout.write(self.style.SUCCESS("  Superuser created successfully."))

    def _create_parking_lot(self):
        """
        Create the default parking lot if it doesn't already exist.

        Returns the ParkingLot instance (new or existing) so it can be
        passed to _create_lot_settings().
        """
        # get_or_create() returns (object, created_bool) — we unpack both.
        # If the lot already exists, it returns the existing one with created=False.
        lot, created = ParkingLot.objects.get_or_create(name="Main Lot")

        if created:
            self.stdout.write(self.style.SUCCESS('  ParkingLot created: "Main Lot"'))
        else:
            self.stdout.write('  ParkingLot "Main Lot" already exists (skipped)')

        return lot

    def _create_lot_settings(self, lot):
        """
        Create LotSettings for the given lot with sensible default values.

        Default values are chosen to be safe and operator-friendly:
          - $5.00/hour is a mid-range urban parking rate
          - 15-minute grace period accommodates quick drop-offs
          - 0.6 confidence threshold is a reasonable starting point for
            synthetic-trained CV models (operators can tune this up as the
            model improves)
          - No daily cap (disabled) — simple use case for a single lot
          - Image retention null = keep forever (can be configured in settings page)
        """
        settings_obj, created = LotSettings.objects.get_or_create(
            lot=lot,
            defaults={
                "rate": Decimal("5.00"),
                "billing_unit": "hour",
                "grace_period_minutes": 15,
                "daily_cap_enabled": False,
                "daily_cap_amount": None,
                "image_retention_days": None,
                "confidence_threshold": 0.6,
            },
        )

        if created:
            self.stdout.write(
                self.style.SUCCESS(
                    f'  LotSettings created for "{lot.name}": '
                    f"$5.00/hr, 15-min grace, confidence≥0.6"
                )
            )
        else:
            self.stdout.write(
                f'  LotSettings for "{lot.name}" already exists (skipped)'
            )
