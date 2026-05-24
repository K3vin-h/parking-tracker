"""
App configuration for the 'accounts' Django app.

The accounts app owns everything related to user identity:
  - The custom User model (Day 1)
  - Authentication (login/logout handled by Django's built-in views + our templates)

WHY a separate app just for the user model?
  Keeping User in its own app follows the single-responsibility principle.
  If we later add profile fields, avatars, or user preferences, they belong here
  without cluttering the parking business logic (apps/parking/).
"""

from django.apps import AppConfig


class AccountsConfig(AppConfig):
    """Configuration class for the accounts application."""

    # Default primary key type for models in this app.
    # Matches DEFAULT_AUTO_FIELD in settings.py — 64-bit integer IDs.
    default_auto_field = 'django.db.models.BigAutoField'

    # Must exactly match the Python module path relative to the project root.
    # Django uses this to find migrations, models, and signals.
    name = 'apps.accounts'

    # Human-readable name shown in the admin and error messages.
    verbose_name = 'Accounts'
