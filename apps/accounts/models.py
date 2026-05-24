"""
Custom User model for the parking tracker.

THE GOLDEN RULE OF DJANGO USER MODELS:
  Always define a custom User model at the START of a new project — before your
  first migration. If you skip this and use Django's built-in User model, you can
  never change it later without wiping your entire database and rebuilding every
  table that references it (since ForeignKeys would point to the old model).

  Django's own documentation puts it plainly:
  "Defining a custom user model at the beginning of a project is highly recommended."
  https://docs.djangoproject.com/en/5.1/topics/auth/customizing/#using-a-custom-user-model-when-starting-a-project

WHAT OUR USER MODEL INHERITS FROM AbstractUser:
  AbstractUser already gives us a fully functional user model with:
    - username:     unique login identifier (e.g., 'jsmith')
    - email:        contact email address
    - password:     stored as a salted hash — NEVER stored as plain text
    - first_name:   optional display name
    - last_name:    optional display name
    - is_staff:     True grants access to the Django admin site
    - is_active:    False disables the account without deleting it (preferred
                    over deletion so FKs from parking sessions don't break)
    - is_superuser: True bypasses all permission checks in the admin
    - date_joined:  auto-set timestamp when the account was created
    - last_login:   auto-updated timestamp on each authentication

ACCESS CONTROL MODEL (from PLAN.md):
  - Single role system: is_staff=True grants full access to all pages.
  - No multi-role / permission groups needed at this stage.
  - Unauthenticated users are redirected to /login/ by @login_required and LoginRequiredMixin.
  - Guest plates (cars whose plates aren't in the LicensePlate table) are represented
    by ParkingSession.user=null, NOT by a special anonymous user account.
"""

from django.contrib.auth.models import AbstractUser


class User(AbstractUser):
    """
    Custom user model for the parking tracker.

    Inherits all fields and behavior from Django's AbstractUser (see module docstring).
    No extra fields are defined at this stage — the class exists solely to allow
    AUTH_USER_MODEL = 'accounts.User' in settings.py.

    This means:
      - All ForeignKey(settings.AUTH_USER_MODEL, ...) fields point here.
      - Future migrations that add fields to User don't require touching every
        table that references it.
      - Third-party packages that respect AUTH_USER_MODEL will use our User automatically.
    """

    class Meta:
        verbose_name = 'user'
        verbose_name_plural = 'users'
