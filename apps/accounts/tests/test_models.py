"""
Tests for the accounts app models.

Tests follow the AAA pattern (Arrange-Act-Assert):
  1. Arrange — set up the data needed for the test
  2. Act     — call the function or method being tested
  3. Assert  — verify the result is what we expected

WHY WRITE TESTS FOR A MODEL THAT JUST PASSES?
  Testing the custom User model verifies:
    1. AUTH_USER_MODEL is correctly pointing to accounts.User
    2. The migration ran and the accounts_user table exists
    3. Standard AbstractUser operations work as expected
    4. Plate relationships (via ForeignKey) resolve correctly
  These tests catch misconfiguration (wrong AUTH_USER_MODEL, broken migrations)
  immediately, rather than discovering the problem at runtime.

  The tests also serve as living documentation — they show HOW to create users
  in tests (using create_user vs create_superuser) for all future test files.
"""

import pytest
from django.contrib.auth import get_user_model

# get_user_model() is the correct way to get the active User model in tests.
# It reads AUTH_USER_MODEL from settings and returns the right class.
# NEVER import User directly in tests — it bypasses AUTH_USER_MODEL.
User = get_user_model()


@pytest.mark.django_db
class TestUserModel:
    """Tests for the custom User model."""

    def test_create_regular_user(self):
        """
        A regular user can be created with username, email, and password.

        create_user() hashes the password and sets is_staff=False, is_superuser=False.
        """
        # Arrange
        username = 'testuser'
        email = 'test@example.com'
        password = 'secure_password_123'

        # Act
        user = User.objects.create_user(
            username=username,
            email=email,
            password=password,
        )

        # Assert — user was saved to the database
        assert user.pk is not None, "User should have a primary key after creation"
        assert user.username == username
        assert user.email == email

    def test_password_is_hashed(self):
        """
        Passwords must NEVER be stored in plain text.

        Django's create_user() automatically hashes passwords using PBKDF2.
        The check_password() method verifies a plain-text password against the hash.
        If we stored passwords in plain text, a database breach would expose all credentials.
        """
        password = 'plain_text_password'
        user = User.objects.create_user(username='hashtest', email='hash@test.com', password=password)

        # The stored password should NOT be the plain-text value
        assert user.password != password, "Password should be hashed, not stored in plain text"

        # But check_password() should verify the original password correctly
        assert user.check_password(password) is True, "check_password() should verify the correct password"

    def test_regular_user_is_not_staff(self):
        """
        Regular users cannot access the admin site.

        is_staff=False by default — create_user() sets this.
        Only create_superuser() or explicit is_staff=True creates admin accounts.
        """
        user = User.objects.create_user(username='notstaff', email='notstaff@test.com', password='pass123')
        assert user.is_staff is False

    def test_superuser_is_staff(self):
        """
        Superusers have is_staff=True and is_superuser=True.

        create_superuser() is used by manage.py createsuperuser and our setup_defaults command.
        Superusers bypass all permission checks in the admin.
        """
        superuser = User.objects.create_superuser(
            username='admin',
            email='admin@test.com',
            password='admin_password',
        )
        assert superuser.is_staff is True
        assert superuser.is_superuser is True

    def test_user_is_active_by_default(self):
        """
        New users are active by default (is_active=True).

        Deactivating a user (is_active=False) disables their account without deleting it.
        This preserves all FK references (like ParkingSession.user) for historical records.
        """
        user = User.objects.create_user(username='activeuser', email='active@test.com', password='pass123')
        assert user.is_active is True

    def test_auth_user_model_is_custom_user(self):
        """
        The active User model (from AUTH_USER_MODEL) should be our custom User class.

        This test protects against misconfiguration in settings.py.
        If AUTH_USER_MODEL pointed to the wrong class, get_user_model() would return
        the wrong class and all user-related operations would fail in production.
        """
        from apps.accounts.models import User as AccountsUser
        assert User is AccountsUser, (
            "AUTH_USER_MODEL must point to apps.accounts.User. "
            "Check AUTH_USER_MODEL in config/settings.py."
        )

    def test_user_string_representation(self):
        """
        The string representation of a User comes from AbstractUser.

        AbstractUser's __str__ returns the username. This is what shows up
        in the Django admin, ForeignKey dropdowns, and debug output.
        """
        user = User.objects.create_user(username='plateowner', email='owner@test.com', password='pass')
        assert str(user) == 'plateowner'
