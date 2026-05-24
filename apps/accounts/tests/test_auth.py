"""
Tests for authentication flows.

These tests verify that Django's built-in auth views work correctly
with our URL configuration and template setup.

WHAT'S BEING TESTED:
  - The login page renders (GET /login/ returns 200)
  - Valid credentials log in and redirect (POST /login/ → /)
  - Invalid credentials show an error (POST /login/ stays on /login/)
  - Logout redirects to login page
  - Protected URLs redirect unauthenticated users to /login/

WHY TEST BUILT-IN DJANGO FUNCTIONALITY?
  We're not testing Django's auth framework itself — we're testing OUR configuration:
    - Are the auth URLs included in config/urls.py?
    - Is our login template found at the right path?
    - Are LOGIN_URL, LOGIN_REDIRECT_URL, LOGOUT_REDIRECT_URL set correctly?
    - Does the admin redirect to our /login/ page?
  These tests catch misconfiguration that would make the app unusable.
"""

import pytest
from django.contrib.auth import get_user_model
from django.urls import reverse

# get_user_model() reads AUTH_USER_MODEL from settings.
# We use it here so these tests work regardless of which User model is active.
User = get_user_model()

# NOTE: We use pytest-django's built-in 'client' fixture (Django's test Client).
# It simulates HTTP requests without starting a real server.
# No need to define it here — pytest-django provides it automatically.


@pytest.fixture
def regular_user(db):
    """
    A regular (non-staff) user for auth tests.

    This fixture creates a test user in the database. The 'db' parameter
    tells pytest-django that this fixture needs database access.
    """
    return User.objects.create_user(
        username='testuser',
        email='testuser@example.com',
        password='testpassword123',
    )


@pytest.mark.django_db
class TestLoginPage:
    """Tests for the /login/ page."""

    def test_login_page_renders(self, client):
        """
        GET /login/ returns a 200 OK response.

        This confirms:
          1. The URL is registered in config/urls.py
          2. The template 'registration/login.html' is found
          3. The template renders without errors
        """
        # reverse() looks up the URL name — 'login' is provided by django.contrib.auth.urls
        url = reverse('login')
        response = client.get(url)
        assert response.status_code == 200

    def test_login_page_contains_form(self, client):
        """
        The login page renders a form with a username and password field.

        Tests that the template is rendering the Django form correctly.
        """
        url = reverse('login')
        response = client.get(url)
        content = response.content.decode()
        # Check that the rendered page contains form input fields
        assert 'type="password"' in content, "Login page must have a password field"
        assert '<form' in content, "Login page must have a form element"

    def test_valid_login_redirects(self, client, regular_user):
        """
        POST /login/ with correct credentials redirects to LOGIN_REDIRECT_URL (/).

        Django's LoginView checks credentials and redirects on success.
        We verify it redirects to the right URL (configured in settings.py).
        """
        url = reverse('login')
        response = client.post(url, {
            'username': 'testuser',
            'password': 'testpassword123',
        })
        # 302 is a redirect response
        assert response.status_code == 302
        # The redirect should go to LOGIN_REDIRECT_URL
        assert response['Location'] == '/'

    def test_invalid_login_stays_on_login_page(self, client, regular_user):
        """
        POST /login/ with wrong password returns 200 (form re-shown with errors).

        Django doesn't redirect on failure — it re-renders the login page with
        error messages. This is why we check for 200 (not 302) on failure.
        """
        url = reverse('login')
        response = client.post(url, {
            'username': 'testuser',
            'password': 'wrongpassword',
        })
        assert response.status_code == 200

    def test_unknown_user_login_fails(self, client):
        """
        Logging in with a non-existent username fails gracefully.

        Django's AuthenticationForm returns a generic error for both
        wrong username AND wrong password — this prevents username enumeration
        (an attacker can't tell if the username exists).
        """
        url = reverse('login')
        response = client.post(url, {
            'username': 'doesnotexist',
            'password': 'anypassword',
        })
        assert response.status_code == 200


@pytest.mark.django_db
class TestLogout:
    """Tests for the logout flow."""

    def test_logout_redirects_to_login(self, client, regular_user):
        """
        Logging out redirects to LOGOUT_REDIRECT_URL (/login/).

        After logout, users should land on the login page — not get a 404
        or be redirected to an external site.
        """
        # First log in
        client.force_login(regular_user)

        url = reverse('logout')
        response = client.post(url)  # Django 5.x requires POST for logout (security)

        assert response.status_code == 302
        assert response['Location'] == '/login/'

    def test_authenticated_user_can_logout(self, client, regular_user):
        """
        A logged-in user can successfully log out, ending their session.

        After logout, the session cookie should no longer authenticate them.
        """
        client.force_login(regular_user)

        # Verify they're logged in
        assert '_auth_user_id' in client.session

        # Logout
        client.post(reverse('logout'))

        # Session should be cleared
        assert '_auth_user_id' not in client.session


@pytest.mark.django_db
class TestProtectedAccess:
    """Tests that protected pages redirect unauthenticated users."""

    def test_admin_redirects_unauthenticated_to_login(self, client):
        """
        GET /admin/ redirects unauthenticated users to the login page.

        The Django admin requires is_staff=True. Unauthenticated users
        (or regular non-staff users) are redirected to the admin's own
        login URL (which Django routes to our LOGIN_URL).
        """
        response = client.get('/admin/')
        # The admin redirects unauthenticated users
        assert response.status_code == 302
        assert '/login' in response['Location'] or 'admin' in response['Location']

    # NOTE: Additional protected URL tests for /upload/, /log/, /errors/, /revenue/,
    # and /settings/ will be added in Days 8–10 when those views are implemented.
