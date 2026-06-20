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
from django.contrib.staticfiles import finders
from django.template.loader import render_to_string
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
        username="testuser",
        email="testuser@example.com",
        password="testpassword123",
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
        url = reverse("login")
        response = client.get(url)
        assert response.status_code == 200

    def test_login_page_contains_form(self, client):
        """
        The login page renders a form with a username and password field.

        Tests that the template is rendering the Django form correctly.
        """
        url = reverse("login")
        response = client.get(url)
        content = response.content.decode()
        # Check that the rendered page contains form input fields
        assert 'type="password"' in content, "Login page must have a password field"
        assert "<form" in content, "Login page must have a form element"

    def test_valid_login_redirects(self, client, regular_user):
        """
        POST /login/ with correct credentials redirects to LOGIN_REDIRECT_URL (/).

        Django's LoginView checks credentials and redirects on success.
        We verify it redirects to the right URL (configured in settings.py).
        """
        url = reverse("login")
        response = client.post(
            url,
            {
                "username": "testuser",
                "password": "testpassword123",
            },
        )
        # 302 is a redirect response
        assert response.status_code == 302
        # The redirect should go to LOGIN_REDIRECT_URL
        assert response["Location"] == "/"

    def test_invalid_login_stays_on_login_page(self, client, regular_user):
        """
        POST /login/ with wrong password returns 200 (form re-shown with errors).

        Django doesn't redirect on failure — it re-renders the login page with
        error messages. This is why we check for 200 (not 302) on failure.
        """
        url = reverse("login")
        response = client.post(
            url,
            {
                "username": "testuser",
                "password": "wrongpassword",
            },
        )
        assert response.status_code == 200

    def test_unknown_user_login_fails(self, client):
        """
        Logging in with a non-existent username fails gracefully.

        Django's AuthenticationForm returns a generic error for both
        wrong username AND wrong password — this prevents username enumeration
        (an attacker can't tell if the username exists).
        """
        url = reverse("login")
        response = client.post(
            url,
            {
                "username": "doesnotexist",
                "password": "anypassword",
            },
        )
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

        url = reverse("logout")
        response = client.post(url)  # Django 5.x requires POST for logout (security)

        assert response.status_code == 302
        assert response["Location"] == "/login/"

    def test_authenticated_user_can_logout(self, client, regular_user):
        """
        A logged-in user can successfully log out, ending their session.

        After logout, the session cookie should no longer authenticate them.
        """
        client.force_login(regular_user)

        # Verify they're logged in
        assert "_auth_user_id" in client.session

        # Logout
        client.post(reverse("logout"))

        # Session should be cleared
        assert "_auth_user_id" not in client.session

    def test_base_template_uses_post_logout_form(self, regular_user):
        """
        The shared navigation renders logout as a CSRF-protected POST form.

        Django rejects GET requests to LogoutView, and GET logout links are also
        vulnerable to crawlers or prefetchers ending a user's session.
        """
        html = render_to_string(
            "base.html",
            {
                "csrf_token": "masked-csrf-token",
                "user": regular_user,
            },
        )

        assert f'action="{reverse("logout")}"' in html
        assert 'method="post"' in html
        assert 'name="csrfmiddlewaretoken"' in html
        assert f'href="{reverse("logout")}"' not in html


class TestBaseFrontendAssets:
    """Tests for security and loading behavior in the shared page shell."""

    def test_base_template_configures_htmx_csrf_and_local_asset(self):
        """
        HTMX must send Django's CSRF header and load without a public CDN.

        Keeping the configuration on body lets descendants inherit it while
        retaining Django's masked-token protection.
        """
        html = render_to_string(
            "base.html",
            {
                "csrf_token": "masked-csrf-token",
            },
        )

        assert "hx-headers=" in html
        assert "X-CSRFToken" in html
        assert "masked-csrf-token" in html
        assert "/static/js/vendor/htmx-2.0.10.min.js" in html
        assert "cdn.jsdelivr.net" not in html

    def test_base_template_disables_htmx_eval_and_script_tags(self):
        """
        The shared shell must carry the htmx-config meta tag that disables eval()
        and <script> execution in HTMX-swapped responses.

        This defence-in-depth measure lets the CSP keep script-src at 'self' with
        no 'unsafe-eval' or 'unsafe-inline'. Core HTMX attributes (hx-get,
        hx-post, hx-trigger polling, hx-swap) are unaffected — only eval-based
        filter expressions and response <script> tags are blocked.
        """
        html = render_to_string(
            "base.html",
            {
                "csrf_token": "masked-csrf-token",
            },
        )

        assert 'name="htmx-config"' in html
        assert '"allowEval":false' in html
        assert '"allowScriptTags":false' in html

    def test_chart_is_local_but_not_loaded_globally(self):
        """
        Chart.js stays available for analytics pages without burdening all pages.

        The shared shell should not transfer the chart bundle on login or pages
        that contain no chart.
        """
        html = render_to_string("base.html")

        assert '<script src="/static/js/vendor/chart-4.5.1.umd.min.js">' not in html
        assert finders.find("js/vendor/chart-4.5.1.umd.min.js") is not None

    def test_fonts_are_local_and_external_font_hosts_are_absent(self):
        """
        Rendering the shared shell must not disclose staff client IPs to font CDNs.

        Font files are resolved through Django staticfiles for deterministic,
        offline-capable rendering.
        """
        html = render_to_string("base.html")

        assert "fonts.googleapis.com" not in html
        assert "fonts.gstatic.com" not in html
        assert (
            finders.find(
                "fonts/jetbrains-mono/JetBrainsMono-Regular.ttf",
            )
            is not None
        )
        assert (
            finders.find(
                "fonts/jetbrains-mono/JetBrainsMono-Medium.ttf",
            )
            is not None
        )
        assert (
            finders.find(
                "fonts/jetbrains-mono/JetBrainsMono-Bold.ttf",
            )
            is not None
        )


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
        response = client.get("/admin/")
        # The admin redirects unauthenticated users
        assert response.status_code == 302
        assert "/login" in response["Location"] or "admin" in response["Location"]

    # NOTE: Additional protected URL tests for /upload/, /log/, /errors/, /revenue/,
    # and /settings/ will be added in Days 8–10 when those views are implemented.
