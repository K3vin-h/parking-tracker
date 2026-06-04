"""
Root URL configuration for the parking tracker project.

HOW DJANGO ROUTING WORKS:
  When Django receives a request for a URL like '/admin/', it reads urlpatterns
  top to bottom and stops at the first pattern that matches. It then calls the
  associated view function (or passes control to an included URL module).

  Think of urlpatterns as a phone directory — Django looks up the URL and routes
  the request to the right "phone number" (view function).

URL STRUCTURE:
  /admin/       → Django's built-in admin interface
  /login/       → Login page (from django.contrib.auth.urls)
  /logout/      → Logout endpoint (from django.contrib.auth.urls)
  /             → Dashboard app (populated in Days 8–10)
  /api/         → Dashboard API endpoints (populated in Day 8)
"""

from django.contrib import admin
from django.db import connection
from django.http import JsonResponse
from django.urls import include, path
from django.conf import settings
from django.conf.urls.static import static


def health_check(request):
    """
    Liveness + readiness probe for Docker HEALTHCHECK and load-balancer checks.

    Verifies that Django is running AND the database connection is alive.
    Returns 200 {"status": "ok"} on success, 503 {"status": "error"} on failure.
    No authentication required — this endpoint is intentionally public so that
    Docker, K8s, and load balancers can poll it without credentials.
    """
    try:
        connection.ensure_connection()
        return JsonResponse({'status': 'ok'})
    except Exception:
        return JsonResponse({'status': 'error'}, status=503)


urlpatterns = [
    # ── Health ────────────────────────────────────────────────────────────────
    # Polled by Dockerfile HEALTHCHECK, docker-compose healthcheck, and future
    # Kubernetes liveness/readiness probes.  Must be first to keep latency low.
    path('health/', health_check),

    # ── Admin ──────────────────────────────────────────────────────────────────
    # Django's built-in admin site — browse and edit all registered models.
    # Only accessible to users with is_staff=True.
    # The URL '/admin/' is conventional; changing it adds minor security through obscurity.
    path('admin/', admin.site.urls),

    # ── Authentication ─────────────────────────────────────────────────────────
    # django.contrib.auth.urls provides a complete set of auth views:
    #   /login/                  → login page (renders registration/login.html)
    #   /logout/                 → logs the user out, redirects to LOGOUT_REDIRECT_URL
    #   /password_change/        → change password form
    #   /password_change/done/   → confirmation after password change
    #   /password_reset/         → request a password reset email
    #   /password_reset/done/    → confirmation that reset email was sent
    #   /reset/<uidb64>/<token>/ → password reset form (from the email link)
    #   /reset/done/             → confirmation after successful reset
    #
    # We only actively use login and logout for Day 1, but including the full set
    # now means password reset is available without any additional code.
    path('', include('django.contrib.auth.urls')),

    # ── Dashboard App ──────────────────────────────────────────────────────────
    # All parking management pages and API endpoints.
    # apps/dashboard/urls.py is currently empty (Day 1 placeholder).
    # Views are added in Days 8–10.
    path('', include('apps.dashboard.urls')),
]

# ── Development Media Serving ──────────────────────────────────────────────────
# In development (DEBUG=True), serve uploaded media files through Django's dev server.
#
# WHY only in DEBUG?
#   Django's static() helper adds URL patterns that map MEDIA_URL to MEDIA_ROOT.
#   This is convenient for development but should NEVER be used in production because:
#     1. Django is slow at serving files compared to nginx/Apache.
#     2. Django doesn't validate that what it's serving from MEDIA_ROOT is safe.
#   In production, configure nginx to serve the media directory directly.
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
