"""
App configuration for the 'dashboard' Django app.

The dashboard app owns the STAFF-ONLY web layer, nested under /staff/ (the
`dashboard:` URL namespace is unchanged for historical reasons):
  - apps/dashboard/views.py        — view functions and class-based views
  - apps/dashboard/api.py          — shared staff_required decorator
  - apps/dashboard/partials_api.py — HTMX endpoints (sessions, stats, correction)
  - apps/dashboard/revenue_api.py  — Chart.js revenue data endpoint
  - apps/dashboard/image_api.py    — private detection-image streaming endpoint
  - apps/dashboard/scan_core.py    — run_plate_scan(): the reusable image → CV →
    session core, now called by the public kiosk (see below)
  - apps/dashboard/urls.py         — URL routing for all /staff/ pages

WHAT /staff/ PROVIDES (is_staff required):
  /staff/               → Main overview: active sessions, today's revenue, recent events
  /staff/log/           → Full session history with filters and pagination
  /staff/errors/        → Low-confidence detection queue with manual correction
  /staff/revenue/       → Revenue analytics with Chart.js charts
  /staff/settings/      → Lot configuration (billing rate, grace period, etc.)
  /staff/api/sessions/  → GET list of sessions (used by HTMX log page)
  /staff/api/dashboard-stats/ → GET active count + revenue (polled every 10s)
  /staff/api/events/<id>/correct/ → PATCH manual plate correction
  /staff/api/revenue-data/ → Chart.js revenue, duration, lot, and hourly data
  /staff/api/events/<id>/image/ → GET private detection image (staff only)

WHERE UPLOAD HAPPENS NOW:
  The staff manual-upload page/API was removed. Uploading a plate image is now
  the public gate kiosk's job (apps.public: `GET /`, `POST /kiosk/scan/`), which
  reuses this app's apps/dashboard/scan_core.py::run_plate_scan for the actual
  image → CV → session logic.

WHY no models in the dashboard app?
  The dashboard is a pure presentation layer. All data lives in apps/parking
  and apps/cv. The dashboard reads from parking models and calls the cv pipeline.
  This keeps the data layer and presentation layer cleanly separated.
"""

from django.apps import AppConfig


class DashboardConfig(AppConfig):
    """Configuration class for the dashboard application."""

    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.dashboard"
    verbose_name = "Dashboard"
