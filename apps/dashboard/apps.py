"""
App configuration for the 'dashboard' Django app.

The dashboard app owns the web layer — all the pages users see and interact with:
  - apps/dashboard/views.py   — view functions and class-based views (Day 8+)
  - apps/dashboard/api.py     — JSON API endpoints for HTMX and Chart.js (Day 8+)
  - apps/dashboard/urls.py    — URL routing for all dashboard pages

WHAT THE DASHBOARD PROVIDES (Days 8–10):
  /               → Main overview: active sessions, today's revenue, recent events
  /upload/        → Image upload form + detection result display
  /log/           → Full session history with filters and pagination
  /errors/        → Low-confidence detection queue with manual correction
  /revenue/       → Revenue analytics with Chart.js charts
  /settings/      → Lot configuration (billing rate, grace period, etc.)
  /api/upload/    → POST endpoint accepting image + event_type
  /api/sessions/  → GET list of sessions (used by HTMX log page)
  /api/dashboard-stats/ → GET active count + revenue (polled every 10s)
  /api/events/<id>/correct/ → PATCH manual plate correction

WHY no models in the dashboard app?
  The dashboard is a pure presentation layer. All data lives in apps/parking
  and apps/cv. The dashboard reads from parking models and calls the cv pipeline.
  This keeps the data layer and presentation layer cleanly separated.

Day 1: App registered with an empty URL file — placeholder only.
"""

from django.apps import AppConfig


class DashboardConfig(AppConfig):
    """Configuration class for the dashboard application."""

    default_auto_field = 'django.db.models.BigAutoField'
    name = 'apps.dashboard'
    verbose_name = 'Dashboard'
