"""Staff-only operator pages and their HTMX/JSON support endpoints."""

from django.urls import path

from . import api, views

# app_name enables namespaced URL reversals: {% url 'dashboard:api_upload' %}
app_name = "dashboard"

urlpatterns = [
    path("", views.DashboardView.as_view(), name="dashboard"),
    path("upload/", views.UploadView.as_view(), name="upload"),
    path("log/", views.LogView.as_view(), name="log"),
    path("errors/", views.ErrorQueueView.as_view(), name="errors"),
    path("revenue/", views.RevenueView.as_view(), name="revenue"),
    path("settings/", views.SettingsView.as_view(), name="settings"),
    path("api/upload/", api.upload, name="api_upload"),
    path("api/sessions/", api.sessions, name="api_sessions"),
    path("api/dashboard-stats/", api.dashboard_stats, name="api_dashboard_stats"),
    path(
        "api/events/<int:event_id>/correct/",
        api.correct_event,
        name="api_correct_event",
    ),
    path("api/revenue-data/", api.revenue_data, name="api_revenue_data"),
    path(
        "api/events/<int:event_id>/image/",
        api.event_image,
        name="api_event_image",
    ),
]
