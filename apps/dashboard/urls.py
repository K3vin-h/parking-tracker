"""
URL patterns for the dashboard app.

Day 1: Empty placeholder — allows config/urls.py to include this file
without raising an ImportError or NoReverseMatch.

All URL patterns are added in Days 8–10 when views are implemented:
  path('',           views.DashboardView.as_view(), name='dashboard'),
  path('upload/',    views.UploadView.as_view(),    name='upload'),
  path('log/',       views.LogView.as_view(),        name='log'),
  path('errors/',    views.ErrorQueueView.as_view(), name='errors'),
  path('revenue/',   views.RevenueView.as_view(),    name='revenue'),
  path('settings/',  views.SettingsView.as_view(),   name='settings'),
  path('api/upload/',           api.upload,           name='api_upload'),
  path('api/sessions/',         api.sessions,         name='api_sessions'),
  path('api/dashboard-stats/',  api.dashboard_stats,  name='api_dashboard_stats'),
  path('api/events/<int:pk>/correct/', api.correct_event, name='api_correct_event'),
  path('api/revenue-data/',     api.revenue_data,     name='api_revenue_data'),
"""

from django.urls import path

# app_name enables namespaced URL reversals: {% url 'dashboard:upload' %}
app_name = 'dashboard'

# Populated in Days 8–10 when views are implemented.
urlpatterns = []
