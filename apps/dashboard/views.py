"""
View functions and class-based views for the dashboard app.

Day 1: Placeholder only.

In Django, a "view" is a Python function or class that:
  1. Receives an HTTP request object
  2. Does some work (reads from the database, calls the CV pipeline, etc.)
  3. Returns an HTTP response (HTML page, JSON, redirect, etc.)

Views are the "controller" in the MVC pattern — they sit between the URL
routing (urls.py) and the data (models.py), deciding what to show and to whom.

All views will be implemented in Days 8–10:
  DashboardView  — main overview page
  UploadView     — image upload and detection result
  LogView        — session history with filters
  ErrorQueueView — low-confidence detection review
  RevenueView    — revenue analytics
  SettingsView   — lot configuration form

All views will use @login_required or LoginRequiredMixin — no public pages.
"""

# Views are added in Days 8–10.
