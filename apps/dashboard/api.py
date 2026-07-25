"""
Shared authorization helper for the staff dashboard API modules.

HISTORY: this module once held the staff plate-upload endpoint. In the China
kiosk redesign, uploading a plate is a *public* gate action (see apps.public), so
the upload view moved out and its reusable, security-critical core now lives in
apps.dashboard.scan_core.run_plate_scan(). What remains here is the one thing the
other dashboard API modules share — the staff-only access check.
"""

from django.contrib.auth.decorators import user_passes_test


def _is_staff(user) -> bool:
    """Centralize the operator role check used by every dashboard API route."""
    return user.is_authenticated and user.is_staff


# WHY user_passes_test instead of a manual 403: both anonymous and authenticated
# non-staff users follow the configured login flow, matching the page-route policy.
staff_required = user_passes_test(_is_staff)
