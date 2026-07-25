"""Public-facing routes: the gate kiosk and the resident self-service portal."""

from django.urls import path

from . import plates, registration, scan, wallet_views

# app_name enables namespaced reversals: {% url 'public:kiosk' %}
app_name = "public"

urlpatterns = [
    # ── Gate kiosk (unauthenticated) ────────────────────────────────────────
    path("", scan.KioskView.as_view(), name="kiosk"),
    path("kiosk/activate/", scan.activate_kiosk, name="kiosk_activate"),
    path("kiosk/scan/", scan.kiosk_scan, name="kiosk_scan"),
    # ── Post-login dispatch (staff → dashboard, resident → wallet) ──────────
    path("post-login/", registration.post_login, name="post_login"),
    # ── Resident account + plates ───────────────────────────────────────────
    path("register/", registration.signup, name="signup"),
    path("plates/", plates.plates, name="plates"),
    path("plates/<int:pk>/delete/", plates.plate_delete, name="plate_delete"),
    # ── Wallet ──────────────────────────────────────────────────────────────
    path("wallet/", wallet_views.wallet, name="wallet"),
    path("wallet/topup/", wallet_views.topup, name="topup"),
]
