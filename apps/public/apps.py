"""App config for the public-facing surface (gate kiosk + resident portal)."""

from django.apps import AppConfig


class PublicConfig(AppConfig):
    # WHY a dedicated app: the public surface (unauthenticated gate kiosk + resident
    # self-service accounts) is a different trust boundary from the staff dashboard.
    # Keeping it separate makes the "public vs staff" split explicit and auditable.
    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.public"
    label = "public"
