# Generated manually so direct database writes preserve operator-setting invariants.

from decimal import Decimal

from django.db import migrations, models


def validate_existing_settings(apps, schema_editor):
    """Abort with record IDs before unsafe historical rows block constraints."""
    LotSettings = apps.get_model("parking", "LotSettings")
    invalid_ids = list(
        LotSettings.objects.filter(
            models.Q(rate__lte=Decimal("0.00"))
            | (
                models.Q(daily_cap_enabled=True)
                & (
                    models.Q(daily_cap_amount__isnull=True)
                    | models.Q(daily_cap_amount__lte=Decimal("0.00"))
                )
            )
            | (
                models.Q(daily_cap_enabled=False)
                & models.Q(daily_cap_amount__isnull=False)
            )
            | models.Q(confidence_threshold__lt=0.0)
            | models.Q(confidence_threshold__gt=1.0)
            | models.Q(image_retention_days__lt=1)
        ).values_list("id", flat=True)
    )
    if invalid_ids:
        raise RuntimeError(
            "Cannot add LotSettings safety constraints. Correct the settings "
            f"records with these IDs, then rerun the migration: {invalid_ids}"
        )


class Migration(migrations.Migration):
    """Add database backstops for settings normally validated by forms."""

    dependencies = [
        ("parking", "0008_canonical_unique_plate"),
    ]

    operations = [
        migrations.RunPython(
            validate_existing_settings,
            migrations.RunPython.noop,
        ),
        migrations.AddConstraint(
            model_name="lotsettings",
            constraint=models.CheckConstraint(
                condition=models.Q(rate__gt=Decimal("0.00")),
                name="lotsettings_rate_positive",
            ),
        ),
        migrations.AddConstraint(
            model_name="lotsettings",
            constraint=models.CheckConstraint(
                condition=(
                    models.Q(
                        daily_cap_enabled=False,
                        daily_cap_amount__isnull=True,
                    )
                    | models.Q(
                        daily_cap_enabled=True,
                        daily_cap_amount__isnull=False,
                        daily_cap_amount__gt=Decimal("0.00"),
                    )
                ),
                name="lotsettings_daily_cap_consistent",
            ),
        ),
        migrations.AddConstraint(
            model_name="lotsettings",
            constraint=models.CheckConstraint(
                condition=models.Q(confidence_threshold__gte=0.0)
                & models.Q(confidence_threshold__lte=1.0),
                name="lotsettings_confidence_range",
            ),
        ),
        migrations.AddConstraint(
            model_name="lotsettings",
            constraint=models.CheckConstraint(
                condition=models.Q(image_retention_days__isnull=True)
                | models.Q(image_retention_days__gte=1),
                name="lotsettings_retention_positive",
            ),
        ),
    ]
