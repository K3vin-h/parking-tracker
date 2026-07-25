from collections import defaultdict

from django.db import migrations, models


def canonicalize_existing_plates(apps, schema_editor):
    """
    Normalize historical plates but refuse to guess ownership when keys collide.

    Record IDs are safe operational identifiers for remediation and avoid putting
    usernames or emails into migration logs.
    """
    LicensePlate = apps.get_model("parking", "LicensePlate")
    db_alias = schema_editor.connection.alias
    canonical_rows = []
    collisions = defaultdict(list)

    for plate in LicensePlate.objects.using(db_alias).all().only("pk", "plate_text"):
        canonical = "".join((plate.plate_text or "").split()).upper()
        canonical_rows.append((plate.pk, canonical))
        collisions[canonical].append(plate.pk)

    duplicate_ids = [ids for ids in collisions.values() if len(ids) > 1]
    if duplicate_ids:
        raise RuntimeError(
            "Canonical license plate collisions must be resolved before migrating: "
            f"{duplicate_ids}"
        )

    for plate_id, canonical in canonical_rows:
        LicensePlate.objects.using(db_alias).filter(pk=plate_id).update(
            plate_text=canonical
        )


class Migration(migrations.Migration):
    dependencies = [
        ("parking", "0007_requestratelimit"),
    ]

    operations = [
        migrations.RunPython(
            canonicalize_existing_plates,
            reverse_code=migrations.RunPython.noop,
        ),
        migrations.RemoveConstraint(
            model_name="licenseplate",
            name="licenseplate_user_plate_unique",
        ),
        migrations.AddConstraint(
            model_name="licenseplate",
            constraint=models.UniqueConstraint(
                fields=("plate_text",),
                name="unique_registered_plate_text",
            ),
        ),
    ]
