import uuid

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    """Persist scoped kiosk capabilities for atomic nonce consumption."""

    dependencies = [
        ("parking", "0009_lotsettings_safety_constraints"),
    ]

    operations = [
        migrations.CreateModel(
            name="KioskDeviceCapability",
            fields=[
                (
                    "id",
                    models.UUIDField(
                        default=uuid.uuid4,
                        editable=False,
                        primary_key=True,
                        serialize=False,
                    ),
                ),
                ("session_key", models.CharField(max_length=40, unique=True)),
                ("token_fingerprint", models.CharField(max_length=64)),
                (
                    "event_type",
                    models.CharField(
                        choices=[("entry", "Entry"), ("exit", "Exit")],
                        max_length=10,
                    ),
                ),
                ("nonce_hash", models.CharField(max_length=64)),
                ("expires_at", models.DateTimeField(db_index=True)),
                (
                    "lot",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="kiosk_capabilities",
                        to="parking.parkinglot",
                    ),
                ),
            ],
        ),
        migrations.CreateModel(
            name="KioskImageReplayDigest",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("digest", models.CharField(max_length=64)),
                ("seen_at", models.DateTimeField(db_index=True)),
                (
                    "capability",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="replay_digests",
                        to="parking.kioskdevicecapability",
                    ),
                ),
            ],
        ),
        migrations.AddConstraint(
            model_name="kioskimagereplaydigest",
            constraint=models.UniqueConstraint(
                fields=("capability", "digest"),
                name="unique_kiosk_image_digest",
            ),
        ),
    ]
