from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("parking", "0006_wallet_wallettransaction"),
    ]

    operations = [
        migrations.CreateModel(
            name="RequestRateLimit",
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
                ("scope", models.CharField(max_length=64)),
                ("identity_hash", models.CharField(max_length=64)),
                ("window_start", models.BigIntegerField()),
                ("count", models.PositiveIntegerField(default=1)),
                ("expires_at", models.DateTimeField(db_index=True)),
            ],
        ),
        migrations.AddConstraint(
            model_name="requestratelimit",
            constraint=models.UniqueConstraint(
                fields=("scope", "identity_hash", "window_start"),
                name="unique_rate_limit_bucket",
            ),
        ),
    ]
