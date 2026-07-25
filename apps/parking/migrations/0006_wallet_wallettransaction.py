from decimal import Decimal

import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("parking", "0005_alter_lotsettings_grace_period_minutes_and_more"),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name="Wallet",
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
                (
                    "balance",
                    models.DecimalField(
                        decimal_places=2,
                        default=Decimal("0.00"),
                        help_text="Current prepaid balance in dollars. May be negative (amount owed).",
                        max_digits=10,
                    ),
                ),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                (
                    "user",
                    models.OneToOneField(
                        help_text="The account this prepaid balance belongs to.",
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="wallet",
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
            ],
            options={
                "verbose_name": "wallet",
                "verbose_name_plural": "wallets",
            },
        ),
        migrations.CreateModel(
            name="WalletTransaction",
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
                (
                    "amount",
                    models.DecimalField(
                        decimal_places=2,
                        help_text="Signed dollar amount: positive = credit, negative = debit.",
                        max_digits=10,
                    ),
                ),
                (
                    "kind",
                    models.CharField(
                        choices=[
                            ("topup", "Top-up"),
                            ("charge", "Parking charge"),
                            ("adjustment", "Adjustment"),
                        ],
                        help_text="What produced this ledger entry.",
                        max_length=20,
                    ),
                ),
                (
                    "description",
                    models.CharField(blank=True, default="", max_length=200),
                ),
                ("reference", models.CharField(blank=True, default="", max_length=100)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                (
                    "session",
                    models.ForeignKey(
                        blank=True,
                        help_text="The parking session this charge settled, if any.",
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        related_name="wallet_transactions",
                        to="parking.parkingsession",
                    ),
                ),
                (
                    "wallet",
                    models.ForeignKey(
                        help_text="The wallet this ledger entry belongs to.",
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="transactions",
                        to="parking.wallet",
                    ),
                ),
            ],
            options={
                "verbose_name": "wallet transaction",
                "verbose_name_plural": "wallet transactions",
                "ordering": ["-created_at", "-pk"],
            },
        ),
        migrations.AddIndex(
            model_name="wallettransaction",
            index=models.Index(
                fields=["wallet", "-created_at"], name="wallet_txn_wallet_time_idx"
            ),
        ),
    ]
