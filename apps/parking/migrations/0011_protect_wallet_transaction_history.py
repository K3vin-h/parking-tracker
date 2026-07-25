# Generated manually to preserve existing wallet ledger rows during parent deletes.

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):
    """Protect immutable wallet transactions from cascading wallet deletion."""

    dependencies = [
        ("parking", "0010_kioskdevicecapability"),
    ]

    operations = [
        migrations.AlterField(
            model_name="wallettransaction",
            name="wallet",
            field=models.ForeignKey(
                help_text="The wallet this ledger entry belongs to.",
                on_delete=django.db.models.deletion.PROTECT,
                related_name="transactions",
                to="parking.wallet",
            ),
        ),
        migrations.AddConstraint(
            model_name="wallettransaction",
            constraint=models.UniqueConstraint(
                condition=models.Q(
                    ("kind", "topup"),
                    models.Q(("reference", ""), _negated=True),
                ),
                fields=("reference",),
                name="unique_wallet_topup_reference",
            ),
        ),
    ]
