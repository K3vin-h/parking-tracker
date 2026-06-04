"""
Migration: add validators to LotSettings bounded fields.

Adds MinValueValidator / MaxValueValidator to three fields:
  - confidence_threshold: must be in [0.0, 1.0]
  - grace_period_minutes: must be >= 0
  - image_retention_days: must be >= 1 (null still allowed for "keep forever")

Django validators are enforced at the model/form layer, not at the database level,
so this migration records the validators in the migration state without altering
any column type or adding a database constraint.
"""

import django.core.validators
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('parking', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='lotsettings',
            name='confidence_threshold',
            field=models.FloatField(
                default=0.6,
                help_text='CV pipeline confidence threshold (0.0–1.0). Detections below this score are flagged as low-confidence and queued for review.',
                validators=[
                    django.core.validators.MinValueValidator(0.0),
                    django.core.validators.MaxValueValidator(1.0),
                ],
            ),
        ),
        migrations.AlterField(
            model_name='lotsettings',
            name='grace_period_minutes',
            field=models.IntegerField(
                default=15,
                help_text='Sessions shorter than this many minutes are free (charged $0.00).',
                validators=[django.core.validators.MinValueValidator(0)],
            ),
        ),
        migrations.AlterField(
            model_name='lotsettings',
            name='image_retention_days',
            field=models.IntegerField(
                blank=True,
                null=True,
                help_text='Delete uploaded images older than this many days. Null means keep forever.',
                validators=[django.core.validators.MinValueValidator(1)],
            ),
        ),
    ]
