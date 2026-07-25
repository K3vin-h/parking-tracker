"""Canonicalize populated emails and make their identity case-insensitively unique."""

from django.db import migrations, models
from django.db.models.functions import Lower


def canonicalize_emails(apps, schema_editor):
    """Abort on collisions before lowercasing so migration never picks an owner."""
    User = apps.get_model("accounts", "User")
    grouped_ids = {}
    empty_ids = []
    for user_id, email in User.objects.exclude(email="").values_list("id", "email"):
        canonical = email.strip().lower()
        if not canonical:
            empty_ids.append(user_id)
            continue
        grouped_ids.setdefault(canonical, []).append(user_id)

    collisions = {
        email: ids for email, ids in grouped_ids.items() if len(ids) > 1
    }
    if collisions:
        conflicting_ids = sorted(
            user_id for ids in collisions.values() for user_id in ids
        )
        raise RuntimeError(
            "Cannot enforce case-insensitive email uniqueness; resolve "
            f"the accounts with these IDs first: {conflicting_ids}"
        )

    User.objects.filter(pk__in=empty_ids).update(email="")
    for canonical, user_ids in grouped_ids.items():
        User.objects.filter(pk=user_ids[0]).update(email=canonical)


class Migration(migrations.Migration):
    """Backfill canonical email values before adding the functional constraint."""

    dependencies = [
        ("accounts", "0001_initial"),
    ]

    operations = [
        migrations.RunPython(canonicalize_emails, migrations.RunPython.noop),
        migrations.AddConstraint(
            model_name="user",
            constraint=models.UniqueConstraint(
                Lower("email"),
                condition=~models.Q(email=""),
                name="unique_user_email_ci",
            ),
        ),
    ]
