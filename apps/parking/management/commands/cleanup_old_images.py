"""
Management command: cleanup_old_images

Purges the image FILE and clears the image FIELD on PlateDetectionEvent rows
that are older than each lot's configured image_retention_days setting, while
KEEPING the event row itself so that plate text, confidence scores, and
timestamps survive as an audit record.  Lots with image_retention_days=None
are skipped (null means "keep forever").

USAGE:
  docker-compose exec web python manage.py cleanup_old_images
  docker-compose exec web python manage.py cleanup_old_images --dry-run

WHY a management command?
  Image retention is an operational concern — it clears disk space on a
  schedule set per-lot by the operator.  A management command is idempotent,
  can be tested standalone, and is easy to wire to a cron job (Day 11).
"""

import logging
from datetime import timedelta

from django.core.management.base import BaseCommand
from django.db.models import Q
from django.utils import timezone

from apps.parking.models import LotSettings, PlateDetectionEvent

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    """
    Delete plate images older than each lot's image_retention_days threshold.
    """

    help = (
        "Purge uploaded plate images older than each lot's image_retention_days setting "
        "(deletes the image file and clears the image field but keeps the event record). "
        "Lots with image_retention_days=None are skipped. "
        "Use --dry-run to preview what would be cleared without making changes."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Log what would be deleted without actually deleting anything.",
        )

    def handle(self, *args, **options):
        dry_run = options["dry_run"]
        total_cleared = 0

        if dry_run:
            self.stdout.write(
                self.style.WARNING("DRY RUN — no files or records will be changed.\n")
            )

        # Only process lots that have a retention policy configured.
        # select_related('lot') avoids an extra query per row when accessing lot.name.
        settings_with_retention = LotSettings.objects.filter(
            image_retention_days__isnull=False
        ).select_related("lot")

        if not settings_with_retention.exists():
            self.stdout.write(
                "No lots have image_retention_days configured. Nothing to do."
            )
            return

        for lot_settings in settings_with_retention:
            lot = lot_settings.lot
            cutoff = timezone.now() - timedelta(days=lot_settings.image_retention_days)

            # Events in this lot, older than the cutoff, that still hold an image.
            # exclude(image='') keeps the command idempotent: once an image is
            # purged the row is no longer re-counted on later runs. The direct lot
            # FK includes unmatched exit review events with session=None;
            # session__lot is a fallback for older rows where event.lot is NULL.
            old_events = PlateDetectionEvent.objects.filter(
                Q(lot=lot) | Q(lot__isnull=True, session__lot=lot),
                timestamp__lt=cutoff,
            ).exclude(image="")

            count = old_events.count()
            if count == 0:
                self.stdout.write(
                    f'  "{lot.name}": no images older than {lot_settings.image_retention_days} days.'
                )
                continue

            if dry_run:
                self.stdout.write(
                    f'  "{lot.name}": would clear {count} image(s) '
                    f"(older than {cutoff.date()} — {lot_settings.image_retention_days} days)."
                )
                continue

            # Purge the image FILE from storage, then clear the image FIELD while
            # KEEPING the PlateDetectionEvent row (plate text, confidence, and
            # timestamps remain as an audit record). Two passes: delete files while
            # iterating, then one bulk UPDATE to empty the field.
            # WHY files first: an interrupted run leaves rows whose file is gone but
            # field still set — harmless and re-cleared next run; the reverse would
            # orphan files with no DB reference, which are far harder to find.
            # Purge each image file, collecting only the rows whose file was removed
            # successfully so the bulk field-clear (and the reported count) match what
            # actually happened.
            event_ids = []
            for event in old_events.iterator():
                # exclude(image='') guarantees a non-empty field, but guard defensively.
                if not event.image:
                    continue
                try:
                    event.image.delete(save=False)
                except Exception as exc:
                    # A single unreadable/remote-backend file must not abort the whole
                    # run. Log it and leave the field set so exclude(image='') re-selects
                    # the row next run (retry) — never orphan the file or skip later lots.
                    logger.warning(
                        "cleanup_old_images: could not delete image for event %d (%s); "
                        "leaving field set to retry next run",
                        event.pk,
                        type(exc).__name__,
                    )
                    continue
                event_ids.append(event.pk)

            cleared_count = PlateDetectionEvent.objects.filter(pk__in=event_ids).update(
                image=""
            )
            total_cleared += cleared_count

            self.stdout.write(
                self.style.SUCCESS(
                    f'  "{lot.name}": cleared {cleared_count} image(s) '
                    f"(older than {cutoff.date()})."
                )
            )
            logger.info(
                'cleanup_old_images: cleared %d images for lot "%s"',
                cleared_count,
                lot.name,
            )

        if not dry_run:
            self.stdout.write(
                self.style.SUCCESS(f"\nTotal cleared: {total_cleared} image(s).")
            )
        else:
            self.stdout.write(
                self.style.WARNING("\nDry run complete. No changes made.")
            )
