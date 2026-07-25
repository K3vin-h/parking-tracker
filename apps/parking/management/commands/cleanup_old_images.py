"""
Management command: cleanup_old_images

Purges the image FILE and clears the image FIELD on PlateDetectionEvent rows
that are older than each lot's configured image_retention_days setting, while
KEEPING the event row itself so that plate text, confidence scores, and
timestamps survive as an audit record.  Lots with image_retention_days=None
are skipped (null means "keep forever"). Unresolved review-queue evidence has
a separate finite ceiling so an abandoned queue cannot retain images forever.

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

from django.conf import settings
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
        "Unresolved review images use UNRESOLVED_IMAGE_RETENTION_DAYS. "
        "Resolved images are skipped when image_retention_days=None. "
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

        # Every lot participates because unresolved evidence has a finite global
        # ceiling even when resolved images are configured to stay forever.
        settings_with_retention = LotSettings.objects.select_related("lot")

        if not settings_with_retention.exists():
            self.stdout.write(
                "No lots have settings configured. Nothing to do."
            )
            return

        for lot_settings in settings_with_retention:
            lot = lot_settings.lot
            now = timezone.now()
            unresolved_cutoff = now - timedelta(
                days=settings.UNRESOLVED_IMAGE_RETENTION_DAYS
            )
            unresolved = Q(manually_corrected=False) & (
                Q(is_low_confidence=True) | Q(session__isnull=True)
            )
            eligible = unresolved & Q(timestamp__lt=unresolved_cutoff)
            if lot_settings.image_retention_days is not None:
                normal_cutoff = now - timedelta(
                    days=lot_settings.image_retention_days
                )
                eligible |= ~unresolved & Q(timestamp__lt=normal_cutoff)

            # Resolved evidence follows the lot policy; unresolved evidence uses
            # the privacy ceiling. The direct lot FK includes unmatched events;
            # session__lot covers legacy rows where event.lot is NULL.
            # exclude(image='') keeps the command idempotent: once an image is
            # purged the row is no longer re-counted on later runs.
            old_events = (
                PlateDetectionEvent.objects.filter(
                    Q(lot=lot) | Q(lot__isnull=True, session__lot=lot),
                    eligible,
                )
                .exclude(image="")
            )

            count = old_events.count()
            if count == 0:
                if lot_settings.image_retention_days is None:
                    message = (
                        f'  "{lot.name}": no eligible images for configured retention.'
                    )
                else:
                    message = (
                        f'  "{lot.name}": no eligible images older than '
                        f"{lot_settings.image_retention_days} days."
                    )
                self.stdout.write(
                    message
                )
                continue

            if dry_run:
                self.stdout.write(
                    f'  "{lot.name}": would clear {count} image(s) '
                    "under the configured retention policies."
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
                    f'  "{lot.name}": cleared {cleared_count} image(s).'
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
