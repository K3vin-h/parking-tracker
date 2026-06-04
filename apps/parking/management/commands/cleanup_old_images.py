"""
Management command: cleanup_old_images

Deletes uploaded plate images that are older than each lot's configured
image_retention_days setting.  Lots with image_retention_days=None are skipped
(null means "keep forever").

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
from django.utils import timezone

from apps.parking.models import LotSettings, PlateDetectionEvent

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    """
    Delete plate images older than each lot's image_retention_days threshold.
    """

    help = (
        'Delete uploaded plate images older than each lot\'s image_retention_days setting. '
        'Lots with image_retention_days=None are skipped. '
        'Use --dry-run to preview what would be deleted without making changes.'
    )

    def add_arguments(self, parser):
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Log what would be deleted without actually deleting anything.',
        )

    def handle(self, *args, **options):
        dry_run = options['dry_run']
        total_deleted = 0

        if dry_run:
            self.stdout.write(self.style.WARNING('DRY RUN — no files or records will be deleted.\n'))

        # Only process lots that have a retention policy configured.
        # select_related('lot') avoids an extra query per row when accessing lot.name.
        settings_with_retention = (
            LotSettings.objects
            .filter(image_retention_days__isnull=False)
            .select_related('lot')
        )

        if not settings_with_retention.exists():
            self.stdout.write('No lots have image_retention_days configured. Nothing to do.')
            return

        for lot_settings in settings_with_retention:
            lot = lot_settings.lot
            cutoff = timezone.now() - timedelta(days=lot_settings.image_retention_days)

            # Events attached to sessions in this lot, older than the retention cutoff.
            # Events with session=None are excluded (no lot affiliation to match against).
            old_events = PlateDetectionEvent.objects.filter(
                session__lot=lot,
                timestamp__lt=cutoff,
            )

            count = old_events.count()
            if count == 0:
                self.stdout.write(f'  "{lot.name}": no events older than {lot_settings.image_retention_days} days.')
                continue

            if dry_run:
                self.stdout.write(
                    f'  "{lot.name}": would delete {count} event(s) '
                    f'(older than {cutoff.date()} — {lot_settings.image_retention_days} days).'
                )
                continue

            # Delete image files from storage first, then delete the DB records.
            # Two-pass approach: iterate events to delete files, then bulk-delete rows.
            # WHY not delete() on each event: bulk ORM delete is far faster for large sets.
            # WHY delete files before DB rows: if the process is interrupted mid-run,
            # orphaned DB rows are harmless (they can be cleaned next run); orphaned
            # files with no DB record are harder to detect.
            event_ids = []
            for event in old_events.iterator():
                if event.image:
                    event.image.delete(save=False)
                event_ids.append(event.pk)

            deleted_count, _ = PlateDetectionEvent.objects.filter(pk__in=event_ids).delete()
            total_deleted += deleted_count

            self.stdout.write(
                self.style.SUCCESS(
                    f'  "{lot.name}": deleted {deleted_count} event(s) '
                    f'(older than {cutoff.date()}).'
                )
            )
            logger.info('cleanup_old_images: deleted %d events for lot "%s"', deleted_count, lot.name)

        if not dry_run:
            self.stdout.write(self.style.SUCCESS(f'\nTotal deleted: {total_deleted} event(s).'))
        else:
            self.stdout.write(self.style.WARNING('\nDry run complete. No changes made.'))
