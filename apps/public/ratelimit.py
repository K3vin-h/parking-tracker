"""
Shared fixed-window throttling for public and authentication endpoints.

Counters live in PostgreSQL so every Gunicorn worker observes the same limit.
Client addresses are accepted from forwarding headers only when the direct peer is
an explicitly trusted proxy, and only SHA-256 digests are persisted.
"""

import hashlib
import logging
import time
from datetime import timedelta
from functools import wraps

from django.conf import settings
from django.db import DatabaseError, transaction
from django.db.models import F
from django.http import HttpResponse
from django.utils import timezone

from apps.parking.models import RequestRateLimit

logger = logging.getLogger("apps.public")

_COUNTED_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


def _client_ip(request) -> str:
    """
    Resolve a client address without accepting spoofed forwarding headers.

    A trusted reverse proxy is responsible for replacing X-Forwarded-For. For all
    other direct peers, REMOTE_ADDR remains authoritative.
    """
    direct_peer = request.META.get("REMOTE_ADDR", "unknown")
    if direct_peer not in settings.TRUSTED_PROXY_IPS:
        return direct_peer

    forwarded = request.META.get("HTTP_X_FORWARDED_FOR", "")
    addresses = [value.strip() for value in forwarded.split(",") if value.strip()]
    return addresses[-1] if addresses else direct_peer


def _identity_hash(identity: str) -> str:
    """Persist a stable limiter key without retaining a raw client identifier."""
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _purge_expired_buckets(now, batch_size: int = 500) -> None:
    """Bound table growth by reclaiming a small global expiry batch per mutation."""
    expired_ids = list(
        RequestRateLimit.objects.filter(expires_at__lt=now)
        .order_by("expires_at")
        .values_list("pk", flat=True)[:batch_size]
    )
    if expired_ids:
        RequestRateLimit.objects.filter(pk__in=expired_ids).delete()


def _increment_counter(
    *, scope: str, identity_hash: str, window_start: int, window_seconds: int
) -> int:
    """Atomically create or increment one shared fixed-window bucket."""
    now = timezone.now()
    expires_at = now + timedelta(seconds=window_seconds)

    _purge_expired_buckets(now)

    with transaction.atomic():
        bucket, created = RequestRateLimit.objects.select_for_update().get_or_create(
            scope=scope,
            identity_hash=identity_hash,
            window_start=window_start,
            defaults={"count": 1, "expires_at": expires_at},
        )
        if created:
            return 1
        RequestRateLimit.objects.filter(pk=bucket.pk).update(count=F("count") + 1)
        bucket.refresh_from_db(fields=["count"])
        return bucket.count


def rate_limit(scope: str, limit: int, window_seconds: int):
    """Allow a bounded number of state-changing requests per shared identity."""
    if limit <= 0 or window_seconds <= 0:
        raise ValueError("rate-limit values must be positive")

    def decorator(view):
        @wraps(view)
        def wrapped(request, *args, **kwargs):
            if request.method not in _COUNTED_METHODS:
                return view(request, *args, **kwargs)

            window_start = int(time.time() // window_seconds)
            identity_hash = _identity_hash(_client_ip(request))
            try:
                count = _increment_counter(
                    scope=scope,
                    identity_hash=identity_hash,
                    window_start=window_start,
                    window_seconds=window_seconds,
                )
            except DatabaseError:
                # Public mutation and credential endpoints fail closed if their abuse
                # control is unavailable; proceeding would silently remove protection.
                logger.exception("Rate-limit database error on scope %s", scope)
                return HttpResponse(
                    "Service temporarily unavailable. Please try again.",
                    status=503,
                )

            if count > limit:
                logger.warning(
                    "Rate limit hit: scope=%s identity=%s",
                    scope,
                    identity_hash[:12],
                )
                return HttpResponse(
                    "Too many requests. Please wait a moment and try again.",
                    status=429,
                )
            return view(request, *args, **kwargs)

        return wrapped

    return decorator
