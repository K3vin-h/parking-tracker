"""Behavioral tests for the shared public-endpoint rate limiter."""

from datetime import timedelta

import pytest
from django.core.cache import cache
from django.http import HttpResponse
from django.test import RequestFactory
from django.urls import reverse
from django.utils import timezone

from apps.parking.models import RequestRateLimit
from apps.public.ratelimit import rate_limit


@pytest.fixture(autouse=True)
def clear_legacy_cache():
    """Keep the pre-remediation cache limiter from leaking between red tests."""
    cache.clear()
    yield
    cache.clear()


@pytest.mark.django_db
def test_counter_survives_process_local_cache_clear():
    """Clearing one worker's cache must not reset the shared request counter."""

    @rate_limit(scope="shared-test", limit=1, window_seconds=60)
    def endpoint(request):
        return HttpResponse("ok")

    factory = RequestFactory()
    first = endpoint(factory.post("/", REMOTE_ADDR="192.0.2.10"))
    cache.clear()
    second = endpoint(factory.post("/", REMOTE_ADDR="192.0.2.10"))

    assert first.status_code == 200
    assert second.status_code == 429


@pytest.mark.django_db
def test_trusted_proxy_uses_forwarded_client_identity(settings):
    """A known proxy must not collapse every public client into one bucket."""
    settings.TRUSTED_PROXY_IPS = frozenset({"10.0.0.10"})

    @rate_limit(scope="proxy-test", limit=1, window_seconds=60)
    def endpoint(request):
        return HttpResponse("ok")

    factory = RequestFactory()
    first = endpoint(
        factory.post(
            "/",
            REMOTE_ADDR="10.0.0.10",
            HTTP_X_FORWARDED_FOR="198.51.100.1",
        )
    )
    second = endpoint(
        factory.post(
            "/",
            REMOTE_ADDR="10.0.0.10",
            HTTP_X_FORWARDED_FOR="198.51.100.2",
        )
    )

    assert first.status_code == 200
    assert second.status_code == 200


@pytest.mark.django_db
def test_trusted_proxy_requires_single_forwarded_address(settings):
    """Multi-hop X-Forwarded-For from a trusted proxy must not become client identity."""
    settings.TRUSTED_PROXY_IPS = frozenset({"10.0.0.10"})

    @rate_limit(scope="proxy-multihop-test", limit=1, window_seconds=60)
    def endpoint(request):
        return HttpResponse("ok")

    factory = RequestFactory()
    first = endpoint(
        factory.post(
            "/",
            REMOTE_ADDR="10.0.0.10",
            HTTP_X_FORWARDED_FOR="198.51.100.1, 203.0.113.9",
        )
    )
    # Both requests collapse to the proxy itself because the header is invalid.
    second = endpoint(
        factory.post(
            "/",
            REMOTE_ADDR="10.0.0.10",
            HTTP_X_FORWARDED_FOR="198.51.100.2, 203.0.113.10",
        )
    )

    assert first.status_code == 200
    assert second.status_code == 429


@pytest.mark.django_db
def test_untrusted_peer_cannot_spoof_forwarded_identity(settings):
    """Forwarding headers from public peers must not bypass an IP limit."""
    settings.TRUSTED_PROXY_IPS = frozenset({"10.0.0.10"})

    @rate_limit(scope="spoof-test", limit=1, window_seconds=60)
    def endpoint(request):
        return HttpResponse("ok")

    factory = RequestFactory()
    first = endpoint(
        factory.post(
            "/",
            REMOTE_ADDR="203.0.113.8",
            HTTP_X_FORWARDED_FOR="198.51.100.1",
        )
    )
    second = endpoint(
        factory.post(
            "/",
            REMOTE_ADDR="203.0.113.8",
            HTTP_X_FORWARDED_FOR="198.51.100.2",
        )
    )

    assert first.status_code == 200
    assert second.status_code == 429


@pytest.mark.django_db
def test_login_post_is_throttled(client):
    """Repeated credential guesses must eventually stop before authentication."""
    url = reverse("login")

    responses = [
        client.post(url, {"username": "unknown", "password": "wrong"})
        for _ in range(6)
    ]

    assert [response.status_code for response in responses[:5]] == [200] * 5
    assert responses[5].status_code == 429


@pytest.mark.django_db
def test_password_reset_post_is_throttled(client):
    """Repeated reset requests must not become an outbound-email denial of service."""
    url = reverse("password_reset")

    responses = [
        client.post(url, {"email": "unknown@example.com"})
        for _ in range(6)
    ]

    assert [response.status_code for response in responses[:5]] == [302] * 5
    assert responses[5].status_code == 429


@pytest.mark.django_db
def test_request_from_new_identity_reclaims_expired_rows():
    """Rotating one-off identities must not grow the limiter table forever."""
    expired = RequestRateLimit.objects.create(
        scope="old-scope",
        identity_hash="a" * 64,
        window_start=1,
        count=1,
        expires_at=timezone.now() - timedelta(seconds=1),
    )

    @rate_limit(scope="cleanup-test", limit=1, window_seconds=60)
    def endpoint(request):
        return HttpResponse("ok")

    response = endpoint(RequestFactory().post("/", REMOTE_ADDR="192.0.2.55"))

    assert response.status_code == 200
    assert not RequestRateLimit.objects.filter(pk=expired.pk).exists()
