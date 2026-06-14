"""Tests for the unauthenticated health probe access gate."""

from unittest.mock import MagicMock, patch

import pytest
from django.test import RequestFactory, override_settings

from config.urls import _is_internal_probe, health_check


@pytest.mark.unit
class TestHealthProbeAccess:
    """Health checks must not trust a proxy's private REMOTE_ADDR."""

    def test_loopback_request_is_internal_probe(self) -> None:
        """Local Docker HEALTHCHECK calls should work without a shared token."""
        request = RequestFactory().get('/health/', REMOTE_ADDR='127.0.0.1')

        assert _is_internal_probe(request) is True

    def test_private_proxy_address_without_token_is_rejected(self) -> None:
        """RFC1918 proxy IPs are not proof that the original client is trusted."""
        request = RequestFactory().get('/health/', REMOTE_ADDR='10.0.0.5')

        assert _is_internal_probe(request) is False

    @override_settings(HEALTH_CHECK_TOKEN='probe-secret')
    def test_matching_probe_token_is_internal_probe(self) -> None:
        """Production load balancers can authenticate with a shared probe token."""
        request = RequestFactory().get(
            '/health/',
            HTTP_X_HEALTH_CHECK_TOKEN='probe-secret',
            REMOTE_ADDR='203.0.113.10',
        )

        assert _is_internal_probe(request) is True

    @override_settings(HEALTH_CHECK_TOKEN='probe-secret')
    def test_wrong_probe_token_is_rejected(self) -> None:
        """A public request with the wrong token must not reach the DB probe."""
        request = RequestFactory().get(
            '/health/',
            HTTP_X_HEALTH_CHECK_TOKEN='wrong-secret',
            REMOTE_ADDR='203.0.113.10',
        )

        assert _is_internal_probe(request) is False

    @override_settings(HEALTH_CHECK_TOKEN='probe-secret')
    def test_loopback_without_configured_token_header_is_rejected(self) -> None:
        """A configured token prevents same-host reverse proxies bypassing auth."""
        request = RequestFactory().get('/health/', REMOTE_ADDR='127.0.0.1')

        assert _is_internal_probe(request) is False

    @override_settings(HEALTH_CHECK_TOKEN='probe-secret')
    def test_loopback_with_wrong_token_is_rejected(self) -> None:
        """Loopback is not trusted when a token is configured but does not match."""
        request = RequestFactory().get(
            '/health/',
            HTTP_X_HEALTH_CHECK_TOKEN='wrong-secret',
            REMOTE_ADDR='127.0.0.1',
        )

        assert _is_internal_probe(request) is False

    def test_forbidden_probe_does_not_query_database(self) -> None:
        """Denied callers should get 403 before any database cursor is opened."""
        request = RequestFactory().get('/health/', REMOTE_ADDR='10.0.0.5')

        with patch('config.urls.connection.cursor') as cursor:
            response = health_check(request)

        assert response.status_code == 403
        cursor.assert_not_called()

    @override_settings(HEALTH_CHECK_TOKEN='probe-secret')
    def test_token_probe_runs_database_check(self) -> None:
        """Authenticated probes should execute the readiness query and return ok."""
        request = RequestFactory().get(
            '/health/',
            HTTP_X_HEALTH_CHECK_TOKEN='probe-secret',
            REMOTE_ADDR='203.0.113.10',
        )
        cursor = MagicMock()
        cursor.fetchone.return_value = (1,)

        with patch('config.urls.connection.cursor') as cursor_factory:
            cursor_factory.return_value.__enter__.return_value = cursor
            response = health_check(request)

        assert response.status_code == 200
        cursor.execute.assert_called_once_with('SELECT 1')
