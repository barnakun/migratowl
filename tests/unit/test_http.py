"""Tests for migratowl.core.http — shared httpx client pool."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

import migratowl.core.http as http_mod
from migratowl.core.http import RetryTransport, close_http_client, get_http_client


@pytest.fixture(autouse=True)
def _reset_singleton() -> None:
    """Ensure each test starts with a fresh singleton."""
    http_mod._client = None


class TestGetHttpClient:
    def test_returns_async_client(self) -> None:
        client = get_http_client()
        assert isinstance(client, httpx.AsyncClient)

    def test_returns_same_instance(self) -> None:
        a = get_http_client()
        b = get_http_client()
        assert a is b

    def test_has_follow_redirects(self) -> None:
        client = get_http_client()
        assert client.follow_redirects is True

    def test_has_correct_timeout(self) -> None:
        with patch("migratowl.core.http.settings") as mock_settings:
            mock_settings.http_timeout = 42.0
            http_mod._client = None  # force re-creation
            client = get_http_client()
        assert client.timeout.connect == 42.0
        assert client.timeout.read == 42.0


class TestCloseHttpClient:
    @pytest.mark.asyncio
    async def test_resets_singleton(self) -> None:
        _ = get_http_client()
        assert http_mod._client is not None
        await close_http_client()
        assert http_mod._client is None

    @pytest.mark.asyncio
    async def test_next_call_creates_new_client(self) -> None:
        first = get_http_client()
        await close_http_client()
        second = get_http_client()
        assert first is not second

    @pytest.mark.asyncio
    async def test_noop_when_none(self) -> None:
        assert http_mod._client is None
        await close_http_client()  # should not raise


def _mock_response(status: int, headers: dict[str, str] | None = None) -> httpx.Response:
    """Build a minimal httpx.Response for transport-level tests."""
    return httpx.Response(status_code=status, headers=headers or {})


def _make_transport(
    responses: list[httpx.Response | httpx.ConnectError],
    *,
    max_retries: int = 3,
    backoff_base: float = 0.5,
) -> tuple[RetryTransport, AsyncMock]:
    """Create a RetryTransport with a mock inner transport returning *responses* in order."""
    inner = AsyncMock(spec=httpx.AsyncBaseTransport)
    side_effects: list[httpx.Response | Exception] = []
    for r in responses:
        if isinstance(r, Exception):
            side_effects.append(r)
        else:
            side_effects.append(r)
    inner.handle_async_request.side_effect = side_effects
    transport = RetryTransport(inner, max_retries=max_retries, backoff_base=backoff_base)
    return transport, inner


# Dummy request used by all transport tests.
_REQ = httpx.Request("GET", "https://example.com")


class TestRetryTransport:
    @pytest.mark.asyncio
    async def test_no_retry_on_success(self) -> None:
        transport, inner = _make_transport([_mock_response(200)])
        resp = await transport.handle_async_request(_REQ)
        assert resp.status_code == 200
        assert inner.handle_async_request.call_count == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", [400, 401, 403, 404])
    async def test_no_retry_on_client_error(self, status: int) -> None:
        transport, inner = _make_transport([_mock_response(status)])
        resp = await transport.handle_async_request(_REQ)
        assert resp.status_code == status
        assert inner.handle_async_request.call_count == 1

    @pytest.mark.asyncio
    async def test_retries_on_500_then_succeeds(self) -> None:
        transport, inner = _make_transport([_mock_response(500), _mock_response(200)])
        with patch("migratowl.core.http.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            resp = await transport.handle_async_request(_REQ)
        assert resp.status_code == 200
        assert inner.handle_async_request.call_count == 2
        mock_sleep.assert_called_once()

    @pytest.mark.asyncio
    async def test_retries_on_429_then_succeeds(self) -> None:
        transport, inner = _make_transport([_mock_response(429), _mock_response(200)])
        with patch("migratowl.core.http.asyncio.sleep", new_callable=AsyncMock):
            resp = await transport.handle_async_request(_REQ)
        assert resp.status_code == 200
        assert inner.handle_async_request.call_count == 2

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", [502, 503, 504])
    async def test_retries_on_5xx(self, status: int) -> None:
        transport, inner = _make_transport([_mock_response(status), _mock_response(200)])
        with patch("migratowl.core.http.asyncio.sleep", new_callable=AsyncMock):
            resp = await transport.handle_async_request(_REQ)
        assert resp.status_code == 200
        assert inner.handle_async_request.call_count == 2

    @pytest.mark.asyncio
    async def test_returns_error_after_max_retries(self) -> None:
        transport, inner = _make_transport(
            [_mock_response(503)] * 4, max_retries=3
        )
        with patch("migratowl.core.http.asyncio.sleep", new_callable=AsyncMock):
            resp = await transport.handle_async_request(_REQ)
        assert resp.status_code == 503
        assert inner.handle_async_request.call_count == 4  # 1 + 3 retries

    @pytest.mark.asyncio
    async def test_retries_on_connect_error(self) -> None:
        transport, inner = _make_transport(
            [httpx.ConnectError("connection refused"), _mock_response(200)]
        )
        with patch("migratowl.core.http.asyncio.sleep", new_callable=AsyncMock):
            resp = await transport.handle_async_request(_REQ)
        assert resp.status_code == 200
        assert inner.handle_async_request.call_count == 2

    @pytest.mark.asyncio
    async def test_raises_connect_error_after_max_retries(self) -> None:
        transport, _ = _make_transport(
            [httpx.ConnectError("fail")] * 4, max_retries=3
        )
        with patch("migratowl.core.http.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(httpx.ConnectError):
                await transport.handle_async_request(_REQ)

    @pytest.mark.asyncio
    async def test_respects_retry_after_header(self) -> None:
        transport, _ = _make_transport(
            [_mock_response(429, {"Retry-After": "2"}), _mock_response(200)]
        )
        with patch("migratowl.core.http.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await transport.handle_async_request(_REQ)
        mock_sleep.assert_called_once_with(2.0)

    @pytest.mark.asyncio
    async def test_exponential_backoff_delays(self) -> None:
        transport, _ = _make_transport(
            [_mock_response(503)] * 3 + [_mock_response(200)],
            max_retries=3,
            backoff_base=0.5,
        )
        with patch("migratowl.core.http.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await transport.handle_async_request(_REQ)
        delays = [call.args[0] for call in mock_sleep.call_args_list]
        assert delays == [0.5, 1.0, 2.0]  # base*2^0, base*2^1, base*2^2

    @pytest.mark.asyncio
    async def test_zero_retries_disables_retry(self) -> None:
        transport, inner = _make_transport(
            [_mock_response(503)], max_retries=0
        )
        resp = await transport.handle_async_request(_REQ)
        assert resp.status_code == 503
        assert inner.handle_async_request.call_count == 1

    def test_client_uses_retry_transport(self) -> None:
        client = get_http_client()
        assert isinstance(client._transport, RetryTransport)
