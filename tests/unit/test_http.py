"""Tests for migratowl.core.http — shared httpx client pool."""

from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest

import migratowl.core.http as http_mod
from migratowl.core.http import close_http_client, get_http_client


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
