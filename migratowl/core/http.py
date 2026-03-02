"""Shared httpx client pool — reuses TCP connections across changelog and registry calls."""

from __future__ import annotations

import httpx

from migratowl.config import settings

_client: httpx.AsyncClient | None = None


def get_http_client() -> httpx.AsyncClient:
    """Return the shared httpx client, creating it lazily on first call."""
    global _client
    if _client is None:
        _client = httpx.AsyncClient(follow_redirects=True, timeout=settings.http_timeout)
    return _client


async def close_http_client() -> None:
    """Close the shared client and reset the singleton. Safe to call when no client exists."""
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None
