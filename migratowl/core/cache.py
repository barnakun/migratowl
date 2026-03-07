"""Incremental scan cache — persists impact assessments keyed by dep version range."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

from migratowl.config import settings

_cache_locks: dict[str, asyncio.Lock] = {}


def _get_cache_lock(project_path: str) -> asyncio.Lock:
    """Return a lock for the given project's cache file.

    Safe without additional locking because asyncio dict mutation is atomic
    within a single event loop tick (no await between check and assignment).
    """
    return _cache_locks.setdefault(project_path, asyncio.Lock())


def _cache_file(project_path: str) -> Path:
    """Return the JSON cache file path for the given project."""
    project_hash = hashlib.sha256(project_path.encode()).hexdigest()[:8]
    cache_dir = Path(settings.cache_path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{project_hash}.json"


def _dep_key(dep_name: str, current_version: str, latest_version: str) -> str:
    return f"{dep_name}:{current_version}:{latest_version}"


async def get_cached_assessment(
    project_path: str,
    dep_name: str,
    current_version: str,
    latest_version: str,
) -> dict | None:
    """Return a previously stored impact assessment, or None on cache miss."""
    async with _get_cache_lock(project_path):
        cache_file = _cache_file(project_path)
        if not cache_file.exists():
            return None
        try:
            data: dict = json.loads(cache_file.read_text())
        except (json.JSONDecodeError, OSError):
            return None
        return data.get(_dep_key(dep_name, current_version, latest_version))


async def set_cached_assessment(
    project_path: str,
    dep_name: str,
    current_version: str,
    latest_version: str,
    assessment: dict,
) -> None:
    """Persist an impact assessment to the project cache."""
    async with _get_cache_lock(project_path):
        cache_file = _cache_file(project_path)
        try:
            data: dict = json.loads(cache_file.read_text()) if cache_file.exists() else {}
        except (json.JSONDecodeError, OSError):
            data = {}
        data[_dep_key(dep_name, current_version, latest_version)] = assessment
        cache_file.write_text(json.dumps(data, indent=2))
