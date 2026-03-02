"""Changelog cache — persists raw changelog text keyed by dep name with TTL."""

from __future__ import annotations

import json
import time
from pathlib import Path

from migratowl.config import settings


def _safe_filename(dep_name: str) -> str:
    """Convert a dep name to a safe flat filename (e.g. @expo/vector-icons → @expo__vector-icons)."""
    return dep_name.replace("/", "__")


def _cache_file(dep_name: str) -> Path:
    """Return the JSON cache file path for the given dependency."""
    cache_dir = Path(settings.changelog_cache_path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{_safe_filename(dep_name)}.json"


def get_cached_changelog(dep_name: str) -> tuple[str, list[str]] | None:
    """Return cached (text, warnings) if fresh, or None on miss/expiry."""
    ttl_minutes = settings.changelog_cache_ttl_minutes
    if ttl_minutes <= 0:
        return None

    cache_file = _cache_file(dep_name)
    if not cache_file.exists():
        return None

    try:
        data: dict = json.loads(cache_file.read_text())
    except (json.JSONDecodeError, OSError):
        return None

    if "text" not in data or "fetched_at" not in data:
        return None

    age_seconds = time.time() - data["fetched_at"]
    if age_seconds > ttl_minutes * 60:
        return None

    return data["text"], data.get("warnings", [])


def set_cached_changelog(dep_name: str, text: str, warnings: list[str]) -> None:
    """Persist changelog text and warnings for a dependency."""
    cache_file = _cache_file(dep_name)
    payload = {
        "text": text,
        "warnings": warnings,
        "fetched_at": time.time(),
    }
    cache_file.write_text(json.dumps(payload, indent=2))
