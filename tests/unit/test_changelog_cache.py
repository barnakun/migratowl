"""Tests for migratowl.core.changelog_cache — changelog HTTP fetch cache."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch


class TestGetCachedChangelog:
    def test_cache_miss_returns_none_when_no_file(self, tmp_path: Path) -> None:
        with patch("migratowl.core.changelog_cache.settings") as mock_settings:
            mock_settings.changelog_cache_path = str(tmp_path / "cl-cache")
            mock_settings.changelog_cache_ttl_minutes = 1440

            from migratowl.core.changelog_cache import get_cached_changelog

            result = get_cached_changelog("requests")

        assert result is None

    def test_cache_hit_returns_stored_text_and_warnings(self, tmp_path: Path) -> None:
        with patch("migratowl.core.changelog_cache.settings") as mock_settings:
            mock_settings.changelog_cache_path = str(tmp_path / "cl-cache")
            mock_settings.changelog_cache_ttl_minutes = 1440

            from migratowl.core.changelog_cache import get_cached_changelog, set_cached_changelog

            set_cached_changelog("requests", "## 2.32.0\n- fix", ["warn1"])
            result = get_cached_changelog("requests")

        assert result is not None
        text, warnings = result
        assert text == "## 2.32.0\n- fix"
        assert warnings == ["warn1"]

    def test_cache_hit_empty_warnings(self, tmp_path: Path) -> None:
        with patch("migratowl.core.changelog_cache.settings") as mock_settings:
            mock_settings.changelog_cache_path = str(tmp_path / "cl-cache")
            mock_settings.changelog_cache_ttl_minutes = 1440

            from migratowl.core.changelog_cache import get_cached_changelog, set_cached_changelog

            set_cached_changelog("flask", "changelog text", [])
            result = get_cached_changelog("flask")

        assert result is not None
        text, warnings = result
        assert text == "changelog text"
        assert warnings == []

    def test_ttl_expiry_returns_none(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "cl-cache"
        cache_dir.mkdir(parents=True)

        # Write a cache entry with fetched_at far in the past
        cache_file = cache_dir / "requests.json"
        cache_file.write_text(
            json.dumps({"text": "old changelog", "warnings": [], "fetched_at": 0})
        )

        with patch("migratowl.core.changelog_cache.settings") as mock_settings:
            mock_settings.changelog_cache_path = str(cache_dir)
            mock_settings.changelog_cache_ttl_minutes = 1440

            from migratowl.core.changelog_cache import get_cached_changelog

            result = get_cached_changelog("requests")

        assert result is None

    def test_ttl_zero_disables_cache(self, tmp_path: Path) -> None:
        with patch("migratowl.core.changelog_cache.settings") as mock_settings:
            mock_settings.changelog_cache_path = str(tmp_path / "cl-cache")
            mock_settings.changelog_cache_ttl_minutes = 0

            from migratowl.core.changelog_cache import get_cached_changelog, set_cached_changelog

            set_cached_changelog("requests", "changelog", [])
            result = get_cached_changelog("requests")

        assert result is None

    def test_corrupt_json_returns_none(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "cl-cache"
        cache_dir.mkdir(parents=True)
        (cache_dir / "requests.json").write_text("not valid json{{{")

        with patch("migratowl.core.changelog_cache.settings") as mock_settings:
            mock_settings.changelog_cache_path = str(cache_dir)
            mock_settings.changelog_cache_ttl_minutes = 1440

            from migratowl.core.changelog_cache import get_cached_changelog

            result = get_cached_changelog("requests")

        assert result is None

    def test_missing_keys_returns_none(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "cl-cache"
        cache_dir.mkdir(parents=True)
        (cache_dir / "requests.json").write_text(json.dumps({"text": "only text"}))

        with patch("migratowl.core.changelog_cache.settings") as mock_settings:
            mock_settings.changelog_cache_path = str(cache_dir)
            mock_settings.changelog_cache_ttl_minutes = 1440

            from migratowl.core.changelog_cache import get_cached_changelog

            result = get_cached_changelog("requests")

        assert result is None

    def test_scoped_npm_package_name_with_slash(self, tmp_path: Path) -> None:
        """Dep names like @expo/vector-icons must not create subdirectories."""
        with patch("migratowl.core.changelog_cache.settings") as mock_settings:
            mock_settings.changelog_cache_path = str(tmp_path / "cl-cache")
            mock_settings.changelog_cache_ttl_minutes = 1440

            from migratowl.core.changelog_cache import get_cached_changelog, set_cached_changelog

            set_cached_changelog("@expo/vector-icons", "changelog", ["w"])
            result = get_cached_changelog("@expo/vector-icons")

        assert result is not None
        assert result[0] == "changelog"
        # Must be a flat file, not a subdirectory
        cache_dir = tmp_path / "cl-cache"
        assert not (cache_dir / "@expo").exists()

    def test_different_deps_independent(self, tmp_path: Path) -> None:
        with patch("migratowl.core.changelog_cache.settings") as mock_settings:
            mock_settings.changelog_cache_path = str(tmp_path / "cl-cache")
            mock_settings.changelog_cache_ttl_minutes = 1440

            from migratowl.core.changelog_cache import get_cached_changelog, set_cached_changelog

            set_cached_changelog("requests", "requests changelog", [])
            set_cached_changelog("flask", "flask changelog", ["w"])

            r1 = get_cached_changelog("requests")
            r2 = get_cached_changelog("flask")
            r3 = get_cached_changelog("django")

        assert r1 is not None and r1[0] == "requests changelog"
        assert r2 is not None and r2[0] == "flask changelog"
        assert r3 is None


class TestSetCachedChangelog:
    def test_creates_cache_directory_if_missing(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "deep" / "nested" / "cl-cache"

        with patch("migratowl.core.changelog_cache.settings") as mock_settings:
            mock_settings.changelog_cache_path = str(cache_dir)
            mock_settings.changelog_cache_ttl_minutes = 1440

            from migratowl.core.changelog_cache import set_cached_changelog

            set_cached_changelog("flask", "text", [])

        assert cache_dir.exists()
        assert (cache_dir / "flask.json").exists()

    def test_overwrites_existing_entry(self, tmp_path: Path) -> None:
        with patch("migratowl.core.changelog_cache.settings") as mock_settings:
            mock_settings.changelog_cache_path = str(tmp_path / "cl-cache")
            mock_settings.changelog_cache_ttl_minutes = 1440

            from migratowl.core.changelog_cache import get_cached_changelog, set_cached_changelog

            set_cached_changelog("requests", "old text", ["old warn"])
            set_cached_changelog("requests", "new text", [])

            result = get_cached_changelog("requests")

        assert result is not None
        assert result[0] == "new text"
        assert result[1] == []

    def test_stored_file_contains_fetched_at_timestamp(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "cl-cache"

        with patch("migratowl.core.changelog_cache.settings") as mock_settings:
            mock_settings.changelog_cache_path = str(cache_dir)
            mock_settings.changelog_cache_ttl_minutes = 1440

            from migratowl.core.changelog_cache import set_cached_changelog

            before = time.time()
            set_cached_changelog("requests", "text", [])
            after = time.time()

        data = json.loads((cache_dir / "requests.json").read_text())
        assert before <= data["fetched_at"] <= after
