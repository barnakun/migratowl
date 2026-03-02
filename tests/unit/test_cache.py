"""Tests for migratowl.core.cache — incremental scan cache."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch


class TestGetCachedAssessment:
    def test_cache_miss_returns_none_when_no_file(self, tmp_path: Path) -> None:
        with patch("migratowl.core.cache.settings") as mock_settings:
            mock_settings.cache_path = str(tmp_path / "cache")

            from migratowl.core.cache import get_cached_assessment

            result = get_cached_assessment("/project", "requests", "2.28.0", "2.32.0")

        assert result is None

    def test_cache_hit_returns_stored_assessment(self, tmp_path: Path) -> None:
        assessment = {"dep_name": "requests", "summary": "No impact", "overall_severity": "info"}

        with patch("migratowl.core.cache.settings") as mock_settings:
            mock_settings.cache_path = str(tmp_path / "cache")

            from migratowl.core.cache import get_cached_assessment, set_cached_assessment

            set_cached_assessment("/project", "requests", "2.28.0", "2.32.0", assessment)
            result = get_cached_assessment("/project", "requests", "2.28.0", "2.32.0")

        assert result == assessment

    def test_cache_miss_different_version(self, tmp_path: Path) -> None:
        assessment = {"dep_name": "requests", "summary": "breaking", "overall_severity": "critical"}

        with patch("migratowl.core.cache.settings") as mock_settings:
            mock_settings.cache_path = str(tmp_path / "cache")

            from migratowl.core.cache import get_cached_assessment, set_cached_assessment

            set_cached_assessment("/project", "requests", "2.28.0", "2.32.0", assessment)
            # Different latest version — must be a cache miss
            result = get_cached_assessment("/project", "requests", "2.28.0", "2.33.0")

        assert result is None

    def test_cache_miss_different_project(self, tmp_path: Path) -> None:
        assessment = {"dep_name": "requests", "summary": "ok", "overall_severity": "info"}

        with patch("migratowl.core.cache.settings") as mock_settings:
            mock_settings.cache_path = str(tmp_path / "cache")

            from migratowl.core.cache import get_cached_assessment, set_cached_assessment

            set_cached_assessment("/project-a", "requests", "2.28.0", "2.32.0", assessment)
            result = get_cached_assessment("/project-b", "requests", "2.28.0", "2.32.0")

        assert result is None

    def test_corrupt_json_returns_none(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(parents=True)

        with patch("migratowl.core.cache.settings") as mock_settings:
            mock_settings.cache_path = str(cache_dir)

            from migratowl.core.cache import _cache_file, get_cached_assessment

            cache_file = _cache_file("/project")
            cache_file.write_text("not valid json{{{")

            result = get_cached_assessment("/project", "requests", "2.28.0", "2.32.0")

        assert result is None


class TestSetCachedAssessment:
    def test_creates_cache_directory_if_missing(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "deep" / "nested" / "cache"

        with patch("migratowl.core.cache.settings") as mock_settings:
            mock_settings.cache_path = str(cache_dir)

            from migratowl.core.cache import set_cached_assessment

            set_cached_assessment("/project", "flask", "2.0.0", "3.0.0", {"summary": "ok"})

        assert cache_dir.exists()

    def test_multiple_deps_stored_in_same_file(self, tmp_path: Path) -> None:
        with patch("migratowl.core.cache.settings") as mock_settings:
            mock_settings.cache_path = str(tmp_path / "cache")

            from migratowl.core.cache import get_cached_assessment, set_cached_assessment

            set_cached_assessment("/project", "requests", "2.28.0", "2.32.0", {"dep": "requests"})
            set_cached_assessment("/project", "flask", "2.0.0", "3.0.0", {"dep": "flask"})

            r1 = get_cached_assessment("/project", "requests", "2.28.0", "2.32.0")
            r2 = get_cached_assessment("/project", "flask", "2.0.0", "3.0.0")

        assert r1 == {"dep": "requests"}
        assert r2 == {"dep": "flask"}

    def test_overwrites_existing_entry(self, tmp_path: Path) -> None:
        with patch("migratowl.core.cache.settings") as mock_settings:
            mock_settings.cache_path = str(tmp_path / "cache")

            from migratowl.core.cache import get_cached_assessment, set_cached_assessment

            set_cached_assessment("/project", "requests", "2.28.0", "2.32.0", {"v": 1})
            set_cached_assessment("/project", "requests", "2.28.0", "2.32.0", {"v": 2})

            result = get_cached_assessment("/project", "requests", "2.28.0", "2.32.0")

        assert result == {"v": 2}

    def test_different_projects_use_separate_files(self, tmp_path: Path) -> None:
        with patch("migratowl.core.cache.settings") as mock_settings:
            mock_settings.cache_path = str(tmp_path / "cache")

            from migratowl.core.cache import _cache_file

            file_a = _cache_file("/project-a")
            file_b = _cache_file("/project-b")

        assert file_a != file_b
