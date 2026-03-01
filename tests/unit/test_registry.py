"""Tests for migratowl.core.registry — package registry queries and outdated detection."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import httpx

from migratowl.core.registry import (
    _extract_changelog_url,
    _extract_repo_url,
    _query_npm,
    _query_pypi,
    find_outdated,
    query_registry,
)
from migratowl.models.schemas import Dependency, Ecosystem, OutdatedDependency, RegistryInfo

# --- Fixtures ---

PYPI_RESPONSE = {
    "info": {
        "name": "requests",
        "version": "2.31.0",
        "home_page": "https://requests.readthedocs.io",
        "project_urls": {
            "Source": "https://github.com/psf/requests",
            "Changelog": "https://github.com/psf/requests/blob/main/HISTORY.md",
        },
    },
}

NPM_RESPONSE = {
    "name": "express",
    "dist-tags": {"latest": "4.19.2"},
    "homepage": "http://expressjs.com/",
    "repository": {"type": "git", "url": "git+https://github.com/expressjs/express.git"},
}


def _mock_pypi_response() -> httpx.Response:
    return httpx.Response(200, json=PYPI_RESPONSE, request=httpx.Request("GET", "https://pypi.org"))


def _mock_npm_response() -> httpx.Response:
    return httpx.Response(200, json=NPM_RESPONSE, request=httpx.Request("GET", "https://registry.npmjs.org"))


# --- query_registry ---


class TestQueryRegistry:
    async def test_dispatches_to_pypi(self) -> None:
        with patch("migratowl.core.registry._query_pypi", new_callable=AsyncMock) as mock:
            mock.return_value = RegistryInfo(name="requests", latest_version="2.31.0")
            result = await query_registry("requests", Ecosystem.PYTHON)
            mock.assert_awaited_once_with("requests")
            assert result.name == "requests"

    async def test_dispatches_to_npm(self) -> None:
        with patch("migratowl.core.registry._query_npm", new_callable=AsyncMock) as mock:
            mock.return_value = RegistryInfo(name="express", latest_version="4.19.2")
            result = await query_registry("express", Ecosystem.NODEJS)
            mock.assert_awaited_once_with("express")
            assert result.name == "express"


# --- _query_pypi ---


class TestQueryPyPI:
    async def test_returns_registry_info(self) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=_mock_pypi_response())
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await _query_pypi("requests")

        assert isinstance(result, RegistryInfo)
        assert result.name == "requests"
        assert result.latest_version == "2.31.0"
        assert result.repository_url == "https://github.com/psf/requests"
        assert result.changelog_url == "https://github.com/psf/requests/blob/main/HISTORY.md"

    async def test_handles_missing_project_urls(self) -> None:
        data = {"info": {"name": "simple", "version": "1.0.0", "home_page": None, "project_urls": None}}
        resp = httpx.Response(200, json=data, request=httpx.Request("GET", "https://pypi.org"))
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await _query_pypi("simple")

        assert result.repository_url is None
        assert result.changelog_url is None


# --- _query_npm ---


class TestQueryNpm:
    async def test_returns_registry_info(self) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=_mock_npm_response())
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await _query_npm("express")

        assert isinstance(result, RegistryInfo)
        assert result.name == "express"
        assert result.latest_version == "4.19.2"
        assert result.homepage_url == "http://expressjs.com/"
        assert "github.com/expressjs/express" in (result.repository_url or "")

    async def test_handles_missing_repo(self) -> None:
        data = {"name": "minimal", "dist-tags": {"latest": "1.0.0"}}
        resp = httpx.Response(200, json=data, request=httpx.Request("GET", "https://registry.npmjs.org"))
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await _query_npm("minimal")

        assert result.repository_url is None


# --- _extract_repo_url ---


class TestExtractRepoUrl:
    def test_source_key(self) -> None:
        urls = {"Source": "https://github.com/user/repo"}
        assert _extract_repo_url(urls) == "https://github.com/user/repo"

    def test_repository_key(self) -> None:
        urls = {"Repository": "https://github.com/user/repo"}
        assert _extract_repo_url(urls) == "https://github.com/user/repo"

    def test_github_key(self) -> None:
        urls = {"GitHub": "https://github.com/user/repo"}
        assert _extract_repo_url(urls) == "https://github.com/user/repo"

    def test_homepage_with_github(self) -> None:
        urls = {"Homepage": "https://github.com/user/repo"}
        assert _extract_repo_url(urls) == "https://github.com/user/repo"

    def test_no_match_returns_none(self) -> None:
        urls = {"Documentation": "https://docs.example.com"}
        assert _extract_repo_url(urls) is None

    def test_none_input(self) -> None:
        assert _extract_repo_url(None) is None

    def test_strips_hash_fragment_from_url(self) -> None:
        """PyPI sometimes includes #readme or similar fragments in repo URLs
        (e.g. tree-sitter-language-pack). The fragment must be stripped."""
        urls = {"Homepage": "https://github.com/Goldziher/tree-sitter-language-pack#readme"}
        assert _extract_repo_url(urls) == "https://github.com/Goldziher/tree-sitter-language-pack"


# --- _extract_changelog_url ---


class TestExtractChangelogUrl:
    def test_changelog_key(self) -> None:
        urls = {"Changelog": "https://github.com/user/repo/blob/main/CHANGELOG.md"}
        assert _extract_changelog_url(urls) == "https://github.com/user/repo/blob/main/CHANGELOG.md"

    def test_changes_key(self) -> None:
        urls = {"Changes": "https://example.com/changes"}
        assert _extract_changelog_url(urls) == "https://example.com/changes"

    def test_release_notes_key(self) -> None:
        urls = {"Release Notes": "https://example.com/releases"}
        assert _extract_changelog_url(urls) == "https://example.com/releases"

    def test_no_match_returns_none(self) -> None:
        urls = {"Homepage": "https://example.com"}
        assert _extract_changelog_url(urls) is None

    def test_none_input(self) -> None:
        assert _extract_changelog_url(None) is None


# --- find_outdated ---


class TestFindOutdated:
    async def test_finds_outdated_deps(self) -> None:
        deps = [
            Dependency(name="requests", current_version="2.28.0", ecosystem=Ecosystem.PYTHON, manifest_path="r.txt"),
            Dependency(name="flask", current_version="2.3.0", ecosystem=Ecosystem.PYTHON, manifest_path="r.txt"),
        ]

        async def mock_query(name: str, eco: Ecosystem) -> RegistryInfo:
            if name == "requests":
                return RegistryInfo(
                    name="requests",
                    latest_version="2.31.0",
                    repository_url="https://github.com/psf/requests",
                )
            return RegistryInfo(
                name="flask",
                latest_version="2.3.0",
            )

        with patch("migratowl.core.registry.query_registry", side_effect=mock_query):
            outdated, errors = await find_outdated(deps)

        assert len(outdated) == 1
        assert outdated[0].name == "requests"
        assert outdated[0].current_version == "2.28.0"
        assert outdated[0].latest_version == "2.31.0"
        assert isinstance(outdated[0], OutdatedDependency)
        assert errors == []

    async def test_all_up_to_date(self) -> None:
        deps = [
            Dependency(name="requests", current_version="2.31.0", ecosystem=Ecosystem.PYTHON, manifest_path="r.txt"),
        ]

        async def mock_query(name: str, eco: Ecosystem) -> RegistryInfo:
            return RegistryInfo(name="requests", latest_version="2.31.0")

        with patch("migratowl.core.registry.query_registry", side_effect=mock_query):
            outdated, errors = await find_outdated(deps)

        assert len(outdated) == 0
        assert errors == []

    async def test_handles_query_error_gracefully(self) -> None:
        deps = [
            Dependency(name="nonexistent", current_version="1.0.0", ecosystem=Ecosystem.PYTHON, manifest_path="r.txt"),
        ]

        async def mock_query(name: str, eco: Ecosystem) -> RegistryInfo:
            raise httpx.HTTPStatusError(
                "Not Found",
                request=httpx.Request("GET", "https://pypi.org"),
                response=httpx.Response(404),
            )

        with patch("migratowl.core.registry.query_registry", side_effect=mock_query):
            outdated, errors = await find_outdated(deps)

        assert len(outdated) == 0

    async def test_empty_deps_list(self) -> None:
        outdated, errors = await find_outdated([])
        assert outdated == []
        assert errors == []

    async def test_pep440_equivalent_versions_not_marked_outdated(self) -> None:
        """'0.13' and '0.13.0' are identical by PEP 440 — must not appear as outdated.
        String comparison '0.13' != '0.13.0' is a false positive (tree-sitter-language-pack regression)."""
        deps = [
            Dependency(name="pkg", current_version="0.13", ecosystem=Ecosystem.PYTHON, manifest_path="r.txt"),
        ]

        async def mock_query(name: str, eco: Ecosystem) -> RegistryInfo:
            return RegistryInfo(name="pkg", latest_version="0.13.0")

        with patch("migratowl.core.registry.query_registry", side_effect=mock_query):
            outdated, errors = await find_outdated(deps)

        assert len(outdated) == 0, "0.13 == 0.13.0 by PEP 440; should not be outdated"

    async def test_returns_error_for_failed_dep(self) -> None:
        """Failed registry queries must appear in the errors list, not silently dropped."""
        deps = [
            Dependency(name="bad-pkg", current_version="1.0.0", ecosystem=Ecosystem.PYTHON, manifest_path="r.txt"),
            Dependency(name="good-pkg", current_version="1.0.0", ecosystem=Ecosystem.PYTHON, manifest_path="r.txt"),
        ]

        async def mock_query(name: str, eco: Ecosystem) -> RegistryInfo:
            if name == "bad-pkg":
                raise httpx.HTTPStatusError(
                    "Not Found",
                    request=httpx.Request("GET", "https://pypi.org"),
                    response=httpx.Response(404),
                )
            return RegistryInfo(name="good-pkg", latest_version="2.0.0")

        with patch("migratowl.core.registry.query_registry", side_effect=mock_query):
            outdated, errors = await find_outdated(deps)

        assert len(errors) == 1
        assert "bad-pkg" in errors[0]

    async def test_caps_concurrent_registry_queries(self) -> None:
        """At most max_concurrent_registry_queries registry calls run concurrently."""
        n_deps = 30
        max_concurrent = 5
        deps = [
            Dependency(name=f"pkg-{i}", current_version="1.0.0", ecosystem=Ecosystem.PYTHON, manifest_path="r.txt")
            for i in range(n_deps)
        ]

        concurrent_count = 0
        peak_concurrent = 0

        async def slow_query(name: str, eco: Ecosystem) -> RegistryInfo:
            nonlocal concurrent_count, peak_concurrent
            concurrent_count += 1
            peak_concurrent = max(peak_concurrent, concurrent_count)
            await asyncio.sleep(0.01)
            concurrent_count -= 1
            return RegistryInfo(name=name, latest_version="2.0.0")

        with (
            patch("migratowl.core.registry.query_registry", side_effect=slow_query),
            patch("migratowl.core.registry.settings") as mock_settings,
        ):
            mock_settings.max_concurrent_registry_queries = max_concurrent
            await find_outdated(deps)

        assert peak_concurrent <= max_concurrent
