"""Package registry queries (PyPI, npm) and outdated dependency detection."""

from __future__ import annotations

import asyncio
import logging
import re

import httpx
from packaging.version import InvalidVersion, Version

from migratowl.config import settings
from migratowl.models.schemas import Dependency, Ecosystem, OutdatedDependency, RegistryInfo

logger = logging.getLogger(__name__)

# Keys to check for repository URLs in PyPI project_urls, in priority order.
_REPO_KEYS = ("Source", "Source Code", "Repository", "Code", "GitHub")

# Keys to check for changelog URLs in PyPI project_urls, in priority order.
_CHANGELOG_KEYS = ("Changelog", "Changes", "Change Log", "Release Notes", "History", "What's New")


async def query_registry(name: str, ecosystem: Ecosystem) -> RegistryInfo:
    """Query the appropriate package registry for the latest version info."""
    if ecosystem == Ecosystem.PYTHON:
        return await _query_pypi(name)
    elif ecosystem == Ecosystem.NODEJS:
        return await _query_npm(name)
    else:
        msg = f"Unsupported ecosystem: {ecosystem}"
        raise ValueError(msg)


async def _query_pypi(name: str) -> RegistryInfo:
    """Query PyPI JSON API for package info."""
    url = f"https://pypi.org/pypi/{name}/json"
    async with httpx.AsyncClient() as client:
        resp = await client.get(url)
        resp.raise_for_status()
        data = resp.json()

    info = data["info"]
    project_urls = info.get("project_urls") or {}

    return RegistryInfo(
        name=info["name"],
        latest_version=info["version"],
        homepage_url=info.get("home_page"),
        repository_url=_extract_repo_url(project_urls),
        changelog_url=_extract_changelog_url(project_urls),
    )


async def _query_npm(name: str) -> RegistryInfo:
    """Query npm registry for package info."""
    url = f"https://registry.npmjs.org/{name}"
    async with httpx.AsyncClient() as client:
        resp = await client.get(url)
        resp.raise_for_status()
        data = resp.json()

    latest_version = data.get("dist-tags", {}).get("latest", "0.0.0")
    homepage = data.get("homepage")

    # Extract repository URL.
    repo_url: str | None = None
    repo = data.get("repository")
    if isinstance(repo, dict):
        raw_url = repo.get("url", "")
        # Clean git+https://...git -> https://...
        repo_url = re.sub(r"^git\+", "", raw_url)
        repo_url = re.sub(r"\.git$", "", repo_url)
    elif isinstance(repo, str):
        repo_url = repo

    return RegistryInfo(
        name=data["name"],
        latest_version=latest_version,
        homepage_url=homepage,
        repository_url=repo_url or None,
        changelog_url=None,
    )


def _extract_repo_url(project_urls: dict | None) -> str | None:
    """Extract repository URL from PyPI project_urls dict."""
    if not project_urls:
        return None

    # Check explicit repo keys first.
    for key in _REPO_KEYS:
        if key in project_urls:
            return _strip_url_fragment(project_urls[key])

    # Fall back: any URL containing github.com or gitlab.com.
    for key, url in project_urls.items():
        if isinstance(url, str) and ("github.com" in url or "gitlab.com" in url):
            return _strip_url_fragment(url)

    return None


def _is_newer(latest: str, current: str) -> bool:
    """Return True only if latest is strictly newer than current by PEP 440.

    Falls back to string inequality for non-PEP-440 version strings (e.g. npm).
    Prevents false positives like '0.13' vs '0.13.0' which are equal by PEP 440.
    """
    try:
        return Version(latest) > Version(current)
    except InvalidVersion:
        return latest != current


def _strip_url_fragment(url: str) -> str:
    """Remove any #fragment from a URL (e.g. 'github.com/org/repo#readme' → 'github.com/org/repo')."""
    return url.split("#")[0]


def _extract_changelog_url(project_urls: dict | None) -> str | None:
    """Extract changelog URL from PyPI project_urls dict."""
    if not project_urls:
        return None

    for key in _CHANGELOG_KEYS:
        if key in project_urls:
            result: str = project_urls[key]
            return result

    return None


async def find_outdated(deps: list[Dependency]) -> tuple[list[OutdatedDependency], list[str]]:
    """Query registries for all deps; return (outdated, errors).

    Concurrent queries are capped at settings.max_concurrent_registry_queries
    to avoid hitting PyPI/npm rate limits with large dependency lists.
    Failed lookups are collected into the errors list rather than silently dropped.
    """
    if not deps:
        return [], []

    sem = asyncio.Semaphore(settings.max_concurrent_registry_queries)
    errors: list[str] = []

    async def _check_one(dep: Dependency) -> OutdatedDependency | None:
        async with sem:
            try:
                info = await query_registry(dep.name, dep.ecosystem)
            except Exception as exc:
                msg = f"Registry query failed for {dep.name}: {exc}"
                logger.warning(msg)
                errors.append(msg)
                return None

            if _is_newer(info.latest_version, dep.current_version):
                return OutdatedDependency(
                    name=dep.name,
                    current_version=dep.current_version,
                    latest_version=info.latest_version,
                    ecosystem=dep.ecosystem,
                    manifest_path=dep.manifest_path,
                    homepage_url=info.homepage_url,
                    repository_url=info.repository_url,
                    changelog_url=info.changelog_url,
                )
            return None

    results = await asyncio.gather(*[_check_one(d) for d in deps])
    return [r for r in results if r is not None], errors
