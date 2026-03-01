"""Tests for changelog fetching and chunking."""

from unittest.mock import AsyncMock, patch

import pytest

from migratowl.core.changelog import (
    chunk_changelog_by_version,
    fetch_changelog,
    filter_chunks_by_version_range,
)

SAMPLE_CHANGELOG = """\
# Changelog

## v3.0.0
- Breaking: Removed old_func
- Added new_func

## v2.0.0
- Breaking: Renamed module
- Fixed bug

## v1.0.0
- Initial release
"""

BRACKET_CHANGELOG = """\
# Changelog

## [3.0.0]
- Breaking change A

## [2.0.0]
- Breaking change B

## [1.0.0]
- Initial release
"""


class TestChunkChangelogByVersion:
    def test_standard_v_prefix_headers(self) -> None:
        chunks = chunk_changelog_by_version(SAMPLE_CHANGELOG)
        assert len(chunks) == 3
        versions = [c["version"] for c in chunks]
        assert versions == ["3.0.0", "2.0.0", "1.0.0"]

    def test_bracket_version_headers(self) -> None:
        chunks = chunk_changelog_by_version(BRACKET_CHANGELOG)
        assert len(chunks) == 3
        versions = [c["version"] for c in chunks]
        assert versions == ["3.0.0", "2.0.0", "1.0.0"]

    def test_content_preserved(self) -> None:
        chunks = chunk_changelog_by_version(SAMPLE_CHANGELOG)
        v3_chunk = next(c for c in chunks if c["version"] == "3.0.0")
        assert "Removed old_func" in v3_chunk["content"]
        assert "Added new_func" in v3_chunk["content"]

    def test_empty_text(self) -> None:
        chunks = chunk_changelog_by_version("")
        assert chunks == []

    def test_no_version_headers(self) -> None:
        text = "Just some text\nwithout any version headers"
        chunks = chunk_changelog_by_version(text)
        assert chunks == []

    def test_single_version(self) -> None:
        text = "## v1.0.0\n- Only version"
        chunks = chunk_changelog_by_version(text)
        assert len(chunks) == 1
        assert chunks[0]["version"] == "1.0.0"
        assert "Only version" in chunks[0]["content"]

    def test_mixed_header_formats(self) -> None:
        text = """\
## v3.0.0
- Change A

## [2.0.0]
- Change B

## 1.0.0
- Change C
"""
        chunks = chunk_changelog_by_version(text)
        assert len(chunks) == 3
        versions = [c["version"] for c in chunks]
        assert versions == ["3.0.0", "2.0.0", "1.0.0"]

    def test_rst_style_headers(self) -> None:
        """RST-style changelogs like requests use `2.32.5 (date)\\n---` format."""
        text = """\
Release History
===============

2.32.5 (2025-08-18)
-------------------

**Bugfixes**

- Fixed SSL caching issue.

2.32.4 (2025-06-10)
-------------------

**Security**
- Fixed CVE-2024-47081.

2.32.3 (2024-05-29)
-------------------

- Minor fixes.
"""
        chunks = chunk_changelog_by_version(text)
        assert len(chunks) == 3
        versions = [c["version"] for c in chunks]
        assert versions == ["2.32.5", "2.32.4", "2.32.3"]
        assert "Fixed SSL" in chunks[0]["content"]

    def test_rst_style_no_date(self) -> None:
        """RST headers without parenthesized date."""
        text = """\
1.0.0
-----

- Initial release.
"""
        chunks = chunk_changelog_by_version(text)
        assert len(chunks) == 1
        assert chunks[0]["version"] == "1.0.0"

    def test_flask_migrate_release_header_format(self) -> None:
        """Flask-Migrate CHANGES.md uses '## Release X.Y.Z - YYYY-MM-DD' format."""
        text = """\
## Release 4.1.0 - 2024-10-12

- Removed MigrateCommand (Flask-Script integration dropped).

## Release 4.0.0 - 2023-04-23

- New CLI integration.

## Release 2.0.2 - 2017-12-01

- Bug fixes.
"""
        chunks = chunk_changelog_by_version(text)
        assert len(chunks) == 3
        versions = [c["version"] for c in chunks]
        assert versions == ["4.1.0", "4.0.0", "2.0.2"]
        assert "MigrateCommand" in chunks[0]["content"]

    def test_bold_release_header_format(self) -> None:
        """Flask-Migrate CHANGELOG uses '**Release 4.0.6** - 2024-03-09' bold format."""
        text = """\
**Release 4.0.6** - 2024-03-09

- Fixed SSL issue.

**Release 4.0.5** - 2024-01-15

- Another fix.

**Release 4.0.0** - 2023-11-01

- Major release.
"""
        chunks = chunk_changelog_by_version(text)
        assert len(chunks) == 3
        versions = [c["version"] for c in chunks]
        assert versions == ["4.0.6", "4.0.5", "4.0.0"]
        assert "Fixed SSL" in chunks[0]["content"]

    def test_flask_rst_version_prefix_format(self) -> None:
        """Flask-style CHANGES.rst uses 'Version X.Y.Z\\n-----------' format."""
        text = """\
Version 3.1.0
-------------

Released 2025-02-05

- Removed deprecated code.
- ``g`` is now a proxy.

Version 3.0.0
-------------

Released 2024-09-30

- Drop Python 3.8 support.
"""
        chunks = chunk_changelog_by_version(text)
        assert len(chunks) == 2
        assert chunks[0]["version"] == "3.1.0"
        assert chunks[1]["version"] == "3.0.0"
        assert "Removed deprecated" in chunks[0]["content"]


class TestFilterChunksByVersionRange:
    def test_filters_between_current_and_latest(self) -> None:
        chunks = chunk_changelog_by_version(SAMPLE_CHANGELOG)
        filtered = filter_chunks_by_version_range(chunks, "1.0.0", "3.0.0")
        versions = [c["version"] for c in filtered]
        assert "1.0.0" not in versions
        assert "2.0.0" in versions
        assert "3.0.0" in versions

    def test_excludes_current_includes_latest(self) -> None:
        chunks = chunk_changelog_by_version(SAMPLE_CHANGELOG)
        filtered = filter_chunks_by_version_range(chunks, "1.0.0", "2.0.0")
        versions = [c["version"] for c in filtered]
        assert versions == ["2.0.0"]

    def test_empty_chunks(self) -> None:
        filtered = filter_chunks_by_version_range([], "1.0.0", "3.0.0")
        assert filtered == []

    def test_no_matching_range(self) -> None:
        chunks = chunk_changelog_by_version(SAMPLE_CHANGELOG)
        filtered = filter_chunks_by_version_range(chunks, "3.0.0", "4.0.0")
        assert filtered == []

    def test_same_current_and_latest(self) -> None:
        chunks = chunk_changelog_by_version(SAMPLE_CHANGELOG)
        filtered = filter_chunks_by_version_range(chunks, "2.0.0", "2.0.0")
        assert filtered == []


class TestFetchFromUrl:
    @pytest.mark.asyncio
    async def test_html_with_no_version_headers_raises_for_fallback(self) -> None:
        """HTML with no parseable version headers must raise to trigger GitHub fallback."""
        from migratowl.core.changelog import _fetch_from_url

        html_content = "<!DOCTYPE html><html><body><h1>Flask Changelog</h1></body></html>"
        mock_response = type("R", (), {"text": html_content, "raise_for_status": lambda self: None})()

        with patch("migratowl.core.changelog.httpx.AsyncClient") as mock_client_cls:
            mock_client = mock_client_cls.return_value.__aenter__.return_value
            mock_client.get = AsyncMock(return_value=mock_response)
            with pytest.raises(ValueError, match="HTML"):
                await _fetch_from_url("https://example.com/changes/")

    @pytest.mark.asyncio
    async def test_html_with_version_headers_stripped_and_returned(self) -> None:
        """HTML pages containing version headers (e.g. ReadTheDocs) are stripped
        to plain text and returned rather than rejected."""
        from migratowl.core.changelog import _fetch_from_url

        html_content = """\
<!DOCTYPE html>
<html><body>
<h2>Version 2.0.0</h2>
<ul><li>Breaking: removed old API.</li></ul>
<h2>Version 1.0.0</h2>
<ul><li>Initial release.</li></ul>
</body></html>"""
        mock_response = type("R", (), {"text": html_content, "raise_for_status": lambda self: None})()

        with patch("migratowl.core.changelog.httpx.AsyncClient") as mock_client_cls:
            mock_client = mock_client_cls.return_value.__aenter__.return_value
            mock_client.get = AsyncMock(return_value=mock_response)
            result = await _fetch_from_url("https://example.com/changes/")

        assert "2.0.0" in result
        assert "1.0.0" in result
        assert "<html>" not in result

    @pytest.mark.asyncio
    async def test_plain_text_response_returned_as_is(self) -> None:
        from migratowl.core.changelog import _fetch_from_url

        rst_content = "Version 3.0\n-----------\n\n- Some change.\n"
        mock_response = type("R", (), {"text": rst_content, "raise_for_status": lambda self: None})()

        with patch("migratowl.core.changelog.httpx.AsyncClient") as mock_client_cls:
            mock_client = mock_client_cls.return_value.__aenter__.return_value
            mock_client.get = AsyncMock(return_value=mock_response)
            result = await _fetch_from_url("https://example.com/CHANGES.rst")

        assert result == rst_content


class TestFetchFromGithub:
    @pytest.mark.asyncio
    async def test_tries_changes_rst_filename(self) -> None:
        """CHANGES.rst (used by Flask, Werkzeug, etc.) must be in the filename list."""
        from migratowl.core.changelog import _fetch_from_github

        fetched_urls: list[str] = []

        async def fake_get(url: str) -> object:
            fetched_urls.append(url)
            status = 200 if url.endswith("CHANGES.rst") else 404
            return type("R", (), {"status_code": status, "text": "Version 1.0\n---\n- x"})()

        with patch("migratowl.core.changelog.httpx.AsyncClient") as mock_client_cls:
            mock_client = mock_client_cls.return_value.__aenter__.return_value
            mock_client.get = AsyncMock(side_effect=fake_get)
            result = await _fetch_from_github("https://github.com/pallets/flask/")

        assert any("CHANGES.rst" in url for url in fetched_urls)
        assert "Version 1.0" in result

    @pytest.mark.asyncio
    async def test_falls_back_to_master_when_main_returns_404(self) -> None:
        """If all filenames 404 on main, retry every filename on master.

        Many repos (e.g. Flask-Migrate) still use master as their default branch.
        """
        from migratowl.core.changelog import _fetch_from_github

        fetched_urls: list[str] = []

        async def fake_get(url: str) -> object:
            fetched_urls.append(url)
            # main branch: everything 404; master + CHANGELOG.md: 200
            if "/main/" in url:
                return type("R", (), {"status_code": 404, "text": ""})()
            if "/master/" in url and url.endswith("CHANGELOG.md"):
                return type("R", (), {"status_code": 200, "text": "## 1.0.0\n- Change"})()
            return type("R", (), {"status_code": 404, "text": ""})()

        with patch("migratowl.core.changelog.httpx.AsyncClient") as mock_client_cls:
            mock_client = mock_client_cls.return_value.__aenter__.return_value
            mock_client.get = AsyncMock(side_effect=fake_get)
            result = await _fetch_from_github("https://github.com/owner/repo")

        assert any("/master/" in url for url in fetched_urls)
        assert "Change" in result

    @pytest.mark.asyncio
    async def test_raises_when_all_branches_and_filenames_return_404(self) -> None:
        """FileNotFoundError is raised only after exhausting all branches and filenames."""
        from migratowl.core.changelog import _fetch_from_github

        async def fake_get(url: str) -> object:
            return type("R", (), {"status_code": 404, "text": ""})()

        with patch("migratowl.core.changelog.httpx.AsyncClient") as mock_client_cls:
            mock_client = mock_client_cls.return_value.__aenter__.return_value
            mock_client.get = AsyncMock(side_effect=fake_get)
            with pytest.raises(FileNotFoundError):
                await _fetch_from_github("https://github.com/owner/repo")

    @pytest.mark.asyncio
    async def test_doc_subpath_tried_after_all_root_files_fail(self) -> None:
        """When all root-level files 404, docs/ subdirectory paths are tried.
        This covers packages like Flask-WTF whose changelog lives at docs/changes.rst.
        """
        from migratowl.core.changelog import _fetch_from_github

        fetched_urls: list[str] = []

        async def fake_get(url: str) -> object:
            fetched_urls.append(url)
            if "docs/changes.rst" in url:
                return type("R", (), {"status_code": 200, "text": "## 1.0.0\n- Change.\n"})()
            return type("R", (), {"status_code": 404, "text": ""})()

        with patch("migratowl.core.changelog.httpx.AsyncClient") as mock_client_cls:
            mock_client = mock_client_cls.return_value.__aenter__.return_value
            mock_client.get = AsyncMock(side_effect=fake_get)
            result = await _fetch_from_github("https://github.com/owner/repo")

        assert any("docs/changes.rst" in url for url in fetched_urls)
        assert "Change" in result

    @pytest.mark.asyncio
    async def test_stub_file_with_github_blob_url_is_followed(self) -> None:
        """A root CHANGELOG.rst that is a stub (no version headers) but contains
        a GitHub blob URL is followed to the real changelog file.
        This covers packages like pytest whose root CHANGELOG.rst is a redirect notice.
        """
        from migratowl.core.changelog import _fetch_from_github

        stub_text = (
            "Changelog\n=========\n\n"
            "The source document can be found at: "
            "https://github.com/owner/repo/blob/main/doc/en/changelog.rst\n"
        )
        real_content = "## 3.0.0\n- Real change.\n\n## 2.0.0\n- Another.\n"

        fetched_urls: list[str] = []

        async def fake_get(url: str) -> object:
            fetched_urls.append(url)
            if url == "https://raw.githubusercontent.com/owner/repo/main/CHANGELOG.rst":
                return type("R", (), {"status_code": 200, "text": stub_text})()
            if url == "https://raw.githubusercontent.com/owner/repo/main/doc/en/changelog.rst":
                return type("R", (), {"status_code": 200, "text": real_content})()
            return type("R", (), {"status_code": 404, "text": ""})()

        with patch("migratowl.core.changelog.httpx.AsyncClient") as mock_client_cls:
            mock_client = mock_client_cls.return_value.__aenter__.return_value
            mock_client.get = AsyncMock(side_effect=fake_get)
            result = await _fetch_from_github("https://github.com/owner/repo")

        assert any("doc/en/changelog.rst" in url for url in fetched_urls)
        assert "Real change" in result

    @pytest.mark.asyncio
    async def test_strips_hash_fragment_from_repository_url(self) -> None:
        """URLs with #fragment (e.g. '...pack#readme') must not embed the fragment
        into raw.githubusercontent.com paths (tree-sitter-language-pack regression)."""
        from migratowl.core.changelog import _fetch_from_github

        fetched_urls: list[str] = []

        async def fake_get(url: str) -> object:
            fetched_urls.append(url)
            if "tree-sitter-language-pack/main/CHANGELOG.md" in url:
                return type("R", (), {"status_code": 200, "text": "## 0.13.0\n- Initial."})()
            return type("R", (), {"status_code": 404, "text": ""})()

        with patch("migratowl.core.changelog.httpx.AsyncClient") as mock_client_cls:
            mock_client = mock_client_cls.return_value.__aenter__.return_value
            mock_client.get = AsyncMock(side_effect=fake_get)
            result = await _fetch_from_github(
                "https://github.com/Goldziher/tree-sitter-language-pack#readme"
            )

        assert all("#" not in u for u in fetched_urls), "Fragment leaked into raw URL"
        assert "Initial" in result

    @pytest.mark.asyncio
    async def test_stub_file_without_github_url_continues_to_next_candidate(self) -> None:
        """A stub with no GitHub blob URL is skipped; search continues to the next file."""
        from migratowl.core.changelog import _fetch_from_github

        stub_text = "Changelog\n=========\n\nSee https://docs.example.com/changes for full history.\n"
        real_content = "## 2.0.0\n- Fixed things.\n"

        async def fake_get(url: str) -> object:
            if url.endswith("CHANGELOG.rst") and "/main/" in url:
                return type("R", (), {"status_code": 200, "text": stub_text})()
            if url.endswith("CHANGES.rst") and "/main/" in url:
                return type("R", (), {"status_code": 200, "text": real_content})()
            return type("R", (), {"status_code": 404, "text": ""})()

        with patch("migratowl.core.changelog.httpx.AsyncClient") as mock_client_cls:
            mock_client = mock_client_cls.return_value.__aenter__.return_value
            mock_client.get = AsyncMock(side_effect=fake_get)
            result = await _fetch_from_github("https://github.com/owner/repo")

        assert "Fixed things" in result


class TestFetchChangelog:
    @pytest.mark.asyncio
    async def test_fetch_from_changelog_url(self) -> None:
        with patch(
            "migratowl.core.changelog._fetch_from_url",
            new_callable=AsyncMock,
            return_value="# Changelog\n## v1.0.0\n- Initial",
        ) as mock_fetch:
            text, warnings = await fetch_changelog(
                changelog_url="https://example.com/CHANGELOG.md",
                repository_url=None,
                dep_name="test-pkg",
            )
            assert "Changelog" in text
            assert warnings == []
            mock_fetch.assert_called_once_with("https://example.com/CHANGELOG.md")

    @pytest.mark.asyncio
    async def test_fallback_to_github(self) -> None:
        with (
            patch(
                "migratowl.core.changelog._fetch_from_url",
                new_callable=AsyncMock,
                side_effect=Exception("not found"),
            ),
            patch(
                "migratowl.core.changelog._fetch_from_github",
                new_callable=AsyncMock,
                return_value="# Changes\n## v1.0.0\n- Init",
            ) as mock_github,
        ):
            text, warnings = await fetch_changelog(
                changelog_url="https://broken.com/CHANGELOG.md",
                repository_url="https://github.com/owner/repo",
                dep_name="test-pkg",
            )
            assert "Changes" in text
            assert warnings == []
            mock_github.assert_called_once_with("https://github.com/owner/repo")

    @pytest.mark.asyncio
    async def test_returns_empty_with_warning_when_all_fail(self) -> None:
        with patch(
            "migratowl.core.changelog._fetch_from_github",
            new_callable=AsyncMock,
            side_effect=Exception("not found"),
        ):
            text, warnings = await fetch_changelog(
                changelog_url=None,
                repository_url="https://github.com/owner/repo",
                dep_name="test-pkg",
            )
            assert text == ""
            assert len(warnings) > 0
            assert "test-pkg" in warnings[0]

    @pytest.mark.asyncio
    async def test_no_urls_provided_returns_warning(self) -> None:
        text, warnings = await fetch_changelog(
            changelog_url=None,
            repository_url=None,
            dep_name="test-pkg",
        )
        assert text == ""
        assert len(warnings) > 0
        assert "test-pkg" in warnings[0]

    @pytest.mark.asyncio
    async def test_fallback_to_github_releases_when_file_probe_fails(self) -> None:
        """When raw file probing finds no CHANGELOG.md, GitHub Releases API is tried."""
        with (
            patch(
                "migratowl.core.changelog._fetch_from_github",
                new_callable=AsyncMock,
                side_effect=FileNotFoundError("no changelog file"),
            ),
            patch(
                "migratowl.core.changelog._fetch_from_github_releases",
                new_callable=AsyncMock,
                return_value="## v1.0.9\n- Fix bug\n\n## v1.0.8\n- Add feature",
            ) as mock_releases,
        ):
            text, warnings = await fetch_changelog(
                changelog_url=None,
                repository_url="https://github.com/owner/repo",
                dep_name="test-pkg",
            )
            assert "1.0.9" in text
            assert warnings == []
            mock_releases.assert_called_once_with("https://github.com/owner/repo")

    @pytest.mark.asyncio
    async def test_returns_warning_when_github_releases_also_fails(self) -> None:
        """Warning is returned only after all three strategies are exhausted."""
        with (
            patch(
                "migratowl.core.changelog._fetch_from_github",
                new_callable=AsyncMock,
                side_effect=FileNotFoundError("no changelog file"),
            ),
            patch(
                "migratowl.core.changelog._fetch_from_github_releases",
                new_callable=AsyncMock,
                side_effect=Exception("rate limited"),
            ),
        ):
            text, warnings = await fetch_changelog(
                changelog_url=None,
                repository_url="https://github.com/owner/repo",
                dep_name="test-pkg",
            )
            assert text == ""
            assert "test-pkg" in warnings[0]


class TestFetchFromGithubReleases:
    @pytest.mark.asyncio
    async def test_converts_releases_to_changelog_text(self) -> None:
        """GitHub releases response body fields become parseable changelog sections."""
        from migratowl.core.changelog import _fetch_from_github_releases

        releases = [
            {"tag_name": "v1.0.9", "body": "- Fix critical bug", "draft": False, "prerelease": False},
            {"tag_name": "v1.0.8", "body": "- Add new feature", "draft": False, "prerelease": False},
        ]
        mock_response = type(
            "R",
            (),
            {"json": lambda self: releases, "raise_for_status": lambda self: None, "headers": {}},
        )()

        with patch("migratowl.core.changelog.httpx.AsyncClient") as mock_client_cls:
            mock_client = mock_client_cls.return_value.__aenter__.return_value
            mock_client.get = AsyncMock(return_value=mock_response)
            result = await _fetch_from_github_releases("https://github.com/owner/repo")

        chunks = chunk_changelog_by_version(result)
        versions = [c["version"] for c in chunks]
        assert "1.0.9" in versions
        assert "1.0.8" in versions
        assert "Fix critical bug" in result
        assert "Add new feature" in result

    @pytest.mark.asyncio
    async def test_skips_draft_and_prerelease_entries(self) -> None:
        """Draft and prerelease entries are excluded from the changelog text."""
        from migratowl.core.changelog import _fetch_from_github_releases

        releases = [
            {"tag_name": "v2.0.0-beta", "body": "Beta stuff", "draft": False, "prerelease": True},
            {"tag_name": "v1.0.0-draft", "body": "Draft stuff", "draft": True, "prerelease": False},
            {"tag_name": "v1.0.0", "body": "Stable release", "draft": False, "prerelease": False},
        ]
        mock_response = type(
            "R",
            (),
            {"json": lambda self: releases, "raise_for_status": lambda self: None, "headers": {}},
        )()

        with patch("migratowl.core.changelog.httpx.AsyncClient") as mock_client_cls:
            mock_client = mock_client_cls.return_value.__aenter__.return_value
            mock_client.get = AsyncMock(return_value=mock_response)
            result = await _fetch_from_github_releases("https://github.com/owner/repo")

        assert "Beta stuff" not in result
        assert "Draft stuff" not in result
        assert "Stable release" in result

    @pytest.mark.asyncio
    async def test_raises_when_no_usable_releases(self) -> None:
        """FileNotFoundError is raised when there are no non-draft, non-prerelease releases."""
        from migratowl.core.changelog import _fetch_from_github_releases

        releases: list = []
        mock_response = type(
            "R",
            (),
            {"json": lambda self: releases, "raise_for_status": lambda self: None, "headers": {}},
        )()

        with patch("migratowl.core.changelog.httpx.AsyncClient") as mock_client_cls:
            mock_client = mock_client_cls.return_value.__aenter__.return_value
            mock_client.get = AsyncMock(return_value=mock_response)
            with pytest.raises(FileNotFoundError):
                await _fetch_from_github_releases("https://github.com/owner/repo")

    @pytest.mark.asyncio
    async def test_sends_auth_header_when_token_configured(self) -> None:
        """Authorization header is sent when MIGRATOWL_GITHUB_TOKEN is set in settings."""
        from migratowl.core.changelog import _fetch_from_github_releases

        releases = [
            {"tag_name": "v1.0.0", "body": "- Change", "draft": False, "prerelease": False},
        ]
        mock_response = type(
            "R",
            (),
            {"json": lambda self: releases, "raise_for_status": lambda self: None, "headers": {}},
        )()

        captured_headers: list[dict] = []

        async def fake_get(url: str, **kwargs) -> object:
            captured_headers.append(kwargs.get("headers", {}))
            return mock_response

        with (
            patch("migratowl.core.changelog.httpx.AsyncClient") as mock_client_cls,
            patch("migratowl.core.changelog.settings") as mock_settings,
        ):
            mock_settings.github_token = "ghp_testtoken123"
            mock_client = mock_client_cls.return_value.__aenter__.return_value
            mock_client.get = AsyncMock(side_effect=fake_get)
            await _fetch_from_github_releases("https://github.com/owner/repo")

        assert any("Authorization" in h for h in captured_headers)
        assert any("ghp_testtoken123" in str(h) for h in captured_headers)

    @pytest.mark.asyncio
    async def test_calls_correct_github_api_url(self) -> None:
        """The GitHub Releases API endpoint is constructed from owner/repo in the URL."""
        from migratowl.core.changelog import _fetch_from_github_releases

        releases = [
            {"tag_name": "v1.0.0", "body": "- Initial", "draft": False, "prerelease": False},
        ]
        mock_response = type(
            "R",
            (),
            {"json": lambda self: releases, "raise_for_status": lambda self: None, "headers": {}},
        )()

        captured_urls: list[str] = []

        async def fake_get(url: str, **kwargs) -> object:
            captured_urls.append(url)
            return mock_response

        with patch("migratowl.core.changelog.httpx.AsyncClient") as mock_client_cls:
            mock_client = mock_client_cls.return_value.__aenter__.return_value
            mock_client.get = AsyncMock(side_effect=fake_get)
            await _fetch_from_github_releases("https://github.com/langchain-ai/langsmith-sdk")

        assert any("api.github.com/repos/langchain-ai/langsmith-sdk/releases" in u for u in captured_urls)

    @pytest.mark.asyncio
    async def test_strips_hash_fragment_from_repository_url(self) -> None:  # noqa: E501
        """URLs like 'github.com/Goldziher/tree-sitter-language-pack#readme' must not
        embed the fragment into the API path (tree-sitter-language-pack regression)."""
        from migratowl.core.changelog import _fetch_from_github_releases

        releases = [
            {"tag_name": "v0.13.0", "body": "- Initial", "draft": False, "prerelease": False},
        ]
        mock_response = type(
            "R",
            (),
            {"json": lambda self: releases, "raise_for_status": lambda self: None, "headers": {}},
        )()

        captured_urls: list[str] = []

        async def fake_get(url: str, **kwargs) -> object:
            captured_urls.append(url)
            return mock_response

        with patch("migratowl.core.changelog.httpx.AsyncClient") as mock_client_cls:
            mock_client = mock_client_cls.return_value.__aenter__.return_value
            mock_client.get = AsyncMock(side_effect=fake_get)
            await _fetch_from_github_releases(
                "https://github.com/Goldziher/tree-sitter-language-pack#readme"
            )

        assert len(captured_urls) == 1
        assert captured_urls[0] == (
            "https://api.github.com/repos/Goldziher/tree-sitter-language-pack/releases?per_page=100"
        )


class TestTryUrlsConcurrently:
    """Tests for the concurrent URL fetching helper."""

    @pytest.mark.asyncio
    async def test_returns_text_of_first_valid_url(self) -> None:
        """Returns the text of the first URL with parseable version chunks."""
        import asyncio

        from migratowl.core.changelog import _try_urls_concurrently

        async def fake_get(url: str) -> object:
            if url == "https://example.com/2":
                return type("R", (), {"status_code": 200, "text": "## v1.0.0\n- Change"})()
            return type("R", (), {"status_code": 404, "text": ""})()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=fake_get)
        sem = asyncio.Semaphore(10)

        result = await _try_urls_concurrently(
            mock_client,
            ["https://example.com/1", "https://example.com/2", "https://example.com/3"],
            sem,
        )
        assert result is not None
        assert "1.0.0" in result

    @pytest.mark.asyncio
    async def test_returns_none_when_all_404(self) -> None:
        """Returns None when all URLs return 404."""
        import asyncio

        from migratowl.core.changelog import _try_urls_concurrently

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=type("R", (), {"status_code": 404, "text": ""})())
        sem = asyncio.Semaphore(10)

        result = await _try_urls_concurrently(
            mock_client,
            ["https://example.com/1", "https://example.com/2"],
            sem,
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_200_but_no_version_chunks(self) -> None:
        """Returns None when all URLs return 200 but with no parseable version headers."""
        import asyncio

        from migratowl.core.changelog import _try_urls_concurrently

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(
            return_value=type("R", (), {"status_code": 200, "text": "No version headers here"})()
        )
        sem = asyncio.Semaphore(10)

        result = await _try_urls_concurrently(mock_client, ["https://example.com/1"], sem)
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_for_empty_url_list(self) -> None:
        """Returns None immediately for an empty URL list."""
        import asyncio

        from migratowl.core.changelog import _try_urls_concurrently

        mock_client = AsyncMock()
        sem = asyncio.Semaphore(10)

        result = await _try_urls_concurrently(mock_client, [], sem)
        assert result is None
        mock_client.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_semaphore_caps_peak_concurrency(self) -> None:
        """Concurrent tasks must not exceed the semaphore limit."""
        import asyncio

        from migratowl.core.changelog import _try_urls_concurrently

        cap = 3
        sem = asyncio.Semaphore(cap)
        peak = 0
        active = 0
        lock = asyncio.Lock()

        async def fake_get(url: str) -> object:
            nonlocal peak, active
            async with lock:
                active += 1
                peak = max(peak, active)
            await asyncio.sleep(0.005)
            async with lock:
                active -= 1
            return type("R", (), {"status_code": 404, "text": ""})()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=fake_get)

        urls = [f"https://example.com/{i}" for i in range(10)]
        await _try_urls_concurrently(mock_client, urls, sem)

        assert peak <= cap


class TestFetchChangelogStrategyOrdering:
    """Tests for token-based strategy ordering in fetch_changelog."""

    @pytest.mark.asyncio
    async def test_with_token_releases_api_tried_before_file_probe(self) -> None:
        """When github_token is set, Releases API is tried before raw file probing."""
        call_order: list[str] = []

        async def mock_releases(repo_url: str) -> str:
            call_order.append("releases")
            return "## v1.0.0\n- Done"

        async def mock_file_probe(repo_url: str) -> str:
            call_order.append("file_probe")
            return "## v1.0.0\n- Done"

        with (
            patch("migratowl.core.changelog._fetch_from_github_releases", mock_releases),
            patch("migratowl.core.changelog._fetch_from_github", mock_file_probe),
            patch("migratowl.core.changelog.settings") as mock_settings,
        ):
            mock_settings.github_token = "ghp_testtoken"
            await fetch_changelog(
                changelog_url=None,
                repository_url="https://github.com/owner/repo",
                dep_name="pkg",
            )

        assert call_order[0] == "releases", f"Expected releases first, got: {call_order}"

    @pytest.mark.asyncio
    async def test_without_token_file_probe_tried_before_releases_api(self) -> None:
        """Without github_token, raw file probing is tried before Releases API."""
        call_order: list[str] = []

        async def mock_releases(repo_url: str) -> str:
            call_order.append("releases")
            return "## v1.0.0\n- Done"

        async def mock_file_probe(repo_url: str) -> str:
            call_order.append("file_probe")
            return "## v1.0.0\n- Done"

        with (
            patch("migratowl.core.changelog._fetch_from_github_releases", mock_releases),
            patch("migratowl.core.changelog._fetch_from_github", mock_file_probe),
            patch("migratowl.core.changelog.settings") as mock_settings,
        ):
            mock_settings.github_token = None
            await fetch_changelog(
                changelog_url=None,
                repository_url="https://github.com/owner/repo",
                dep_name="pkg",
            )

        assert call_order[0] == "file_probe", f"Expected file_probe first, got: {call_order}"

    @pytest.mark.asyncio
    async def test_with_token_and_releases_fails_file_probe_is_fallback(self) -> None:
        """With token, if Releases API fails, raw file probing is used as fallback."""
        call_order: list[str] = []

        async def mock_releases(repo_url: str) -> str:
            call_order.append("releases")
            raise FileNotFoundError("no releases")

        async def mock_file_probe(repo_url: str) -> str:
            call_order.append("file_probe")
            return "## v1.0.0\n- Found via file probe"

        with (
            patch("migratowl.core.changelog._fetch_from_github_releases", mock_releases),
            patch("migratowl.core.changelog._fetch_from_github", mock_file_probe),
            patch("migratowl.core.changelog.settings") as mock_settings,
        ):
            mock_settings.github_token = "ghp_testtoken"
            text, warnings = await fetch_changelog(
                changelog_url=None,
                repository_url="https://github.com/owner/repo",
                dep_name="pkg",
            )

        assert call_order == ["releases", "file_probe"]
        assert "Found via file probe" in text
        assert warnings == []


# ---------------------------------------------------------------------------
# GitHub Releases pagination
# ---------------------------------------------------------------------------


class TestGitHubReleasesPagination:
    @pytest.mark.asyncio
    async def test_fetches_all_pages_when_link_next_header_present(self) -> None:
        """When the GitHub API returns a Link: <next> header, all pages are fetched."""
        import httpx

        from migratowl.core.changelog import _fetch_from_github_releases

        page1_releases = [{"tag_name": "v3.0.0", "body": "## Breaking", "draft": False, "prerelease": False}]
        page2_releases = [{"tag_name": "v2.0.0", "body": "## Stable", "draft": False, "prerelease": False}]

        call_count = 0

        async def mock_get(url: str, **kwargs):  # type: ignore[no-untyped-def]
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First page — include Link header pointing to page 2
                resp = httpx.Response(
                    200,
                    json=page1_releases,
                    headers={"Link": '<https://api.github.com/repos/owner/repo/releases?page=2>; rel="next"'},
                    request=httpx.Request("GET", url),
                )
                return resp
            else:
                # Second page — no Link header (last page)
                return httpx.Response(
                    200,
                    json=page2_releases,
                    request=httpx.Request("GET", url),
                )

        mock_client = AsyncMock()
        mock_client.get.side_effect = mock_get
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            patch("migratowl.core.changelog.settings") as mock_settings,
        ):
            mock_settings.github_token = ""
            text = await _fetch_from_github_releases("https://github.com/owner/repo")

        assert call_count == 2, "Must make 2 requests (one per page)"
        assert "v3.0.0" in text
        assert "v2.0.0" in text

    @pytest.mark.asyncio
    async def test_single_page_no_pagination_needed(self) -> None:
        """When there is no Link: next header, only one request is made."""
        import httpx

        from migratowl.core.changelog import _fetch_from_github_releases

        releases = [{"tag_name": "v1.0.0", "body": "Initial release", "draft": False, "prerelease": False}]

        call_count = 0

        async def mock_get(url: str, **kwargs):  # type: ignore[no-untyped-def]
            nonlocal call_count
            call_count += 1
            return httpx.Response(200, json=releases, request=httpx.Request("GET", url))

        mock_client = AsyncMock()
        mock_client.get.side_effect = mock_get
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            patch("migratowl.core.changelog.settings") as mock_settings,
        ):
            mock_settings.github_token = ""
            text = await _fetch_from_github_releases("https://github.com/owner/repo")

        assert call_count == 1
        assert "v1.0.0" in text
