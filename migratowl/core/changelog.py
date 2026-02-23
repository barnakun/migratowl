"""Changelog fetching and chunking for dependency analysis."""

from __future__ import annotations

import re

import html2text as _html2text
import httpx
from packaging.version import InvalidVersion, Version


async def fetch_changelog(
    changelog_url: str | None,
    repository_url: str | None,
    dep_name: str,
) -> tuple[str, list[str]]:
    """Fetch changelog text, trying changelog_url first, then GitHub raw fallback.

    Returns (text, warnings) where warnings is a list of diagnostic messages
    explaining why the changelog could not be fetched (empty on success).
    """
    if not changelog_url and not repository_url:
        return "", [f"No changelog URL or repository URL provided for {dep_name}"]

    if changelog_url:
        try:
            return await _fetch_from_url(changelog_url), []
        except Exception:
            pass

    if repository_url:
        try:
            return await _fetch_from_github(repository_url), []
        except Exception:
            pass

    return "", [f"Could not fetch changelog for {dep_name}"]


async def _fetch_from_url(url: str) -> str:
    """Fetch raw text from a URL with redirect following.

    If the response is HTML, strips it to plain text with html2text and checks
    for parseable version headers.  Raises ValueError if no version headers are
    found after stripping (triggers the GitHub raw-file fallback).
    """
    async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
        response = await client.get(url)
        response.raise_for_status()
        text = response.text
        if text.lstrip().startswith(("<", "<!DOCTYPE", "<!doctype")):
            converter = _html2text.HTML2Text()
            converter.ignore_links = True
            converter.ignore_images = True
            converter.body_width = 0
            stripped = converter.handle(text)
            if not chunk_changelog_by_version(stripped):
                raise ValueError(f"HTML response with no parseable version headers: {url}")
            return stripped
        return text


# Regex to find a GitHub blob URL embedded in stub/redirect files.
_GITHUB_BLOB_RE = re.compile(r"https?://github\.com/([^/\s]+)/([^/\s]+)/blob/([^/\s]+)/([^\s`>\"']+)")

# Changelog filenames tried at the root and inside every subdirectory.
_CHANGELOG_FILENAMES: list[str] = [
    "CHANGELOG.md",
    "CHANGELOG.rst",
    "CHANGES.md",
    "CHANGES.rst",
    "HISTORY.md",
    "HISTORY.rst",
    "NEWS.md",
    "NEWS.rst",
    "changelog.md",
    "changelog.rst",
    "changes.md",
    "changes.rst",
]

# Root-level filenames tried first (covers the vast majority of packages).
_ROOT_FILENAMES = _CHANGELOG_FILENAMES

# Subdirectory prefixes searched after all root files fail or are stubs.
_SUBDIRECTORY_ROOTS: list[str] = [
    "docs/",
    "doc/",
    "doc/en/",
    "docs/en/",
]

# Doc-subdirectory paths: Cartesian product of roots × filenames.
_DOC_FILENAMES: list[str] = [f"{subdir}{name}" for subdir in _SUBDIRECTORY_ROOTS for name in _CHANGELOG_FILENAMES]


async def _fetch_from_github(repository_url: str) -> str:
    """Try common changelog filenames on raw.githubusercontent.com.

    Strategy:
    1. Try root-level filenames on main then master.
    2. If a file returns 200 but has no version headers it is a stub —
       scan it for a GitHub blob URL and follow that URL directly.
    3. If all root files fail, retry with doc-subdirectory paths.
    """
    match = re.search(r"github\.com[/:]([^/]+)/([^/.]+)", repository_url)
    if not match:
        raise ValueError(f"Cannot parse GitHub URL: {repository_url}")

    owner, repo = match.group(1), match.group(2)
    branches = ["main", "master"]

    async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
        for filenames_group in (_ROOT_FILENAMES, _DOC_FILENAMES):
            for branch in branches:
                for filename in filenames_group:
                    url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{filename}"
                    try:
                        response = await client.get(url)
                        if response.status_code != 200:
                            continue
                        text = response.text
                        if chunk_changelog_by_version(text):
                            return text
                        # Stub: look for a GitHub blob URL and follow it once.
                        m = _GITHUB_BLOB_RE.search(text)
                        if m:
                            raw_url = (
                                f"https://raw.githubusercontent.com/{m.group(1)}/{m.group(2)}/{m.group(3)}/{m.group(4)}"
                            )
                            try:
                                r2 = await client.get(raw_url)
                                if r2.status_code == 200 and chunk_changelog_by_version(r2.text):
                                    return r2.text
                            except Exception:
                                pass
                    except Exception:
                        continue

    raise FileNotFoundError(f"No changelog found for {owner}/{repo}")


# Matches a bare version number at the start of a cleaned string: 1.2.3 or 1.2
_VERSION_RE = re.compile(r"^(\d+\.\d+(?:\.\d+)?)")


def _parse_version_from_line(line: str) -> str | None:
    """Signal A+C: extract version if the line's primary purpose is naming a version.

    Strips formatting markup (##, **, [], optional single-word prefix, v-prefix),
    then checks that what remains is a version number with at most a brief suffix
    (date, dash, parenthesised date).  Long content after the version → returns None.
    """
    s = line.strip()
    # Strip markdown heading markers
    s = re.sub(r"^#{1,6}\s*", "", s).strip()
    # Strip leading/trailing bold markers
    s = re.sub(r"^\*{1,2}", "", s).strip()
    s = re.sub(r"\*{1,2}$", "", s).strip()
    # Strip leading bracket (keep closing bracket for now)
    s = re.sub(r"^\[", "", s).strip()

    # Optional single-word prefix: "Release", "Version", etc. (1–30 alpha chars)
    m = re.match(r"^([A-Za-z]\w{0,29})\s+(.*)", s)
    if m:
        s = m.group(2).strip()

    # Strip leading v/V
    if len(s) > 1 and s[0] in ("v", "V") and s[1].isdigit():
        s = s[1:]

    # Strip trailing bracket (from [3.0.0] style)
    s = re.sub(r"^\[?", "", s).strip()
    s = re.sub(r"]?", "", s, count=1).strip()

    m = _VERSION_RE.match(s)
    if not m:
        return None

    version = m.group(1)
    remainder = s[m.end() :].strip()

    # Allow: nothing, ], closing **, optional "- YYYY-MM-DD" or "(YYYY-MM-DD)"
    remainder = re.sub(r"^[]* ]+", "", remainder).strip()
    remainder = re.sub(r"^[-–]\s*\d{4}[\d\-]*\s*", "", remainder).strip()
    remainder = re.sub(r"^\(\d{4}[\d\-]*\)\s*", "", remainder).strip()

    # If more than two words of content remain, this line is not a version header
    if len(remainder.split()) > 2:
        return None

    return version


def _is_header_position(i: int, lines: list[str]) -> bool:
    """Signal B: True if line i carries header-level structural markup.

    Accepts:
    - Markdown ATX heading  (## …)
    - Bold-wrapped line     (**Release …**)
    - RST setext underline  (next line is ---/=== of sufficient length)
    - Bare version preceded by a blank line (or at start of file)
    """
    raw = lines[i]
    stripped = raw.strip()

    # ATX heading
    if re.match(r"^#{1,6}\s", raw):
        return True

    # Bold wrapper: starts with ** (but not *** which is a HR, and not * list item)
    if re.match(r"^\*{1,2}[^*\s]", stripped):
        return True

    # RST setext underline: next non-empty line is ---/=== of length ≥ 3
    if i + 1 < len(lines):
        next_line = lines[i + 1].strip()
        if re.fullmatch(r"[-=]{3,}", next_line):
            return True

    # Bare version number (possibly with date) at start of file or after blank line
    bare = stripped
    bare = re.sub(r"^v", "", bare)
    bare = re.sub(r"\s*\([\d\-]+\)\s*$", "", bare).strip()
    bare = re.sub(r"\s*[-–]\s*[\d\-]+\s*$", "", bare).strip()
    if re.fullmatch(r"\d+\.\d+(?:\.\d+)?", bare):
        if i == 0 or lines[i - 1].strip() == "":
            return True

    return False


def chunk_changelog_by_version(text: str) -> list[dict]:
    """Split changelog text into per-version chunks.

    Uses a three-signal structural approach that handles all common formats:
    - Markdown ATX headings: ## v1.0.0, ## [3.0.0], ## Release 4.1.0 - 2024-10-12
    - Bold headers: **Release 4.0.6** - 2024-03-09
    - RST setext: 2.32.5 (2025-08-18)\\n---, Version 3.1.0\\n---
    - Bare version: 1.0.0 (preceded by blank line)

    Each chunk: {"version": "2.0.0", "content": "..."}
    """
    if not text.strip():
        return []

    lines = text.splitlines()
    # Reconstruct line start offsets for slicing the original text
    offsets: list[int] = []
    pos = 0
    for line in lines:
        offsets.append(pos)
        pos += len(line) + 1  # +1 for the newline

    header_positions: list[tuple[int, str, int]] = []  # (line_index, version, char_offset)
    for i, line in enumerate(lines):
        version = _parse_version_from_line(line)
        if version and _is_header_position(i, lines):
            header_positions.append((i, version, offsets[i]))

    if not header_positions:
        return []

    chunks = []
    for idx, (line_i, version, char_start) in enumerate(header_positions):
        # Content starts after this header line (and the RST underline if present)
        content_line = line_i + 1
        # Skip RST underline
        if content_line < len(lines) and re.fullmatch(r"[-=]{3,}", lines[content_line].strip()):
            content_line += 1
        content_start = offsets[content_line] if content_line < len(lines) else len(text)

        if idx + 1 < len(header_positions):
            content_end = header_positions[idx + 1][2]
        else:
            content_end = len(text)

        content = text[content_start:content_end].strip()
        chunks.append({"version": version, "content": content})

    return chunks


def _parse_version(v: str) -> Version:
    """Parse a version string using packaging.version."""
    return Version(v)


def filter_chunks_by_version_range(
    chunks: list[dict],
    current_version: str,
    latest_version: str,
) -> list[dict]:
    """Return chunks with versions > current and <= latest."""
    if not chunks:
        return []

    try:
        current = _parse_version(current_version)
        latest = _parse_version(latest_version)
    except InvalidVersion:
        # Fallback: simple tuple comparison
        current = tuple(int(x) for x in current_version.split("."))  # type: ignore[assignment]
        latest = tuple(int(x) for x in latest_version.split("."))  # type: ignore[assignment]

    filtered = []
    for chunk in chunks:
        try:
            v = _parse_version(chunk["version"])
        except InvalidVersion:
            try:
                v = tuple(int(x) for x in chunk["version"].split("."))  # type: ignore[assignment]
            except (ValueError, AttributeError):
                continue

        if current < v <= latest:  # type: ignore[operator]
            filtered.append(chunk)

    return filtered
