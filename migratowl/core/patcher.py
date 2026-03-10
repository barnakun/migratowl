"""Patch generation — LLM-powered migration patches for breaking changes."""

import difflib
import logging
from pathlib import Path

from migratowl.core.llm import get_llm_semaphore, get_structured_llm
from migratowl.core.prompts import PATCH_GENERATION_PROMPT
from migratowl.models.schemas import ImpactAssessment, PatchSet, PatchSuggestion

logger = logging.getLogger(__name__)


async def generate_patches(assessments: list[ImpactAssessment], project_path: str) -> list[PatchSet]:
    """Generate migration patches for each assessment that has impacts.

    For each assessment with non-empty impacts, calls _generate_patch_for_dep
    to produce an LLM-generated PatchSet.
    """
    if not assessments:
        return []

    patches: list[PatchSet] = []
    for assessment in assessments:
        if assessment.impacts:
            try:
                patch_set = await _generate_patch_for_dep(assessment, project_path)
            except Exception:
                logger.warning(
                    "Patch generation failed for %s, returning empty patch set",
                    assessment.dep_name,
                    exc_info=True,
                )
                patch_set = PatchSet(dep_name=assessment.dep_name, patches=[], unified_diff="")
            patches.append(patch_set)
    return patches


async def _generate_patch_for_dep(assessment: ImpactAssessment, project_path: str) -> PatchSet:
    """Generate a PatchSet for a single dependency using the LLM."""
    impacts_text = _build_impacts_context(assessment, project_path)

    structured_llm = get_structured_llm(PatchSet)
    chain = PATCH_GENERATION_PROMPT | structured_llm
    async with get_llm_semaphore():
        result: PatchSet = await chain.ainvoke(
            {
                "project_path": project_path,
                "dep_name": assessment.dep_name,
                "current_version": assessment.versions.get("current", "?"),
                "latest_version": assessment.versions.get("latest", "?"),
                "impacts_text": impacts_text,
            }
        )
    result.patches = [
        p
        for p in result.patches
        if _is_code_patch(p)
        and p.original_code.strip() != p.patched_code.strip()
        and not _is_comment_only_change(p)
        and _validate_patch_against_file(p, project_path)
    ]
    diffs = []
    for p in result.patches:
        if p.original_code and p.patched_code:
            diffs.append(create_unified_diff(p.file_path, p.original_code, p.patched_code))
    result.unified_diff = "\n".join(diffs)
    return result


# Comment prefixes by file extension for _is_code_patch filtering.
_COMMENT_PREFIXES: dict[str, tuple[str, ...]] = {
    ".py": ("#",),
    ".js": ("//",),
    ".ts": ("//",),
    ".jsx": ("//",),
    ".tsx": ("//",),
}


def _strip_line_comments(line: str, prefixes: tuple[str, ...]) -> str:
    """Strip trailing comment from a line, respecting string literals.

    Walks char-by-char tracking quote state so that comment prefixes inside
    strings (e.g. '#' in "http://...") are not treated as comments.
    """
    in_quote: str | None = None
    i = 0
    while i < len(line):
        ch = line[i]
        if in_quote is not None:
            if ch == "\\" and i + 1 < len(line):
                i += 2  # skip escaped char
                continue
            if ch == in_quote:
                in_quote = None
            i += 1
            continue
        if ch in ('"', "'"):
            in_quote = ch
            i += 1
            continue
        for prefix in prefixes:
            if line[i : i + len(prefix)] == prefix:
                return line[:i]
        i += 1
    return line


def _is_comment_only_change(patch: PatchSuggestion) -> bool:
    """Return True if the only difference between original and patched is comments/blanks."""
    ext = Path(patch.file_path).suffix
    prefixes = _COMMENT_PREFIXES.get(ext, ("#",))

    def _strip_comments(code: str) -> str:
        lines = []
        for line in code.splitlines():
            stripped = _strip_line_comments(line, prefixes).rstrip()
            if stripped:
                lines.append(stripped)
        return "\n".join(lines)

    return _strip_comments(patch.original_code) == _strip_comments(patch.patched_code)


def _is_code_patch(patch: PatchSuggestion) -> bool:
    """Return True if the patch modifies actual code, not just comments/whitespace."""
    ext = Path(patch.file_path).suffix
    prefixes = _COMMENT_PREFIXES.get(ext, ("#",))

    for line in patch.original_code.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if any(stripped.startswith(p) for p in prefixes):
            continue
        return True
    return False


def _validate_patch_against_file(patch: PatchSuggestion, project_path: str = "") -> bool:
    """Return True if original_code is found in the file on disk."""
    path = Path(patch.file_path)
    if not path.is_absolute() and project_path:
        path = Path(project_path) / path
    if not path.is_file():
        return False
    content = path.read_text()
    return patch.original_code.strip() in content


def create_unified_diff(file_path: str, original: str, patched: str) -> str:
    """Create a unified diff string from original and patched content.

    Uses difflib.unified_diff with --- and +++ headers.
    Ensures trailing newlines so diff lines are never concatenated.
    """
    if original and not original.endswith("\n"):
        original += "\n"
    if patched and not patched.endswith("\n"):
        patched += "\n"

    original_lines = original.splitlines(keepends=True)
    patched_lines = patched.splitlines(keepends=True)

    diff_lines = difflib.unified_diff(
        original_lines,
        patched_lines,
        fromfile=file_path,
        tofile=file_path,
    )
    return "".join(diff_lines)


def _parse_file_line_ref(ref: str) -> tuple[str, int | None]:
    """Parse 'filepath:line' into (filepath, line_number). Returns (ref, None) if no line."""
    parts = ref.rsplit(":", 1)
    if len(parts) == 2:
        try:
            return (parts[0], int(parts[1]))
        except ValueError:
            pass
    return (ref, None)


def _read_code_context(file_path: str, line: int, context_lines: int = 5, base_path: str = "") -> str | None:
    """Read ±context_lines around a line number from a file on disk. Returns None if file missing."""
    path = Path(file_path)
    if not path.is_absolute() and base_path:
        path = Path(base_path) / path
    if not path.is_file():
        return None
    all_lines = path.read_text().splitlines()
    start = max(0, line - 1 - context_lines)
    end = min(len(all_lines), line + context_lines)
    return "\n".join(all_lines[start:end])


def _build_impacts_context(assessment: ImpactAssessment, project_path: str = "") -> str:
    """Format an ImpactAssessment into a readable context string for the LLM."""
    lines: list[str] = []
    lines.append("## Impact Assessment")
    lines.append(f"**Summary:** {assessment.summary}")
    lines.append(f"**Overall Severity:** {assessment.overall_severity.value}")
    lines.append("")

    for impact in assessment.impacts:
        lines.append(f"### {impact.breaking_change}")
        lines.append(f"- Severity: {impact.severity.value}")
        lines.append(f"- Explanation: {impact.explanation}")
        lines.append(f"- Suggested fix: {impact.suggested_fix}")
        if impact.affected_usages:
            lines.append("- Affected code:")
            for usage in impact.affected_usages:
                file_path, line_num = _parse_file_line_ref(usage)
                snippet = None
                if line_num is not None:
                    snippet = _read_code_context(file_path, line_num, base_path=project_path)
                if snippet:
                    lines.append(f"  **{usage}:**")
                    lines.append("  ```")
                    for snippet_line in snippet.splitlines():
                        lines.append(f"  {snippet_line}")
                    lines.append("  ```")
                else:
                    lines.append(f"  {usage}")
        lines.append("")

    return "\n".join(lines)
