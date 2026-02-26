"""Patch generation — LLM-powered migration patches for breaking changes."""

import difflib

from migratowl.config import active_model
from migratowl.core.llm import client
from migratowl.models.schemas import ImpactAssessment, PatchSet


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
            patch_set = await _generate_patch_for_dep(assessment, project_path)
            patches.append(patch_set)
    return patches


async def _generate_patch_for_dep(assessment: ImpactAssessment, project_path: str) -> PatchSet:
    """Generate a PatchSet for a single dependency using the LLM."""
    impacts_text = _build_impacts_context(assessment)

    result: PatchSet = await client.chat.completions.create(
        model=active_model(),
        response_model=PatchSet,
        max_retries=2,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a code migration expert. Given the impact assessment below, generate concrete code patches that fix the breaking changes. Return a PatchSet with file-level patches showing original and patched code."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Project path: {project_path}\n"
                    f"Dependency: {assessment.dep_name}\n"
                    f"Current version: {assessment.versions.get('current', '?')}\n"
                    f"Latest version: {assessment.versions.get('latest', '?')}\n\n"
                    f"{impacts_text}"
                ),
            },
        ],
    )
    return result


def create_unified_diff(file_path: str, original: str, patched: str) -> str:
    """Create a unified diff string from original and patched content.

    Uses difflib.unified_diff with --- and +++ headers.
    """
    original_lines = original.splitlines(keepends=True)
    patched_lines = patched.splitlines(keepends=True)

    diff_lines = difflib.unified_diff(
        original_lines,
        patched_lines,
        fromfile=file_path,
        tofile=file_path,
    )
    return "".join(diff_lines)


def _build_impacts_context(assessment: ImpactAssessment) -> str:
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
            lines.append(f"- Affected files: {', '.join(impact.affected_usages)}")
        lines.append("")

    return "\n".join(lines)
