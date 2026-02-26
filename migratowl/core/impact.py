"""Impact assessment — cross-references breaking changes with code usages."""

from migratowl.config import active_model
from migratowl.core.llm import client
from migratowl.models.schemas import (
    BreakingChange,
    CodeUsage,
    ImpactAssessment,
    Severity,
)


async def assess_impact(
    dep_name: str,
    current_version: str,
    latest_version: str,
    breaking_changes: list[BreakingChange],
    code_usages: list[CodeUsage],
) -> ImpactAssessment:
    """Assess the impact of breaking changes on project code.

    Returns INFO severity early if there are no breaking changes or no code usages.
    Otherwise, calls the LLM to produce a detailed impact assessment.
    """
    if not breaking_changes or not code_usages:
        return ImpactAssessment(
            dep_name=dep_name,
            versions={"current": current_version, "latest": latest_version},
            impacts=[],
            summary=f"No impact detected for {dep_name}",
            overall_severity=Severity.INFO,
        )

    context = _build_impact_context(breaking_changes, code_usages)

    result: ImpactAssessment = await client.chat.completions.create(
        model=active_model(),
        response_model=ImpactAssessment,
        max_retries=2,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a dependency migration expert. Analyze the breaking changes and code usages below to assess the impact on the project. Return a structured impact assessment."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Dependency: {dep_name}\n"
                    f"Current version: {current_version}\n"
                    f"Latest version: {latest_version}\n\n"
                    f"{context}"
                ),
            },
        ],
    )
    # Always populate versions from our own args — LLMs routinely omit this field.
    result.versions = {"current": current_version, "latest": latest_version}
    return result


def _build_impact_context(
    breaking_changes: list[BreakingChange],
    code_usages: list[CodeUsage],
) -> str:
    """Format breaking changes and code usages into a readable context string."""
    lines: list[str] = []

    lines.append("## Breaking Changes")
    for bc in breaking_changes:
        lines.append(f"- **{bc.api_name}** ({bc.change_type.value}): {bc.description}")
        lines.append(f"  Migration hint: {bc.migration_hint}")

    lines.append("")
    lines.append("## Code Usages")
    for usage in code_usages:
        lines.append(f"- {usage.file_path}:{usage.line_number} — {usage.usage_type} of `{usage.symbol}`")
        lines.append(f"  ```{usage.code_snippet}```")

    return "\n".join(lines)
