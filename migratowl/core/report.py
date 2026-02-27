"""Report building, rendering, and export."""

from datetime import UTC, datetime

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from migratowl.models.schemas import (
    AnalysisReport,
    ImpactAssessment,
    PatchSet,
    Severity,
)


def build_report(
    project_path: str,
    assessments: list[ImpactAssessment],
    patches: list[PatchSet],
    errors: list[str],
) -> AnalysisReport:
    """Assemble an AnalysisReport from assessments, patches, and errors."""
    critical_count = sum(1 for a in assessments if a.overall_severity == Severity.CRITICAL)

    return AnalysisReport(
        project_path=project_path,
        timestamp=datetime.now(tz=UTC).isoformat(),
        total_dependencies=len(assessments),
        outdated_count=len(assessments),
        critical_count=critical_count,
        assessments=assessments,
        patches=patches,
        errors=errors,
    )


def render_report(report: AnalysisReport, console: Console | None = None) -> None:
    """Render a report to the terminal using Rich. Never uses print()."""
    if console is None:
        console = Console()

    # Summary panel
    summary_table = Table(title="Summary", show_header=True)
    summary_table.add_column("Metric", style="bold")
    summary_table.add_column("Value", justify="right")
    summary_table.add_row("Project", report.project_path)
    summary_table.add_row("Timestamp", report.timestamp)
    summary_table.add_row("Total Dependencies", str(report.total_dependencies))
    summary_table.add_row("Outdated", str(report.outdated_count))
    summary_table.add_row("Critical", str(report.critical_count))

    console.print(Panel(summary_table, title="MigratOwl Analysis Report"))

    # Per-dependency details
    if report.assessments:
        dep_table = Table(title="Dependency Details", show_header=True)
        dep_table.add_column("Dependency", style="bold")
        dep_table.add_column("Versions")
        dep_table.add_column("Severity")
        dep_table.add_column("Impacts")
        dep_table.add_column("Summary")

        severity_styles = {
            Severity.CRITICAL: "bold red",
            Severity.WARNING: "yellow",
            Severity.INFO: "green",
        }

        for assessment in report.assessments:
            current = assessment.versions.get("current", "?")
            latest = assessment.versions.get("latest", "?")
            sev = assessment.overall_severity
            dep_table.add_row(
                assessment.dep_name,
                f"{current} -> {latest}",
                f"[{severity_styles.get(sev, '')}]{sev.value.upper()}[/]",
                str(len(assessment.impacts)),
                assessment.summary,
            )

        console.print(dep_table)

    # Errors
    if report.errors:
        console.print(Panel("\n".join(f"- {e}" for e in report.errors), title="Errors"))


def export_json(report: AnalysisReport) -> str:
    """Export report as a JSON string."""
    return report.model_dump_json(indent=2)


def export_markdown(report: AnalysisReport) -> str:
    """Export report as a formatted Markdown string."""
    lines: list[str] = []

    lines.append("# MigratOwl Analysis Report")
    lines.append("")
    lines.append(f"**Project:** {report.project_path}")
    lines.append(f"**Timestamp:** {report.timestamp}")
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total Dependencies | {report.total_dependencies} |")
    lines.append(f"| Outdated | {report.outdated_count} |")
    lines.append(f"| Critical | {report.critical_count} |")
    lines.append("")

    lines.append("## Dependency Details")
    lines.append("")

    for assessment in report.assessments:
        current = assessment.versions.get("current", "?")
        latest = assessment.versions.get("latest", "?")
        sev = assessment.overall_severity.value.upper()

        lines.append(f"### {assessment.dep_name} ({current} -> {latest})")
        lines.append("")
        lines.append(f"**Severity:** {sev}")
        lines.append(f"**Summary:** {assessment.summary}")
        lines.append("")

        if assessment.impacts:
            lines.append("| Breaking Change | Severity | Affected Files | Suggested Fix |")
            lines.append("|----------------|----------|----------------|---------------|")
            for impact in assessment.impacts:
                usages = ", ".join(impact.affected_usages) if impact.affected_usages else "N/A"
                lines.append(
                    f"| {impact.breaking_change} "
                    f"| {impact.severity.value.upper()} "
                    f"| {usages} "
                    f"| {impact.suggested_fix} |"
                )
            lines.append("")

        if assessment.warnings:
            lines.append("**Diagnostics:**")
            for w in assessment.warnings:
                lines.append(f"- {w}")
            lines.append("")

    if report.errors:
        lines.append("## Errors")
        lines.append("")
        for error in report.errors:
            lines.append(f"- {error}")
        lines.append("")

    return "\n".join(lines)
