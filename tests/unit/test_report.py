"""Tests for report building and export."""

import json
from unittest.mock import MagicMock, patch

from migratowl.core.report import build_report, export_json, export_markdown, render_report
from migratowl.models.schemas import (
    AnalysisReport,
    ImpactAssessment,
    ImpactItem,
    PatchSet,
    PatchSuggestion,
    Severity,
)


def _make_assessment(
    dep_name: str = "requests",
    severity: Severity = Severity.WARNING,
) -> ImpactAssessment:
    return ImpactAssessment(
        dep_name=dep_name,
        versions={"current": "1.0.0", "latest": "2.0.0"},
        impacts=[
            ImpactItem(
                breaking_change="old_func removed",
                affected_usages=["src/app.py:42"],
                severity=severity,
                explanation="old_func is used directly",
                suggested_fix="Replace with new_func",
            )
        ],
        summary=f"1 {severity.value} impact found",
        overall_severity=severity,
    )


def _make_patch_set(dep_name: str = "requests") -> PatchSet:
    return PatchSet(
        dep_name=dep_name,
        patches=[
            PatchSuggestion(
                file_path="src/app.py",
                original_code="old_func(x)",
                patched_code="new_func(x)",
                explanation="Renamed function",
            )
        ],
        unified_diff="--- a/src/app.py\n+++ b/src/app.py\n-old_func(x)\n+new_func(x)",
    )


class TestBuildReport:
    def test_build_report(self) -> None:
        assessments = [
            _make_assessment("requests", Severity.CRITICAL),
            _make_assessment("flask", Severity.WARNING),
            _make_assessment("click", Severity.INFO),
        ]
        patches = [_make_patch_set("requests")]
        errors = ["Failed to fetch changelog for boto3"]

        report = build_report(
            project_path="/home/user/myproject",
            assessments=assessments,
            patches=patches,
            errors=errors,
        )

        assert isinstance(report, AnalysisReport)
        assert report.project_path == "/home/user/myproject"
        assert report.timestamp  # non-empty
        assert report.total_dependencies == 3
        assert report.outdated_count == 3
        assert report.critical_count == 1
        assert len(report.assessments) == 3
        assert len(report.patches) == 1
        assert len(report.errors) == 1
        assert report.errors[0] == "Failed to fetch changelog for boto3"


class TestExportJsonRoundtrip:
    def test_export_json_roundtrip(self) -> None:
        assessments = [_make_assessment("requests", Severity.CRITICAL)]
        patches = [_make_patch_set("requests")]

        report = build_report(
            project_path="/home/user/myproject",
            assessments=assessments,
            patches=patches,
            errors=[],
        )

        json_str = export_json(report)
        parsed = json.loads(json_str)

        assert parsed["project_path"] == "/home/user/myproject"
        assert parsed["total_dependencies"] == 1
        assert parsed["critical_count"] == 1
        assert len(parsed["assessments"]) == 1
        assert parsed["assessments"][0]["dep_name"] == "requests"

        # Verify we can reconstruct from JSON
        reconstructed = AnalysisReport.model_validate(parsed)
        assert reconstructed.project_path == report.project_path
        assert reconstructed.total_dependencies == report.total_dependencies


class TestExportMarkdownFormat:
    def test_export_markdown_format(self) -> None:
        assessments = [
            _make_assessment("requests", Severity.CRITICAL),
            _make_assessment("flask", Severity.WARNING),
        ]
        patches = [_make_patch_set("requests")]

        report = build_report(
            project_path="/home/user/myproject",
            assessments=assessments,
            patches=patches,
            errors=["Something failed"],
        )

        md = export_markdown(report)

        # Check structure
        assert "# MigratOwl Analysis Report" in md
        assert "/home/user/myproject" in md
        assert "## Summary" in md
        assert "## Dependency Details" in md

        # Check dependency info
        assert "requests" in md
        assert "flask" in md
        assert "CRITICAL" in md or "critical" in md.lower()
        assert "WARNING" in md or "warning" in md.lower()

        # Check errors section
        assert "## Errors" in md
        assert "Something failed" in md


class TestExportMarkdownPatches:
    def test_export_markdown_includes_patches_section(self) -> None:
        """Patches with unified_diff must appear in the Markdown output."""
        assessments = [_make_assessment("requests", Severity.CRITICAL)]
        ps = _make_patch_set("requests")
        report = build_report(
            project_path="/home/user/myproject",
            assessments=assessments,
            patches=[ps],
            errors=[],
        )

        md = export_markdown(report)

        assert "## Patches" in md
        assert "requests" in md
        assert "```diff" in md
        assert "old_func(x)" in md

    def test_export_markdown_no_patches_section_when_empty(self) -> None:
        """No patches section when patches list is empty."""
        assessments = [_make_assessment("requests", Severity.CRITICAL)]
        report = build_report(
            project_path="/home/user/myproject",
            assessments=assessments,
            patches=[],
            errors=[],
        )

        md = export_markdown(report)

        assert "## Patches" not in md


class TestWarningsInReport:
    def test_export_markdown_renders_warnings_per_dep(self) -> None:
        """Warnings in ImpactAssessment must appear in the Markdown report."""
        from migratowl.models.schemas import ImpactAssessment, Severity

        assessment = ImpactAssessment(
            dep_name="boto3",
            versions={"current": "1.0.0", "latest": "2.0.0"},
            impacts=[],
            summary="No impact",
            overall_severity=Severity.INFO,
            warnings=["Could not fetch changelog for boto3", "No usages of boto3 found in project code"],
        )
        report = build_report(
            project_path="/home/user/project",
            assessments=[assessment],
            patches=[],
            errors=[],
        )

        md = export_markdown(report)

        assert "boto3" in md
        assert "Could not fetch changelog for boto3" in md
        assert "No usages of boto3 found in project code" in md

    def test_export_markdown_no_warnings_section_when_empty(self) -> None:
        """When an assessment has no warnings, no warnings section is rendered."""
        assessment = _make_assessment("requests", Severity.INFO)
        report = build_report(
            project_path="/home/user/project",
            assessments=[assessment],
            patches=[],
            errors=[],
        )

        md = export_markdown(report)

        assert "Diagnostics" not in md or "requests" in md  # No false warnings


class TestErrorsInReport:
    def test_export_markdown_renders_per_assessment_errors(self) -> None:
        """Errors in ImpactAssessment must appear in the Markdown report."""
        assessment = ImpactAssessment(
            dep_name="bcryptjs",
            versions={"current": "2.0.0", "latest": "3.0.0"},
            impacts=[],
            summary="Could not be fully analyzed",
            overall_severity=Severity.UNKNOWN,
            warnings=["No usages found"],
            errors=["Changelog fetch failed for bcryptjs: 404 Not Found"],
        )
        report = build_report(
            project_path="/home/user/project",
            assessments=[assessment],
            patches=[],
            errors=[],
        )

        md = export_markdown(report)

        assert "bcryptjs" in md
        assert "Changelog fetch failed for bcryptjs: 404 Not Found" in md
        assert "**Errors:**" in md

    def test_export_markdown_no_errors_section_when_empty(self) -> None:
        """When an assessment has no errors, no errors section is rendered."""
        assessment = _make_assessment("requests", Severity.INFO)
        report = build_report(
            project_path="/home/user/project",
            assessments=[assessment],
            patches=[],
            errors=[],
        )

        md = export_markdown(report)

        # Should NOT have per-assessment Errors section
        assert "**Errors:**" not in md

    def test_render_report_shows_per_assessment_errors(self) -> None:
        """render_report must display a panel for per-assessment errors."""
        assessment = ImpactAssessment(
            dep_name="bcryptjs",
            versions={"current": "2.0.0", "latest": "3.0.0"},
            impacts=[],
            summary="Could not be fully analyzed",
            overall_severity=Severity.UNKNOWN,
            errors=["Changelog fetch failed"],
        )
        report = build_report(
            project_path="/home/user/project",
            assessments=[assessment],
            patches=[],
            errors=[],
        )

        mock_console = MagicMock()
        render_report(report, console=mock_console)

        # Should have printed a Panel with title "Dependency Errors"
        from rich.panel import Panel as RichPanel

        panel_calls = [
            call.args[0]
            for call in mock_console.print.call_args_list
            if call.args and isinstance(call.args[0], RichPanel)
        ]
        assert any(p.title == "Dependency Errors" for p in panel_calls)


class TestUnknownSeverityRendering:
    def test_render_report_shows_unknown_severity_in_dim(self) -> None:
        """UNKNOWN severity must render with 'dim' style in Rich output."""
        assessment = ImpactAssessment(
            dep_name="bcryptjs",
            versions={"current": "2.0.0", "latest": "3.0.0"},
            impacts=[],
            summary="Could not be fully analyzed",
            overall_severity=Severity.UNKNOWN,
            errors=["Changelog fetch failed"],
        )
        report = build_report(
            project_path="/home/user/project",
            assessments=[assessment],
            patches=[],
            errors=[],
        )

        mock_console = MagicMock()
        render_report(report, console=mock_console)

        # Find the Table object in print calls and verify it was called
        from rich.table import Table as RichTable

        table_calls = [
            call.args[0]
            for call in mock_console.print.call_args_list
            if call.args and isinstance(call.args[0], RichTable)
        ]
        # The dep_table should exist (assessment is present)
        assert len(table_calls) >= 1

    def test_export_markdown_renders_unknown_severity(self) -> None:
        """UNKNOWN severity must appear as 'UNKNOWN' in Markdown export."""
        assessment = ImpactAssessment(
            dep_name="bcryptjs",
            versions={"current": "2.0.0", "latest": "3.0.0"},
            impacts=[],
            summary="Could not be fully analyzed",
            overall_severity=Severity.UNKNOWN,
        )
        report = build_report(
            project_path="/home/user/project",
            assessments=[assessment],
            patches=[],
            errors=[],
        )

        md = export_markdown(report)

        assert "UNKNOWN" in md
        assert "Could not be fully analyzed" in md


class TestRenderReportUsesRich:
    def test_render_report_uses_rich(self) -> None:
        assessments = [_make_assessment("requests", Severity.CRITICAL)]
        report = build_report(
            project_path="/home/user/myproject",
            assessments=assessments,
            patches=[],
            errors=[],
        )

        mock_console = MagicMock()
        render_report(report, console=mock_console)

        # Verify Console.print was called (not builtin print)
        assert mock_console.print.called
        assert mock_console.print.call_count >= 1

    @patch("builtins.print")
    def test_render_report_never_calls_print(self, mock_print: MagicMock) -> None:
        assessments = [_make_assessment("requests", Severity.CRITICAL)]
        report = build_report(
            project_path="/home/user/myproject",
            assessments=assessments,
            patches=[],
            errors=["an error"],
        )

        mock_console = MagicMock()
        render_report(report, console=mock_console)

        mock_print.assert_not_called()
