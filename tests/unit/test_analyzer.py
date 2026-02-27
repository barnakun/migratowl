"""Tests for LangGraph orchestration (analyzer.py)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest
from langgraph.types import Command, Send

from migratowl.models.schemas import (
    AnalysisReport,
    BreakingChange,
    ChangeType,
    CodeUsage,
    Dependency,
    Ecosystem,
    ImpactAssessment,
    OutdatedDependency,
    RAGQueryResult,
    Severity,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_parent_state(**overrides) -> dict:
    """Create a minimal AnalysisState dict for testing parent graph nodes."""
    state: dict = {
        "project_path": "/tmp/myproject",
        "fix_mode": False,
        "dependencies": [],
        "impact_assessments": [],
        "patches": [],
        "report": "",
        "errors": [],
    }
    state.update(overrides)
    return state


def _make_dep_state(**overrides) -> dict:
    """Create a minimal DepAnalysisState dict for testing worker nodes."""
    state: dict = {
        "dep_name": "requests",
        "current_version": "2.28.0",
        "latest_version": "2.31.0",
        "project_path": "/tmp/myproject",
        "changelog_url": "",
        "repository_url": "",
        "changelog": "",
        "rag_results": [],
        "rag_confidence": 0.0,
        "retry_count": 0,
        "code_usages": [],
        "impact": {},
        "warnings": [],
    }
    state.update(overrides)
    return state


# ---------------------------------------------------------------------------
# scan_dependencies_node
# ---------------------------------------------------------------------------


class TestScanDependenciesNode:
    @pytest.mark.asyncio
    async def test_scan_dependencies_node_returns_command(self) -> None:
        from migratowl.core.analyzer import scan_dependencies_node

        mock_deps = [
            Dependency(name="requests", current_version="2.28.0", ecosystem=Ecosystem.PYTHON, manifest_path="req.txt"),
        ]
        mock_outdated = [
            OutdatedDependency(
                name="requests",
                current_version="2.28.0",
                latest_version="2.31.0",
                ecosystem=Ecosystem.PYTHON,
                manifest_path="req.txt",
            ),
        ]

        with (
            patch("migratowl.core.analyzer.scanner.scan_project", new_callable=AsyncMock, return_value=mock_deps),
            patch("migratowl.core.analyzer.registry.find_outdated", new_callable=AsyncMock, return_value=mock_outdated),
        ):
            state = _make_parent_state()
            result = await scan_dependencies_node(state)

        assert isinstance(result, Command)
        assert result.goto == "fan_out"
        assert "dependencies" in result.update
        assert len(result.update["dependencies"]) == 1
        assert result.update["dependencies"][0]["name"] == "requests"


# ---------------------------------------------------------------------------
# fan_out_deps
# ---------------------------------------------------------------------------


class TestFanOutDeps:
    def test_fan_out_deps_returns_send_objects(self) -> None:
        from migratowl.core.analyzer import fan_out_deps

        deps = [
            {
                "name": "requests",
                "current_version": "2.28.0",
                "latest_version": "2.31.0",
                "project_path": "/tmp/myproject",
                "changelog_url": "",
                "repository_url": "",
            },
            {
                "name": "flask",
                "current_version": "2.0.0",
                "latest_version": "3.0.0",
                "project_path": "/tmp/myproject",
                "changelog_url": "",
                "repository_url": "",
            },
        ]
        state = _make_parent_state(dependencies=deps)
        result = fan_out_deps(state)

        assert isinstance(result, list)
        assert len(result) == 2
        for item in result:
            assert isinstance(item, Send)
        assert result[0].node == "analyze_dep"
        assert result[1].node == "analyze_dep"
        assert result[0].arg["dep_name"] == "requests"
        assert result[1].arg["dep_name"] == "flask"

    def test_fan_out_deps_passes_changelog_urls(self) -> None:
        from migratowl.core.analyzer import fan_out_deps

        deps = [
            {
                "name": "requests",
                "current_version": "2.28.0",
                "latest_version": "2.31.0",
                "project_path": "/tmp/myproject",
                "changelog_url": "https://example.com/CHANGELOG.md",
                "repository_url": "https://github.com/psf/requests",
            },
        ]
        state = _make_parent_state(dependencies=deps)
        result = fan_out_deps(state)

        assert result[0].arg["changelog_url"] == "https://example.com/CHANGELOG.md"
        assert result[0].arg["repository_url"] == "https://github.com/psf/requests"


# ---------------------------------------------------------------------------
# rag_analyze_node
# ---------------------------------------------------------------------------


class TestRagAnalyzeNode:
    @pytest.mark.asyncio
    async def test_rag_analyze_node_low_confidence_routes_to_refine(self) -> None:
        from migratowl.core.analyzer import rag_analyze_node

        mock_rag_result = RAGQueryResult(
            breaking_changes=[],
            confidence=0.3,
            source_chunks=["chunk1"],
        )

        with patch("migratowl.core.analyzer.rag.query", new_callable=AsyncMock, return_value=mock_rag_result):
            state = _make_dep_state(
                changelog="some changelog text",
                rag_confidence=0.3,
                retry_count=0,
            )
            result = await rag_analyze_node(state)

        assert isinstance(result, Command)
        assert result.goto == "refine_query"

    @pytest.mark.asyncio
    async def test_rag_analyze_node_high_confidence_routes_to_parse_code(self) -> None:
        from migratowl.core.analyzer import rag_analyze_node

        mock_rag_result = RAGQueryResult(
            breaking_changes=[
                BreakingChange(
                    api_name="old_func",
                    change_type=ChangeType.REMOVED,
                    description="Removed",
                    migration_hint="Use new_func",
                )
            ],
            confidence=0.9,
            source_chunks=["chunk1"],
        )

        with patch("migratowl.core.analyzer.rag.query", new_callable=AsyncMock, return_value=mock_rag_result):
            state = _make_dep_state(changelog="some changelog text")
            result = await rag_analyze_node(state)

        assert isinstance(result, Command)
        assert result.goto == "parse_code"
        assert "rag_results" in result.update

    @pytest.mark.asyncio
    async def test_rag_analyze_node_max_retries_routes_to_parse_code(self) -> None:
        from migratowl.core.analyzer import rag_analyze_node

        mock_rag_result = RAGQueryResult(
            breaking_changes=[],
            confidence=0.2,
            source_chunks=[],
        )

        with patch("migratowl.core.analyzer.rag.query", new_callable=AsyncMock, return_value=mock_rag_result):
            state = _make_dep_state(
                changelog="some changelog text",
                retry_count=3,
            )
            result = await rag_analyze_node(state)

        assert isinstance(result, Command)
        assert result.goto == "parse_code"


# ---------------------------------------------------------------------------
# assess_impact_node — fan-in key
# ---------------------------------------------------------------------------


class TestAssessImpactNode:
    @pytest.mark.asyncio
    async def test_assess_impact_returns_impact_assessments_key(self) -> None:
        from migratowl.core.analyzer import assess_impact_node

        mock_assessment = ImpactAssessment(
            dep_name="requests",
            versions={"current": "2.28.0", "latest": "2.31.0"},
            impacts=[],
            summary="No impact",
            overall_severity=Severity.INFO,
        )

        with patch(
            "migratowl.core.analyzer.impact.assess_impact", new_callable=AsyncMock, return_value=mock_assessment
        ):
            state = _make_dep_state(
                rag_results=[],
                code_usages=[],
            )
            result = await assess_impact_node(state)

        assert isinstance(result, Command)
        assert "impact_assessments" in result.update
        assert isinstance(result.update["impact_assessments"], list)
        assert len(result.update["impact_assessments"]) == 1


# ---------------------------------------------------------------------------
# fetch_changelog_node — URL passthrough
# ---------------------------------------------------------------------------


class TestFetchChangelogNode:
    @pytest.mark.asyncio
    async def test_fetch_changelog_passes_urls(self) -> None:
        from migratowl.core.analyzer import fetch_changelog_node

        with patch(
            "migratowl.core.analyzer.changelog.fetch_changelog",
            new_callable=AsyncMock,
            return_value=("changelog text", []),
        ) as mock_fetch:
            state = _make_dep_state(
                changelog_url="https://example.com/CHANGELOG.md",
                repository_url="https://github.com/psf/requests",
            )
            result = await fetch_changelog_node(state)

        mock_fetch.assert_called_once_with(
            changelog_url="https://example.com/CHANGELOG.md",
            repository_url="https://github.com/psf/requests",
            dep_name="requests",
        )
        assert result.update["changelog"] == "changelog text"

    @pytest.mark.asyncio
    async def test_fetch_changelog_empty_url_passes_none(self) -> None:
        from migratowl.core.analyzer import fetch_changelog_node

        with patch(
            "migratowl.core.analyzer.changelog.fetch_changelog",
            new_callable=AsyncMock,
            return_value=("", []),
        ) as mock_fetch:
            state = _make_dep_state(changelog_url="", repository_url="")
            await fetch_changelog_node(state)

        mock_fetch.assert_called_once_with(
            changelog_url=None,
            repository_url=None,
            dep_name="requests",
        )

    @pytest.mark.asyncio
    async def test_fetch_changelog_node_includes_warnings_in_state(self) -> None:
        """When fetch_changelog returns warnings, they are stored in state update."""
        from migratowl.core.analyzer import fetch_changelog_node

        with patch(
            "migratowl.core.analyzer.changelog.fetch_changelog",
            new_callable=AsyncMock,
            return_value=("", ["No changelog URL or repository URL provided for requests"]),
        ):
            state = _make_dep_state(changelog_url="", repository_url="")
            result = await fetch_changelog_node(state)

        assert "warnings" in result.update
        assert len(result.update["warnings"]) == 1
        assert "requests" in result.update["warnings"][0]

    @pytest.mark.asyncio
    async def test_fetch_changelog_node_no_warnings_on_success(self) -> None:
        """When fetch_changelog succeeds, warnings list is empty."""
        from migratowl.core.analyzer import fetch_changelog_node

        with patch(
            "migratowl.core.analyzer.changelog.fetch_changelog",
            new_callable=AsyncMock,
            return_value=("## 1.0.0\n- change", []),
        ):
            state = _make_dep_state()
            result = await fetch_changelog_node(state)

        assert result.update.get("warnings", []) == []


# ---------------------------------------------------------------------------
# embed_changelog_node — version-range pre-filtering
# ---------------------------------------------------------------------------


class TestEmbedChangelogNode:
    @pytest.mark.asyncio
    async def test_embed_changelog_node_filters_to_version_range(self) -> None:
        """embed_changelog_node must only embed chunks between current and latest version."""
        from migratowl.core.analyzer import embed_changelog_node

        all_chunks = [
            {"version": "1.0.0", "content": "old"},
            {"version": "2.0.0", "content": "relevant"},
            {"version": "3.0.0", "content": "latest"},
        ]
        with (
            patch(
                "migratowl.core.analyzer.changelog.chunk_changelog_by_version",
                return_value=all_chunks,
            ),
            patch(
                "migratowl.core.analyzer.rag.embed_changelog",
                new_callable=AsyncMock,
            ) as mock_embed,
        ):
            state = _make_dep_state(
                changelog="## 1.0.0\nold\n## 2.0.0\nrelevant\n## 3.0.0\nlatest",
                current_version="1.0.0",
                latest_version="3.0.0",
            )
            await embed_changelog_node(state)

        embedded_chunks = mock_embed.call_args.args[1]
        versions = [c["version"] for c in embedded_chunks]
        assert "1.0.0" not in versions  # excluded: == current
        assert "2.0.0" in versions  # included: current < v <= latest
        assert "3.0.0" in versions  # included: == latest


# ---------------------------------------------------------------------------
# Warning propagation tests
# ---------------------------------------------------------------------------


class TestWarningPropagation:
    @pytest.mark.asyncio
    async def test_embed_changelog_warns_when_no_parseable_chunks(self) -> None:
        """When changelog text produces no parseable version chunks, a warning is emitted."""
        from migratowl.core.analyzer import embed_changelog_node

        with (
            patch(
                "migratowl.core.analyzer.changelog.chunk_changelog_by_version",
                return_value=[],
            ),
            patch("migratowl.core.analyzer.rag.embed_changelog", new_callable=AsyncMock),
        ):
            state = _make_dep_state(
                changelog="no version headers here",
                current_version="1.0.0",
                latest_version="2.0.0",
            )
            result = await embed_changelog_node(state)

        assert "warnings" in result.update
        assert len(result.update["warnings"]) > 0
        assert "requests" in result.update["warnings"][0]

    @pytest.mark.asyncio
    async def test_embed_changelog_warns_when_no_chunks_in_range(self) -> None:
        """When chunks exist but none fall in the version range, a warning is emitted."""
        from migratowl.core.analyzer import embed_changelog_node

        all_chunks = [{"version": "0.9.0", "content": "old stuff"}]

        with (
            patch(
                "migratowl.core.analyzer.changelog.chunk_changelog_by_version",
                return_value=all_chunks,
            ),
            patch("migratowl.core.analyzer.rag.embed_changelog", new_callable=AsyncMock),
        ):
            state = _make_dep_state(
                changelog="## 0.9.0\nold stuff",
                current_version="1.0.0",
                latest_version="2.0.0",
            )
            result = await embed_changelog_node(state)

        assert "warnings" in result.update
        assert len(result.update["warnings"]) > 0

    @pytest.mark.asyncio
    async def test_embed_changelog_no_warnings_on_success(self) -> None:
        """When chunks are found in range, no warnings are emitted."""
        from migratowl.core.analyzer import embed_changelog_node

        all_chunks = [{"version": "1.5.0", "content": "something"}, {"version": "2.0.0", "content": "latest"}]

        with (
            patch(
                "migratowl.core.analyzer.changelog.chunk_changelog_by_version",
                return_value=all_chunks,
            ),
            patch("migratowl.core.analyzer.rag.embed_changelog", new_callable=AsyncMock),
        ):
            state = _make_dep_state(
                changelog="## 1.5.0\nsomething\n## 2.0.0\nlatest",
                current_version="1.0.0",
                latest_version="2.0.0",
            )
            result = await embed_changelog_node(state)

        assert result.update.get("warnings", []) == []

    @pytest.mark.asyncio
    async def test_assess_impact_attaches_state_warnings_to_assessment(self) -> None:
        """Warnings accumulated in state are attached to the ImpactAssessment."""
        from migratowl.core.analyzer import assess_impact_node

        mock_assessment = ImpactAssessment(
            dep_name="requests",
            versions={"current": "2.28.0", "latest": "2.31.0"},
            impacts=[],
            summary="No impact",
            overall_severity=Severity.INFO,
        )

        with patch(
            "migratowl.core.analyzer.impact.assess_impact", new_callable=AsyncMock, return_value=mock_assessment
        ):
            state = _make_dep_state(
                rag_results=[],
                code_usages=[
                    {
                        "file_path": "a.py",
                        "line_number": 1,
                        "usage_type": "import",
                        "symbol": "requests",
                        "code_snippet": "import requests",
                    },
                ],
                warnings=["No changelog found for requests"],
            )
            result = await assess_impact_node(state)

        assessment_dict = result.update["impact_assessments"][0]
        assert "warnings" in assessment_dict
        assert "No changelog found for requests" in assessment_dict["warnings"]

    @pytest.mark.asyncio
    async def test_assess_impact_warns_when_no_code_usages(self) -> None:
        """When no code usages are found, a diagnostic warning is emitted."""
        from migratowl.core.analyzer import assess_impact_node

        mock_assessment = ImpactAssessment(
            dep_name="requests",
            versions={"current": "2.28.0", "latest": "2.31.0"},
            impacts=[],
            summary="No impact",
            overall_severity=Severity.INFO,
        )

        with patch(
            "migratowl.core.analyzer.impact.assess_impact", new_callable=AsyncMock, return_value=mock_assessment
        ):
            state = _make_dep_state(
                rag_results=[],
                code_usages=[],
                warnings=[],
            )
            result = await assess_impact_node(state)

        assessment_dict = result.update["impact_assessments"][0]
        assert "warnings" in assessment_dict
        assert any("usage" in w.lower() or "requests" in w for w in assessment_dict["warnings"])


# ---------------------------------------------------------------------------
# scan_dependencies_node — URL inclusion
# ---------------------------------------------------------------------------


class TestScanDependenciesNodeUrls:
    @pytest.mark.asyncio
    async def test_scan_includes_changelog_urls(self) -> None:
        from migratowl.core.analyzer import scan_dependencies_node

        mock_deps = [
            Dependency(name="requests", current_version="2.28.0", ecosystem=Ecosystem.PYTHON, manifest_path="req.txt"),
        ]
        mock_outdated = [
            OutdatedDependency(
                name="requests",
                current_version="2.28.0",
                latest_version="2.31.0",
                ecosystem=Ecosystem.PYTHON,
                manifest_path="req.txt",
                changelog_url="https://example.com/CHANGELOG.md",
                repository_url="https://github.com/psf/requests",
            ),
        ]

        with (
            patch("migratowl.core.analyzer.scanner.scan_project", new_callable=AsyncMock, return_value=mock_deps),
            patch("migratowl.core.analyzer.registry.find_outdated", new_callable=AsyncMock, return_value=mock_outdated),
        ):
            state = _make_parent_state()
            result = await scan_dependencies_node(state)

        dep = result.update["dependencies"][0]
        assert dep["changelog_url"] == "https://example.com/CHANGELOG.md"
        assert dep["repository_url"] == "https://github.com/psf/requests"


# ---------------------------------------------------------------------------
# route_after_fan_in
# ---------------------------------------------------------------------------


class TestRouteAfterFanIn:
    def test_route_after_fan_in_fix_mode(self) -> None:
        from migratowl.core.analyzer import route_after_fan_in

        state = _make_parent_state(fix_mode=True)
        result = route_after_fan_in(state)

        assert isinstance(result, Command)
        assert result.goto == "generate_patches"

    def test_route_after_fan_in_no_fix(self) -> None:
        from migratowl.core.analyzer import route_after_fan_in

        state = _make_parent_state(fix_mode=False)
        result = route_after_fan_in(state)

        assert isinstance(result, Command)
        assert result.goto == "generate_report"


# ---------------------------------------------------------------------------
# build_analysis_graph
# ---------------------------------------------------------------------------


class TestBuildAnalysisGraph:
    def test_build_analysis_graph_returns_compiled_graph(self) -> None:
        from migratowl.core.analyzer import build_analysis_graph

        graph = build_analysis_graph()
        # CompiledGraph has an invoke method
        assert hasattr(graph, "invoke")
        assert hasattr(graph, "ainvoke")

    def test_graph_has_route_results_node(self) -> None:
        from migratowl.core.analyzer import build_analysis_graph

        graph = build_analysis_graph()
        node_names = set(graph.get_graph().nodes)
        assert "route_results" in node_names
        assert "fan_out" in node_names
        assert "analyze_dep" in node_names


# ---------------------------------------------------------------------------
# analyze (end-to-end with all mocks)
# ---------------------------------------------------------------------------


class TestAnalyze:
    @pytest.mark.asyncio
    async def test_analyze_returns_json_string(self) -> None:
        from migratowl.core.analyzer import analyze

        mock_deps = [
            Dependency(name="requests", current_version="2.28.0", ecosystem=Ecosystem.PYTHON, manifest_path="req.txt"),
        ]
        mock_outdated = [
            OutdatedDependency(
                name="requests",
                current_version="2.28.0",
                latest_version="2.31.0",
                ecosystem=Ecosystem.PYTHON,
                manifest_path="req.txt",
            ),
        ]
        mock_rag_result = RAGQueryResult(
            breaking_changes=[
                BreakingChange(
                    api_name="old_func",
                    change_type=ChangeType.REMOVED,
                    description="Removed old_func",
                    migration_hint="Use new_func",
                )
            ],
            confidence=0.9,
            source_chunks=["chunk1"],
        )
        mock_usages = [
            CodeUsage(
                file_path="src/app.py",
                line_number=10,
                usage_type="import",
                symbol="requests",
                code_snippet="import requests",
            ),
        ]
        mock_impact = ImpactAssessment(
            dep_name="requests",
            versions={"current": "2.28.0", "latest": "2.31.0"},
            impacts=[],
            summary="No impact",
            overall_severity=Severity.INFO,
        )
        mock_report = AnalysisReport(
            project_path="/tmp/myproject",
            timestamp="2024-01-01T00:00:00",
            total_dependencies=1,
            outdated_count=1,
            critical_count=0,
            assessments=[mock_impact],
            patches=[],
            errors=[],
        )

        with (
            patch("migratowl.core.analyzer.scanner.scan_project", new_callable=AsyncMock, return_value=mock_deps),
            patch("migratowl.core.analyzer.registry.find_outdated", new_callable=AsyncMock, return_value=mock_outdated),
            patch(
                "migratowl.core.analyzer.changelog.fetch_changelog",
                new_callable=AsyncMock,
                return_value=("## 2.31.0\nSome changes", []),
            ),
            patch(
                "migratowl.core.analyzer.changelog.chunk_changelog_by_version",
                return_value=[{"version": "2.31.0", "content": "Some changes"}],
            ),
            patch("migratowl.core.analyzer.rag.embed_changelog", new_callable=AsyncMock),
            patch("migratowl.core.analyzer.rag.query", new_callable=AsyncMock, return_value=mock_rag_result),
            patch("migratowl.core.analyzer.code_parser.find_usages", new_callable=AsyncMock, return_value=mock_usages),
            patch("migratowl.core.analyzer.impact.assess_impact", new_callable=AsyncMock, return_value=mock_impact),
            patch("migratowl.core.analyzer.report.build_report", return_value=mock_report),
            patch("migratowl.core.analyzer.report.export_json", return_value=mock_report.model_dump_json(indent=2)),
        ):
            result = await analyze("/tmp/myproject", fix_mode=False)

        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed["project_path"] == "/tmp/myproject"

    @pytest.mark.asyncio
    async def test_analyze_populates_assessments_in_report(self) -> None:
        """assess_impact_node must write impact_assessments back to the parent state."""
        from migratowl.core.analyzer import analyze

        mock_deps = [
            Dependency(name="requests", current_version="2.28.0", ecosystem=Ecosystem.PYTHON, manifest_path="req.txt"),
        ]
        mock_outdated = [
            OutdatedDependency(
                name="requests",
                current_version="2.28.0",
                latest_version="2.31.0",
                ecosystem=Ecosystem.PYTHON,
                manifest_path="req.txt",
            ),
        ]
        mock_rag_result = RAGQueryResult(breaking_changes=[], confidence=0.9, source_chunks=[])
        mock_impact = ImpactAssessment(
            dep_name="requests",
            versions={"current": "2.28.0", "latest": "2.31.0"},
            impacts=[],
            summary="Minor update",
            overall_severity=Severity.INFO,
        )

        with (
            patch("migratowl.core.analyzer.scanner.scan_project", new_callable=AsyncMock, return_value=mock_deps),
            patch("migratowl.core.analyzer.registry.find_outdated", new_callable=AsyncMock, return_value=mock_outdated),
            patch(
                "migratowl.core.analyzer.changelog.fetch_changelog",
                new_callable=AsyncMock,
                return_value=("## 2.31.0\nSome changes", []),
            ),
            patch(
                "migratowl.core.analyzer.changelog.chunk_changelog_by_version",
                return_value=[{"version": "2.31.0", "content": "Some changes"}],
            ),
            patch("migratowl.core.analyzer.rag.embed_changelog", new_callable=AsyncMock),
            patch("migratowl.core.analyzer.rag.query", new_callable=AsyncMock, return_value=mock_rag_result),
            patch("migratowl.core.analyzer.code_parser.find_usages", new_callable=AsyncMock, return_value=[]),
            patch("migratowl.core.analyzer.impact.assess_impact", new_callable=AsyncMock, return_value=mock_impact),
        ):
            result = await analyze("/tmp/myproject", fix_mode=False)

        parsed = json.loads(result)
        assert parsed["total_dependencies"] == 1
        assert parsed["outdated_count"] == 1
        assert len(parsed["assessments"]) == 1
        assert parsed["assessments"][0]["dep_name"] == "requests"

    @pytest.mark.asyncio
    async def test_analyze_with_multiple_deps_does_not_raise_concurrent_update_error(self) -> None:
        """Parallel fan-out with 2+ deps must not raise InvalidUpdateError on project_path."""
        from migratowl.core.analyzer import analyze

        mock_deps = [
            Dependency(name="requests", current_version="2.28.0", ecosystem=Ecosystem.PYTHON, manifest_path="req.txt"),
            Dependency(name="flask", current_version="2.0.0", ecosystem=Ecosystem.PYTHON, manifest_path="req.txt"),
        ]
        mock_outdated = [
            OutdatedDependency(
                name="requests",
                current_version="2.28.0",
                latest_version="2.31.0",
                ecosystem=Ecosystem.PYTHON,
                manifest_path="req.txt",
            ),
            OutdatedDependency(
                name="flask",
                current_version="2.0.0",
                latest_version="3.0.0",
                ecosystem=Ecosystem.PYTHON,
                manifest_path="req.txt",
            ),
        ]
        mock_rag_result = RAGQueryResult(
            breaking_changes=[],
            confidence=0.9,
            source_chunks=["chunk1"],
        )
        mock_impact = ImpactAssessment(
            dep_name="requests",
            versions={"current": "2.28.0", "latest": "2.31.0"},
            impacts=[],
            summary="No impact",
            overall_severity=Severity.INFO,
        )
        mock_report = AnalysisReport(
            project_path="/tmp/myproject",
            timestamp="2024-01-01T00:00:00",
            total_dependencies=2,
            outdated_count=2,
            critical_count=0,
            assessments=[mock_impact],
            patches=[],
            errors=[],
        )

        with (
            patch("migratowl.core.analyzer.scanner.scan_project", new_callable=AsyncMock, return_value=mock_deps),
            patch("migratowl.core.analyzer.registry.find_outdated", new_callable=AsyncMock, return_value=mock_outdated),
            patch(
                "migratowl.core.analyzer.changelog.fetch_changelog",
                new_callable=AsyncMock,
                return_value=("## 2.31.0\nSome changes", []),
            ),
            patch(
                "migratowl.core.analyzer.changelog.chunk_changelog_by_version",
                return_value=[{"version": "2.31.0", "content": "Some changes"}],
            ),
            patch("migratowl.core.analyzer.rag.embed_changelog", new_callable=AsyncMock),
            patch("migratowl.core.analyzer.rag.query", new_callable=AsyncMock, return_value=mock_rag_result),
            patch("migratowl.core.analyzer.code_parser.find_usages", new_callable=AsyncMock, return_value=[]),
            patch("migratowl.core.analyzer.impact.assess_impact", new_callable=AsyncMock, return_value=mock_impact),
            patch("migratowl.core.analyzer.report.build_report", return_value=mock_report),
            patch("migratowl.core.analyzer.report.export_json", return_value=mock_report.model_dump_json(indent=2)),
        ):
            result = await analyze("/tmp/myproject", fix_mode=False)

        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed["project_path"] == "/tmp/myproject"
