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
        "total_dependencies": 0,
        "dependencies": [],
        "all_code_usages": [],
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
        "all_code_usages": [],
        "code_usages": [],
        "warnings": [],
        "node_errors": [],
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
            patch(
                "migratowl.core.analyzer.registry.find_outdated",
                new_callable=AsyncMock,
                return_value=(mock_outdated, []),
            ),
        ):
            state = _make_parent_state()
            result = await scan_dependencies_node(state)

        assert isinstance(result, Command)
        assert result.goto == "parse_all_code"
        assert "dependencies" in result.update
        assert len(result.update["dependencies"]) == 1
        assert result.update["dependencies"][0]["name"] == "requests"
        assert result.update["total_dependencies"] == 1


# ---------------------------------------------------------------------------
# parse_all_code_node
# ---------------------------------------------------------------------------


class TestParseAllCodeNode:
    @pytest.mark.asyncio
    async def test_returns_usages(self) -> None:
        """parse_all_code_node stores all usages and routes to fan_out."""
        from migratowl.core.analyzer import parse_all_code_node

        mock_usages = [
            CodeUsage(
                file_path="a.py", line_number=1, usage_type="import",
                symbol="requests", code_snippet="import requests",
            ),
            CodeUsage(
                file_path="a.py", line_number=2, usage_type="import",
                symbol="flask", code_snippet="from flask import Flask",
            ),
        ]

        with patch(
            "migratowl.core.analyzer.code_parser.find_all_usages",
            new_callable=AsyncMock,
            return_value=mock_usages,
        ):
            state = _make_parent_state()
            result = await parse_all_code_node(state)

        assert result.goto == "fan_out"
        assert len(result.update["all_code_usages"]) == 2

    @pytest.mark.asyncio
    async def test_handles_error(self) -> None:
        """parse_all_code_node continues to fan_out with empty usages on error."""
        from migratowl.core.analyzer import parse_all_code_node

        with patch(
            "migratowl.core.analyzer.code_parser.find_all_usages",
            new_callable=AsyncMock,
            side_effect=RuntimeError("tree-sitter crash"),
        ):
            state = _make_parent_state()
            result = await parse_all_code_node(state)

        assert result.goto == "fan_out"
        assert result.update["all_code_usages"] == []
        assert any("parsing failed" in e.lower() for e in result.update["errors"])


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
        # node_errors must be initialised in the Send payload
        assert result[0].arg["node_errors"] == []
        assert result[1].arg["node_errors"] == []
        # all_code_usages must be passed from parent state
        assert result[0].arg["all_code_usages"] == []
        assert result[1].arg["all_code_usages"] == []

    def test_fan_out_deps_passes_all_code_usages(self) -> None:
        from migratowl.core.analyzer import fan_out_deps

        usages = [
            {
                "file_path": "a.py", "line_number": 1,
                "usage_type": "import", "symbol": "requests",
                "code_snippet": "import requests",
            },
        ]
        deps = [
            {
                "name": "requests",
                "current_version": "2.28.0",
                "latest_version": "2.31.0",
                "project_path": "/tmp/myproject",
                "changelog_url": "",
                "repository_url": "",
            },
        ]
        state = _make_parent_state(dependencies=deps, all_code_usages=usages)
        result = fan_out_deps(state)

        assert result[0].arg["all_code_usages"] == usages

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
    async def test_rag_analyze_node_low_confidence_routes_to_parse_code(self) -> None:
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
            )
            result = await rag_analyze_node(state)

        assert isinstance(result, Command)
        assert result.goto == "parse_code"
        assert result.update["rag_confidence"] == 0.3

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

        with (
            patch("migratowl.core.analyzer.changelog_cache.get_cached_changelog", return_value=None),
            patch("migratowl.core.analyzer.changelog_cache.set_cached_changelog"),
            patch(
                "migratowl.core.analyzer.changelog.fetch_changelog",
                new_callable=AsyncMock,
                return_value=("changelog text", []),
            ) as mock_fetch,
        ):
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

        with (
            patch("migratowl.core.analyzer.changelog_cache.get_cached_changelog", return_value=None),
            patch("migratowl.core.analyzer.changelog_cache.set_cached_changelog"),
            patch(
                "migratowl.core.analyzer.changelog.fetch_changelog",
                new_callable=AsyncMock,
                return_value=("", []),
            ) as mock_fetch,
        ):
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

        with (
            patch("migratowl.core.analyzer.changelog_cache.get_cached_changelog", return_value=None),
            patch("migratowl.core.analyzer.changelog_cache.set_cached_changelog"),
            patch(
                "migratowl.core.analyzer.changelog.fetch_changelog",
                new_callable=AsyncMock,
                return_value=("", ["No changelog URL or repository URL provided for requests"]),
            ),
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

        with (
            patch("migratowl.core.analyzer.changelog_cache.get_cached_changelog", return_value=None),
            patch("migratowl.core.analyzer.changelog_cache.set_cached_changelog"),
            patch(
                "migratowl.core.analyzer.changelog.fetch_changelog",
                new_callable=AsyncMock,
                return_value=("## 1.0.0\n- change", []),
            ),
        ):
            state = _make_dep_state()
            result = await fetch_changelog_node(state)

        assert result.update.get("warnings", []) == []

    @pytest.mark.asyncio
    async def test_fetch_changelog_node_uses_changelog_cache_on_hit(self) -> None:
        """When changelog cache has a hit, fetch_changelog is not called."""
        from migratowl.core.analyzer import fetch_changelog_node

        with (
            patch(
                "migratowl.core.analyzer.changelog_cache.get_cached_changelog",
                return_value=("cached text", ["cached warn"]),
            ),
            patch(
                "migratowl.core.analyzer.changelog.fetch_changelog",
                new_callable=AsyncMock,
            ) as mock_fetch,
        ):
            state = _make_dep_state()
            result = await fetch_changelog_node(state)

        mock_fetch.assert_not_called()
        assert result.update["changelog"] == "cached text"
        assert result.update["warnings"] == ["cached warn"]

    @pytest.mark.asyncio
    async def test_fetch_changelog_node_saves_to_changelog_cache_on_miss(self) -> None:
        """When changelog cache misses, fetched text is saved to cache."""
        from migratowl.core.analyzer import fetch_changelog_node

        with (
            patch(
                "migratowl.core.analyzer.changelog_cache.get_cached_changelog",
                return_value=None,
            ),
            patch(
                "migratowl.core.analyzer.changelog.fetch_changelog",
                new_callable=AsyncMock,
                return_value=("fetched text", ["warn"]),
            ),
            patch(
                "migratowl.core.analyzer.changelog_cache.set_cached_changelog",
            ) as mock_set,
        ):
            state = _make_dep_state()
            result = await fetch_changelog_node(state)

        mock_set.assert_called_once_with("requests", "fetched text", ["warn"])
        assert result.update["changelog"] == "fetched text"


# ---------------------------------------------------------------------------
# fetch_changelog_node — error handling
# ---------------------------------------------------------------------------


class TestFetchChangelogNodeErrorHandling:
    @pytest.mark.asyncio
    async def test_fetch_failure_returns_degraded_assessment_and_end(self) -> None:
        """When fetch_changelog raises, node returns degraded assessment + routes to END."""
        from langgraph.graph import END

        from migratowl.core.analyzer import fetch_changelog_node

        with (
            patch("migratowl.core.analyzer.changelog_cache.get_cached_changelog", return_value=None),
            patch(
                "migratowl.core.analyzer.changelog.fetch_changelog",
                new_callable=AsyncMock,
                side_effect=RuntimeError("HTTP 404"),
            ),
        ):
            state = _make_dep_state()
            result = await fetch_changelog_node(state)

        assert isinstance(result, Command)
        assert result.goto == END
        assessment = result.update["impact_assessments"][0]
        assert "HTTP 404" in assessment["errors"][0]

    @pytest.mark.asyncio
    async def test_cache_read_failure_nonfatal_continues(self) -> None:
        """Cache read failure in fetch_changelog_node is non-fatal — fetch proceeds."""
        from migratowl.core.analyzer import fetch_changelog_node

        with (
            patch(
                "migratowl.core.analyzer.changelog_cache.get_cached_changelog",
                side_effect=RuntimeError("cache broken"),
            ),
            patch("migratowl.core.analyzer.changelog_cache.set_cached_changelog"),
            patch(
                "migratowl.core.analyzer.changelog.fetch_changelog",
                new_callable=AsyncMock,
                return_value=("changelog text", []),
            ),
        ):
            state = _make_dep_state()
            result = await fetch_changelog_node(state)

        assert result.goto == "embed_changelog"
        assert result.update["changelog"] == "changelog text"

    @pytest.mark.asyncio
    async def test_cache_write_failure_nonfatal_continues(self) -> None:
        """Cache write failure in fetch_changelog_node is non-fatal — continues to embed."""
        from migratowl.core.analyzer import fetch_changelog_node

        with (
            patch("migratowl.core.analyzer.changelog_cache.get_cached_changelog", return_value=None),
            patch(
                "migratowl.core.analyzer.changelog_cache.set_cached_changelog",
                side_effect=RuntimeError("write error"),
            ),
            patch(
                "migratowl.core.analyzer.changelog.fetch_changelog",
                new_callable=AsyncMock,
                return_value=("changelog text", []),
            ),
        ):
            state = _make_dep_state()
            result = await fetch_changelog_node(state)

        assert result.goto == "embed_changelog"
        assert result.update["changelog"] == "changelog text"


# ---------------------------------------------------------------------------
# embed_changelog_node — error handling
# ---------------------------------------------------------------------------


class TestEmbedChangelogNodeErrorHandling:
    @pytest.mark.asyncio
    async def test_embed_failure_returns_degraded_assessment_and_end(self) -> None:
        """When embed_changelog raises, node returns degraded assessment + routes to END."""
        from langgraph.graph import END

        from migratowl.core.analyzer import embed_changelog_node

        with (
            patch(
                "migratowl.core.analyzer.changelog.chunk_changelog_by_version",
                side_effect=RuntimeError("ChromaDB down"),
            ),
        ):
            state = _make_dep_state(changelog="some text")
            result = await embed_changelog_node(state)

        assert isinstance(result, Command)
        assert result.goto == END
        assessment = result.update["impact_assessments"][0]
        assert "ChromaDB down" in assessment["errors"][0]


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
# rag_analyze_node — error handling (warning instead of silent)
# ---------------------------------------------------------------------------


class TestRagAnalyzeNodeErrorHandling:
    @pytest.mark.asyncio
    async def test_rag_exception_continues_with_error(self) -> None:
        """RAG exception must continue to parse_code with an error recorded."""
        from migratowl.core.analyzer import rag_analyze_node

        with patch(
            "migratowl.core.analyzer.rag.query",
            new_callable=AsyncMock,
            side_effect=RuntimeError("LLM validation failed"),
        ):
            state = _make_dep_state()
            result = await rag_analyze_node(state)

        assert result.goto == "parse_code"
        assert result.update["rag_results"] == []
        assert result.update["rag_confidence"] == 0.0
        assert "node_errors" in result.update
        assert any("LLM validation failed" in e or "RAG" in e for e in result.update["node_errors"])


# ---------------------------------------------------------------------------
# parse_code_node — error handling
# ---------------------------------------------------------------------------


class TestParseCodeNode:
    @pytest.mark.asyncio
    async def test_filters_usages_from_state(self) -> None:
        """parse_code_node filters all_code_usages for this dep, no I/O."""
        from migratowl.core.analyzer import parse_code_node

        usages = [
            CodeUsage(
                file_path="a.py", line_number=1, usage_type="import",
                symbol="requests", code_snippet="import requests",
            ).model_dump(),
            CodeUsage(
                file_path="a.py", line_number=2, usage_type="import",
                symbol="flask", code_snippet="from flask import Flask",
            ).model_dump(),
        ]
        state = _make_dep_state(all_code_usages=usages)
        result = await parse_code_node(state)

        assert result.goto == "assess_impact"
        assert len(result.update["code_usages"]) == 1
        assert result.update["code_usages"][0]["symbol"] == "requests"

    @pytest.mark.asyncio
    async def test_empty_all_code_usages_yields_empty_code_usages(self) -> None:
        """Empty all_code_usages yields empty code_usages (no error)."""
        from migratowl.core.analyzer import parse_code_node

        state = _make_dep_state(all_code_usages=[])
        result = await parse_code_node(state)

        assert result.goto == "assess_impact"
        assert result.update["code_usages"] == []


# ---------------------------------------------------------------------------
# assess_impact_node — error handling
# ---------------------------------------------------------------------------


class TestAssessImpactNodeErrorHandling:
    @pytest.mark.asyncio
    async def test_assess_impact_failure_returns_degraded_assessment(self) -> None:
        """assess_impact failure must return degraded assessment + END."""
        from langgraph.graph import END

        from migratowl.core.analyzer import assess_impact_node

        with patch(
            "migratowl.core.analyzer.impact.assess_impact",
            new_callable=AsyncMock,
            side_effect=RuntimeError("LLM down"),
        ):
            state = _make_dep_state(rag_results=[], code_usages=[])
            result = await assess_impact_node(state)

        assert result.goto == END
        assessment = result.update["impact_assessments"][0]
        assert "LLM down" in assessment["errors"][0]

    @pytest.mark.asyncio
    async def test_cache_write_failure_nonfatal(self) -> None:
        """Cache write failure in assess_impact_node is non-fatal."""
        from migratowl.core.analyzer import assess_impact_node

        mock_assessment = ImpactAssessment(
            dep_name="requests",
            versions={"current": "2.28.0", "latest": "2.31.0"},
            impacts=[],
            summary="No impact",
            overall_severity=Severity.INFO,
        )

        with (
            patch(
                "migratowl.core.analyzer.impact.assess_impact",
                new_callable=AsyncMock,
                return_value=mock_assessment,
            ),
            patch(
                "migratowl.core.analyzer.cache.set_cached_assessment",
                new_callable=AsyncMock,
                side_effect=RuntimeError("disk full"),
            ),
        ):
            state = _make_dep_state(rag_results=[], code_usages=[])
            result = await assess_impact_node(state)

        # Should still succeed despite cache write failure
        from langgraph.graph import END

        assert result.goto == END
        assert len(result.update["impact_assessments"]) == 1
        assert result.update["impact_assessments"][0]["dep_name"] == "requests"

    @pytest.mark.asyncio
    async def test_state_errors_propagated_to_assessment(self) -> None:
        """Errors accumulated in state (e.g. from RAG failure) must appear in the assessment."""
        from migratowl.core.analyzer import assess_impact_node

        mock_assessment = ImpactAssessment(
            dep_name="requests",
            versions={"current": "2.28.0", "latest": "2.31.0"},
            impacts=[],
            summary="No impact",
            overall_severity=Severity.INFO,
        )

        with (
            patch(
                "migratowl.core.analyzer.impact.assess_impact",
                new_callable=AsyncMock,
                return_value=mock_assessment,
            ),
            patch("migratowl.core.analyzer.cache.set_cached_assessment", new_callable=AsyncMock),
        ):
            state = _make_dep_state(
                rag_results=[],
                code_usages=[],
                node_errors=["RAG analysis failed for requests: AuthenticationError (HTTP 401)"],
            )
            result = await assess_impact_node(state)

        assessment = result.update["impact_assessments"][0]
        assert "RAG analysis failed" in assessment["errors"][0]

    @pytest.mark.asyncio
    async def test_severity_set_to_unknown_when_errors_present(self) -> None:
        """When node_errors are present and LLM returned INFO, severity must be UNKNOWN
        (analysis is incomplete, not 'no issues found')."""
        from migratowl.core.analyzer import assess_impact_node

        mock_assessment = ImpactAssessment(
            dep_name="requests",
            versions={"current": "2.28.0", "latest": "2.31.0"},
            impacts=[],
            summary="No impact detected for requests",
            overall_severity=Severity.INFO,
        )

        with (
            patch(
                "migratowl.core.analyzer.impact.assess_impact",
                new_callable=AsyncMock,
                return_value=mock_assessment,
            ),
            patch("migratowl.core.analyzer.cache.set_cached_assessment", new_callable=AsyncMock),
        ):
            state = _make_dep_state(
                rag_results=[],
                code_usages=[],
                node_errors=["RAG analysis failed for requests: AuthenticationError (HTTP 401)"],
            )
            result = await assess_impact_node(state)

        assessment = result.update["impact_assessments"][0]
        assert assessment["overall_severity"] == "unknown"
        assert "Could not be fully analyzed" in assessment["summary"]

    @pytest.mark.asyncio
    async def test_severity_set_to_unknown_when_warnings_and_no_impacts(self) -> None:
        """When warnings indicate incomplete data and no actual impacts found,
        severity must be UNKNOWN (we have no data, not 'no issues')."""
        from migratowl.core.analyzer import assess_impact_node

        mock_assessment = ImpactAssessment(
            dep_name="bcryptjs",
            versions={"current": "2.0.0", "latest": "3.0.0"},
            impacts=[],
            summary="No impact detected",
            overall_severity=Severity.INFO,
        )

        with (
            patch(
                "migratowl.core.analyzer.impact.assess_impact",
                new_callable=AsyncMock,
                return_value=mock_assessment,
            ),
            patch("migratowl.core.analyzer.cache.set_cached_assessment", new_callable=AsyncMock),
        ):
            state = _make_dep_state(
                dep_name="bcryptjs",
                current_version="2.0.0",
                latest_version="3.0.0",
                rag_results=[],
                code_usages=[],
                warnings=["Could not fetch changelog for bcryptjs"],
                node_errors=[],
            )
            result = await assess_impact_node(state)

        assessment = result.update["impact_assessments"][0]
        assert assessment["overall_severity"] == "unknown"
        assert "Could not be fully analyzed" in assessment["summary"]

    @pytest.mark.asyncio
    async def test_severity_stays_info_when_no_warnings_no_errors(self) -> None:
        """Clean analysis with no impacts stays INFO (genuinely no issues)."""
        from migratowl.core.analyzer import assess_impact_node

        mock_assessment = ImpactAssessment(
            dep_name="requests",
            versions={"current": "2.28.0", "latest": "2.31.0"},
            impacts=[],
            summary="No impact detected",
            overall_severity=Severity.INFO,
        )

        with (
            patch(
                "migratowl.core.analyzer.impact.assess_impact",
                new_callable=AsyncMock,
                return_value=mock_assessment,
            ),
            patch("migratowl.core.analyzer.cache.set_cached_assessment", new_callable=AsyncMock),
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
                warnings=[],
                node_errors=[],
            )
            result = await assess_impact_node(state)

        assessment = result.update["impact_assessments"][0]
        assert assessment["overall_severity"] == "info"

    @pytest.mark.asyncio
    async def test_severity_not_downgraded_when_errors_present(self) -> None:
        """If the LLM returned CRITICAL, errors should not downgrade it to WARNING."""
        from migratowl.core.analyzer import assess_impact_node

        mock_assessment = ImpactAssessment(
            dep_name="requests",
            versions={"current": "2.28.0", "latest": "2.31.0"},
            impacts=[],
            summary="Critical breaking changes",
            overall_severity=Severity.CRITICAL,
        )

        with (
            patch(
                "migratowl.core.analyzer.impact.assess_impact",
                new_callable=AsyncMock,
                return_value=mock_assessment,
            ),
            patch("migratowl.core.analyzer.cache.set_cached_assessment", new_callable=AsyncMock),
        ):
            state = _make_dep_state(
                rag_results=[],
                code_usages=[],
                node_errors=["Some error"],
            )
            result = await assess_impact_node(state)

        assessment = result.update["impact_assessments"][0]
        # Should stay CRITICAL, not downgrade to WARNING
        assert assessment["overall_severity"] == "critical"


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
            patch(
                "migratowl.core.analyzer.registry.find_outdated",
                new_callable=AsyncMock,
                return_value=(mock_outdated, []),
            ),
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


@patch("migratowl.core.analyzer._preflight_api_check", new_callable=AsyncMock)
class TestAnalyze:
    @pytest.mark.asyncio
    async def test_analyze_returns_json_string(self, _mock_preflight: AsyncMock) -> None:
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
            patch(
                "migratowl.core.analyzer.registry.find_outdated",
                new_callable=AsyncMock,
                return_value=(mock_outdated, []),
            ),
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
            patch(
                "migratowl.core.analyzer.code_parser.find_all_usages",
                new_callable=AsyncMock, return_value=mock_usages,
            ),
            patch("migratowl.core.analyzer.impact.assess_impact", new_callable=AsyncMock, return_value=mock_impact),
            patch("migratowl.core.analyzer.cache.get_cached_assessment", return_value=None),
            patch("migratowl.core.analyzer.cache.set_cached_assessment", new_callable=AsyncMock),
            patch("migratowl.core.analyzer.changelog_cache.get_cached_changelog", return_value=None),
            patch("migratowl.core.analyzer.changelog_cache.set_cached_changelog"),
            patch("migratowl.core.analyzer.report.build_report", return_value=mock_report),
            patch("migratowl.core.analyzer.report.export_json", return_value=mock_report.model_dump_json(indent=2)),
        ):
            result = await analyze("/tmp/myproject", fix_mode=False)

        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed["project_path"] == "/tmp/myproject"

    @pytest.mark.asyncio
    async def test_analyze_populates_assessments_in_report(self, _mock_preflight: AsyncMock) -> None:
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
            patch(
                "migratowl.core.analyzer.registry.find_outdated",
                new_callable=AsyncMock,
                return_value=(mock_outdated, []),
            ),
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
            patch("migratowl.core.analyzer.code_parser.find_all_usages", new_callable=AsyncMock, return_value=[]),
            patch("migratowl.core.analyzer.impact.assess_impact", new_callable=AsyncMock, return_value=mock_impact),
            patch("migratowl.core.analyzer.cache.get_cached_assessment", return_value=None),
            patch("migratowl.core.analyzer.cache.set_cached_assessment", new_callable=AsyncMock),
            patch("migratowl.core.analyzer.changelog_cache.get_cached_changelog", return_value=None),
            patch("migratowl.core.analyzer.changelog_cache.set_cached_changelog"),
        ):
            result = await analyze("/tmp/myproject", fix_mode=False)

        parsed = json.loads(result)
        assert parsed["total_dependencies"] == 1
        assert parsed["outdated_count"] == 1
        assert len(parsed["assessments"]) == 1
        assert parsed["assessments"][0]["dep_name"] == "requests"

    @pytest.mark.asyncio
    async def test_analyze_with_multiple_deps_does_not_raise_concurrent_update_error(self, _mock_preflight: AsyncMock) -> None:
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
            patch(
                "migratowl.core.analyzer.registry.find_outdated",
                new_callable=AsyncMock,
                return_value=(mock_outdated, []),
            ),
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
            patch("migratowl.core.analyzer.code_parser.find_all_usages", new_callable=AsyncMock, return_value=[]),
            patch("migratowl.core.analyzer.impact.assess_impact", new_callable=AsyncMock, return_value=mock_impact),
            patch("migratowl.core.analyzer.cache.get_cached_assessment", return_value=None),
            patch("migratowl.core.analyzer.cache.set_cached_assessment", new_callable=AsyncMock),
            patch("migratowl.core.analyzer.changelog_cache.get_cached_changelog", return_value=None),
            patch("migratowl.core.analyzer.changelog_cache.set_cached_changelog"),
            patch("migratowl.core.analyzer.report.build_report", return_value=mock_report),
            patch("migratowl.core.analyzer.report.export_json", return_value=mock_report.model_dump_json(indent=2)),
        ):
            result = await analyze("/tmp/myproject", fix_mode=False)

        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed["project_path"] == "/tmp/myproject"


# ---------------------------------------------------------------------------
# generate_report_node — errors round-trip
# ---------------------------------------------------------------------------


class TestGenerateReportNodeErrorsRoundTrip:
    @pytest.mark.asyncio
    async def test_assessment_with_errors_survives_model_validate(self) -> None:
        """Assessment dicts with errors field must survive model_validate in generate_report_node."""
        from migratowl.core.analyzer import generate_report_node

        assessment_dict = ImpactAssessment(
            dep_name="bcryptjs",
            versions={"current": "2.0.0", "latest": "3.0.0"},
            impacts=[],
            summary="Could not be fully analyzed",
            overall_severity=Severity.UNKNOWN,
            warnings=["No changelog found"],
            errors=["Changelog fetch failed for bcryptjs: 404 Not Found"],
        ).model_dump()

        state = _make_parent_state(
            impact_assessments=[assessment_dict],
            patches=[],
            errors=["Registry error for some-pkg"],
        )

        with (
            patch("migratowl.core.analyzer.report.build_report") as mock_build,
            patch("migratowl.core.analyzer.report.export_json", return_value="{}"),
        ):
            await generate_report_node(state)

        # build_report must receive ImpactAssessment objects with errors preserved
        called_assessments = mock_build.call_args.kwargs["assessments"]
        assert len(called_assessments) == 1
        assert called_assessments[0].errors == ["Changelog fetch failed for bcryptjs: 404 Not Found"]
        assert called_assessments[0].warnings == ["No changelog found"]


# ---------------------------------------------------------------------------
# scan_dependencies_node — registry error propagation
# ---------------------------------------------------------------------------


class TestScanDependenciesRegistryErrors:
    @pytest.mark.asyncio
    async def test_registry_errors_propagate_into_state(self) -> None:
        """Failed registry lookups must be put into AnalysisState['errors']."""
        from migratowl.core.analyzer import scan_dependencies_node

        mock_deps = [
            Dependency(name="bad-pkg", current_version="1.0.0", ecosystem=Ecosystem.PYTHON, manifest_path="req.txt"),
        ]
        registry_errors = ["Registry query failed for bad-pkg: Not Found"]

        with (
            patch("migratowl.core.analyzer.scanner.scan_project", new_callable=AsyncMock, return_value=mock_deps),
            patch(
                "migratowl.core.analyzer.registry.find_outdated",
                new_callable=AsyncMock,
                return_value=([], registry_errors),
            ),
        ):
            state = _make_parent_state()
            result = await scan_dependencies_node(state)

        assert isinstance(result, Command)
        assert "errors" in result.update
        assert len(result.update["errors"]) == 1
        assert "bad-pkg" in result.update["errors"][0]

    @pytest.mark.asyncio
    async def test_no_errors_when_all_registry_lookups_succeed(self) -> None:
        """When all registry lookups succeed, errors list is empty."""
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
            patch(
                "migratowl.core.analyzer.registry.find_outdated",
                new_callable=AsyncMock,
                return_value=(mock_outdated, []),
            ),
        ):
            state = _make_parent_state()
            result = await scan_dependencies_node(state)

        assert result.update.get("errors", []) == []


# ---------------------------------------------------------------------------
# check_cache_node
# ---------------------------------------------------------------------------


class TestCheckCacheNode:
    @pytest.mark.asyncio
    async def test_cache_hit_routes_to_end_with_assessment(self) -> None:
        """A cache hit must skip the full pipeline and return the cached assessment."""
        from langgraph.graph import END

        from migratowl.core.analyzer import check_cache_node

        cached = {"dep_name": "requests", "summary": "cached", "overall_severity": "info"}

        with patch("migratowl.core.analyzer.cache.get_cached_assessment", return_value=cached):
            state = _make_dep_state()
            result = await check_cache_node(state)

        assert isinstance(result, Command)
        assert result.goto == END
        assert result.update["impact_assessments"] == [cached]

    @pytest.mark.asyncio
    async def test_cache_miss_routes_to_fetch_changelog(self) -> None:
        """A cache miss must route to fetch_changelog to run the full pipeline."""
        from migratowl.core.analyzer import check_cache_node

        with patch("migratowl.core.analyzer.cache.get_cached_assessment", return_value=None):
            state = _make_dep_state()
            result = await check_cache_node(state)

        assert isinstance(result, Command)
        assert result.goto == "fetch_changelog"
        assert result.update == {}


# ---------------------------------------------------------------------------
# _make_degraded_assessment helper
# ---------------------------------------------------------------------------


class TestMakeDegradedAssessment:
    def test_produces_correct_structure(self) -> None:
        from migratowl.core.analyzer import _make_degraded_assessment

        state = _make_dep_state(warnings=["some prior warning"])
        result = _make_degraded_assessment(state, "Something went wrong")

        assert result["dep_name"] == "requests"
        assert result["versions"] == {"current": "2.28.0", "latest": "2.31.0"}
        assert result["impacts"] == []
        assert "Could not be fully analyzed" in result["summary"]
        assert result["overall_severity"] == "unknown"
        assert "some prior warning" in result["warnings"]
        assert "Something went wrong" in result["errors"]

    def test_produces_valid_impact_assessment(self) -> None:
        from migratowl.core.analyzer import _make_degraded_assessment

        state = _make_dep_state()
        result = _make_degraded_assessment(state, "test error")
        # Must be deserializable back into ImpactAssessment
        ia = ImpactAssessment.model_validate(result)
        assert ia.dep_name == "requests"
        assert ia.errors == ["test error"]


# ---------------------------------------------------------------------------
# check_cache_node — cache read failure
# ---------------------------------------------------------------------------


class TestCheckCacheNodeErrorHandling:
    @pytest.mark.asyncio
    async def test_cache_read_failure_routes_to_fetch_changelog(self) -> None:
        """Cache read failure must not crash — route to fetch_changelog."""
        from migratowl.core.analyzer import check_cache_node

        with patch(
            "migratowl.core.analyzer.cache.get_cached_assessment",
            side_effect=RuntimeError("disk error"),
        ):
            state = _make_dep_state()
            result = await check_cache_node(state)

        assert isinstance(result, Command)
        assert result.goto == "fetch_changelog"


# ---------------------------------------------------------------------------
# assess_impact_node — cache saving
# ---------------------------------------------------------------------------


class TestAssessImpactNodeCacheSave:
    @pytest.mark.asyncio
    async def test_saves_assessment_to_cache(self) -> None:
        """assess_impact_node must persist the assessment to cache after computing it."""
        from migratowl.core.analyzer import assess_impact_node

        mock_assessment = ImpactAssessment(
            dep_name="requests",
            versions={"current": "2.28.0", "latest": "2.31.0"},
            impacts=[],
            summary="No impact",
            overall_severity=Severity.INFO,
        )

        with (
            patch("migratowl.core.analyzer.impact.assess_impact", new_callable=AsyncMock, return_value=mock_assessment),
            patch("migratowl.core.analyzer.cache.set_cached_assessment", new_callable=AsyncMock) as mock_set,
        ):
            state = _make_dep_state(
                dep_name="requests",
                current_version="2.28.0",
                latest_version="2.31.0",
                project_path="/tmp/myproject",
            )
            await assess_impact_node(state)

        mock_set.assert_called_once()
        call_kwargs = mock_set.call_args
        assert call_kwargs.args[0] == "/tmp/myproject"
        assert call_kwargs.args[1] == "requests"
        assert call_kwargs.args[2] == "2.28.0"
        assert call_kwargs.args[3] == "2.31.0"


# ---------------------------------------------------------------------------
# get_dep_semaphore / fetch_changelog_node concurrency cap
# ---------------------------------------------------------------------------


class TestDepSemaphore:
    def test_get_dep_semaphore_returns_asyncio_semaphore(self) -> None:
        import asyncio

        import migratowl.core.analyzer as analyzer_module
        from migratowl.core.analyzer import get_dep_semaphore

        analyzer_module._dep_semaphore = None
        with patch("migratowl.core.analyzer.settings") as mock_settings:
            mock_settings.max_concurrent_deps = 10
            mock_settings.confidence_threshold = 0.6
            sem = get_dep_semaphore()
            assert isinstance(sem, asyncio.Semaphore)
        analyzer_module._dep_semaphore = None

    def test_get_dep_semaphore_returns_singleton(self) -> None:
        import migratowl.core.analyzer as analyzer_module
        from migratowl.core.analyzer import get_dep_semaphore

        analyzer_module._dep_semaphore = None
        with patch("migratowl.core.analyzer.settings") as mock_settings:
            mock_settings.max_concurrent_deps = 5
            mock_settings.confidence_threshold = 0.6
            s1 = get_dep_semaphore()
            s2 = get_dep_semaphore()
            assert s1 is s2
        analyzer_module._dep_semaphore = None

    @pytest.mark.asyncio
    async def test_fetch_changelog_node_caps_concurrent_executions(self) -> None:
        """At most max_concurrent_deps fetch_changelog nodes run simultaneously."""
        import asyncio

        import migratowl.core.analyzer as analyzer_module
        from migratowl.core.analyzer import fetch_changelog_node

        max_concurrent = 5
        analyzer_module._dep_semaphore = None
        active = 0
        peak = 0

        async def slow_fetch(**kwargs):  # noqa: ARG001
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.02)
            active -= 1
            return ("changelog text", [])

        with (
            patch("migratowl.core.analyzer.changelog.fetch_changelog", side_effect=slow_fetch),
            patch("migratowl.core.analyzer.changelog_cache.get_cached_changelog", return_value=None),
            patch("migratowl.core.analyzer.changelog_cache.set_cached_changelog"),
            patch("migratowl.core.analyzer.settings") as mock_settings,
        ):
            mock_settings.max_concurrent_deps = max_concurrent
            mock_settings.confidence_threshold = 0.6
            await asyncio.gather(*[fetch_changelog_node(_make_dep_state()) for _ in range(20)])

        assert peak <= max_concurrent, f"Expected peak ≤ {max_concurrent}, got {peak}"
        analyzer_module._dep_semaphore = None


# ---------------------------------------------------------------------------
# _clean_error_message
# ---------------------------------------------------------------------------


class TestCleanErrorMessage:
    def test_simple_exception_returns_short_message(self) -> None:
        from migratowl.core.analyzer import _clean_error_message

        exc = RuntimeError("connection refused")
        result = _clean_error_message(exc)
        assert result == "connection refused"

    def test_strips_raw_api_error_dict(self) -> None:
        """OpenAI errors often repr as a dict with nested 'error' key.
        _clean_error_message should extract just the human-readable part."""
        from migratowl.core.analyzer import _clean_error_message

        # Simulate an OpenAI-style error with a long repr
        exc = Exception(
            "Error code: 401 - {'error': {'message': 'Incorrect API key provided: sk-proj-***abc.', "
            "'type': 'invalid_request_error', 'param': None, 'code': 'invalid_api_key'}}"
        )
        result = _clean_error_message(exc)
        # Should NOT contain the raw dict or API key fragment
        assert "sk-proj-" not in result
        assert "{'error'" not in result
        # Should contain the useful info
        assert "401" in result

    def test_httpx_status_error_cleaned(self) -> None:
        """httpx HTTPStatusError includes full URL — extract status + short URL."""
        from migratowl.core.analyzer import _clean_error_message

        # httpx.HTTPStatusError str() looks like:
        # "Client error '404 Not Found' for url 'https://raw.githubusercontent.com/...long...'"
        exc = Exception(
            "Client error '404 Not Found' for url "
            "'https://raw.githubusercontent.com/user/repo/main/CHANGELOG.md'"
        )
        result = _clean_error_message(exc)
        assert "404 Not Found" in result
        # Should not include the full raw URL
        assert len(result) <= 200

    def test_pydantic_validation_error_cleaned(self) -> None:
        """Pydantic ValidationError is multi-line and verbose — extract count + model."""
        from migratowl.core.analyzer import _clean_error_message

        # Simulate pydantic.ValidationError str() output
        exc = Exception(
            "3 validation errors for RAGQueryResult\n"
            "breaking_changes -> 0 -> description\n"
            "  field required (type=value_error.missing)\n"
            "breaking_changes -> 1 -> severity\n"
            "  field required (type=value_error.missing)\n"
            "confidence\n"
            "  field required (type=value_error.missing)"
        )
        result = _clean_error_message(exc)
        assert "validation error" in result.lower()
        # Should be concise, not multi-line
        assert "\n" not in result

    def test_long_message_truncated(self) -> None:
        from migratowl.core.analyzer import _clean_error_message

        exc = RuntimeError("x" * 500)
        result = _clean_error_message(exc)
        assert len(result) <= 200


# ---------------------------------------------------------------------------
# Logging verbosity — tracebacks at DEBUG only
# ---------------------------------------------------------------------------


class TestLoggingVerbosity:
    @pytest.mark.asyncio
    async def test_fetch_failure_no_traceback_at_warning_level(self, caplog: pytest.LogCaptureFixture) -> None:
        """Failed nodes should log at WARNING without exc_info (no traceback).
        Full tracebacks should only appear at DEBUG level."""
        import logging

        from migratowl.core.analyzer import fetch_changelog_node

        with (
            patch("migratowl.core.analyzer.changelog_cache.get_cached_changelog", return_value=None),
            patch(
                "migratowl.core.analyzer.changelog.fetch_changelog",
                new_callable=AsyncMock,
                side_effect=RuntimeError("HTTP 404"),
            ),
            caplog.at_level(logging.WARNING, logger="migratowl.core.analyzer"),
        ):
            await fetch_changelog_node(_make_dep_state())

        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        # Warning records should NOT have exc_info (no traceback)
        for record in warning_records:
            assert record.exc_info is None or record.exc_info[0] is None, (
                "WARNING log should not include traceback (exc_info)"
            )

    @pytest.mark.asyncio
    async def test_fetch_failure_traceback_at_debug_level(self, caplog: pytest.LogCaptureFixture) -> None:
        """Full traceback should be available at DEBUG level."""
        import logging

        from migratowl.core.analyzer import fetch_changelog_node

        with (
            patch("migratowl.core.analyzer.changelog_cache.get_cached_changelog", return_value=None),
            patch(
                "migratowl.core.analyzer.changelog.fetch_changelog",
                new_callable=AsyncMock,
                side_effect=RuntimeError("HTTP 404"),
            ),
            caplog.at_level(logging.DEBUG, logger="migratowl.core.analyzer"),
        ):
            await fetch_changelog_node(_make_dep_state())

        debug_records = [r for r in caplog.records if r.levelno == logging.DEBUG]
        assert any(r.exc_info is not None and r.exc_info[0] is not None for r in debug_records), (
            "DEBUG log should include traceback (exc_info)"
        )

    @pytest.mark.asyncio
    async def test_error_messages_in_assessment_are_clean(self) -> None:
        """Error messages stored in degraded assessments should be concise, not raw dicts."""
        from langgraph.graph import END

        from migratowl.core.analyzer import fetch_changelog_node

        api_error = Exception(
            "Error code: 401 - {'error': {'message': 'Incorrect API key', "
            "'type': 'invalid_request_error', 'param': None, 'code': 'invalid_api_key'}}"
        )

        with (
            patch("migratowl.core.analyzer.changelog_cache.get_cached_changelog", return_value=None),
            patch(
                "migratowl.core.analyzer.changelog.fetch_changelog",
                new_callable=AsyncMock,
                side_effect=api_error,
            ),
        ):
            result = await fetch_changelog_node(_make_dep_state())

        assert result.goto == END
        error_msg = result.update["impact_assessments"][0]["errors"][0]
        assert "{'error'" not in error_msg
        assert len(error_msg) <= 200


# ---------------------------------------------------------------------------
# Pre-flight API check
# ---------------------------------------------------------------------------


class TestPreflightCheck:
    @pytest.mark.asyncio
    async def test_preflight_success_continues_analysis(self) -> None:
        """When the pre-flight check passes, analysis proceeds normally."""
        from migratowl.core.analyzer import _preflight_api_check

        with patch(
            "migratowl.core.analyzer.llm.get_embedding",
            new_callable=AsyncMock,
            return_value=[0.1] * 10,
        ):
            # Should not raise
            await _preflight_api_check()

    @pytest.mark.asyncio
    async def test_preflight_auth_error_raises_with_clear_message(self) -> None:
        """When the API key is invalid, pre-flight raises with a user-friendly message."""
        from openai import AuthenticationError

        from migratowl.core.analyzer import _preflight_api_check

        auth_error = AuthenticationError(
            message="Incorrect API key",
            response=AsyncMock(status_code=401),
            body={"error": {"message": "Incorrect API key", "code": "invalid_api_key"}},
        )

        with (
            patch(
                "migratowl.core.analyzer.llm.get_embedding",
                new_callable=AsyncMock,
                side_effect=auth_error,
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            await _preflight_api_check()

        assert exc_info.value.code == 1

    @pytest.mark.asyncio
    async def test_preflight_connection_error_raises_with_clear_message(self) -> None:
        """When the API is unreachable, pre-flight raises with a user-friendly message."""
        from openai import APIConnectionError

        from migratowl.core.analyzer import _preflight_api_check

        with (
            patch(
                "migratowl.core.analyzer.llm.get_embedding",
                new_callable=AsyncMock,
                side_effect=APIConnectionError(request=AsyncMock()),
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            await _preflight_api_check()

        assert exc_info.value.code == 1

    @pytest.mark.asyncio
    async def test_preflight_transient_error_does_not_block(self) -> None:
        """Transient errors (rate limit, 500) should NOT block analysis."""
        from migratowl.core.analyzer import _preflight_api_check

        with patch(
            "migratowl.core.analyzer.llm.get_embedding",
            new_callable=AsyncMock,
            side_effect=RuntimeError("temporary glitch"),
        ):
            # Should not raise — transient errors are not fatal
            await _preflight_api_check()

    @pytest.mark.asyncio
    async def test_analyze_calls_preflight(self) -> None:
        """analyze() must call _preflight_api_check before running the graph."""
        from migratowl.core.analyzer import analyze

        with (
            patch(
                "migratowl.core.analyzer._preflight_api_check",
                new_callable=AsyncMock,
                side_effect=SystemExit(1),
            ) as mock_preflight,
            pytest.raises(SystemExit),
        ):
            await analyze("/tmp/myproject")

        mock_preflight.assert_called_once()
