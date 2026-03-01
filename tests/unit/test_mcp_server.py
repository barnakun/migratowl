"""Tests for the MCP server."""

from unittest.mock import AsyncMock, patch

import pytest

from migratowl.interfaces.mcp_server import mcp
from migratowl.models.schemas import AnalysisReport


def _make_report_json(project_path: str = "/tmp/project") -> str:
    report = AnalysisReport(
        project_path=project_path,
        timestamp="2026-01-01T00:00:00+00:00",
        total_dependencies=1,
        outdated_count=1,
        critical_count=0,
        assessments=[],
        patches=[],
        errors=[],
    )
    return report.model_dump_json()


class TestAnalyzeDependenciesTool:
    @pytest.mark.asyncio
    async def test_analyze_dependencies_tool_exists(self) -> None:
        assert await mcp._tool_manager.has_tool("analyze_dependencies")

    @pytest.mark.asyncio
    async def test_analyze_dependencies_calls_analyzer(self) -> None:
        with patch(
            "migratowl.core.analyzer.analyze",
            new_callable=AsyncMock,
            return_value=_make_report_json(),
        ) as mock_analyze:
            result = await mcp._tool_manager.call_tool(
                "analyze_dependencies",
                {"project_path": "/tmp/project", "fix": False},
            )

            mock_analyze.assert_called_once_with("/tmp/project", fix_mode=False)
            # call_tool returns a ToolResult; verify the mock was called correctly
            assert result is not None


class TestGetImpactReportTool:
    @pytest.mark.asyncio
    async def test_get_impact_report_tool_exists(self) -> None:
        assert await mcp._tool_manager.has_tool("get_impact_report")
