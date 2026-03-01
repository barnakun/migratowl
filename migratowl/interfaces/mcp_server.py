"""MCP server for MigratOwl — exposes analysis tools via FastMCP."""

import json
from typing import Any

from fastmcp import FastMCP

mcp = FastMCP("migratowl")


@mcp.tool()
async def analyze_dependencies(project_path: str, fix: bool = False) -> dict[str, Any]:
    """Analyze project dependencies for breaking changes and generate migration report."""
    from migratowl.core.analyzer import analyze

    result = await analyze(project_path, fix_mode=fix)
    parsed: dict[str, Any] = json.loads(result)
    return parsed


@mcp.tool()
async def get_impact_report(project_path: str) -> dict[str, Any]:
    """Get a detailed impact report for outdated dependencies."""
    from migratowl.core.analyzer import analyze

    result = await analyze(project_path, fix_mode=False)
    parsed: dict[str, Any] = json.loads(result)
    return parsed


@mcp.tool()
async def suggest_migration(project_path: str) -> dict[str, Any]:
    """Suggest migration patches for outdated dependencies."""
    from migratowl.core.analyzer import analyze

    result = await analyze(project_path, fix_mode=True)
    parsed: dict[str, Any] = json.loads(result)
    return parsed
