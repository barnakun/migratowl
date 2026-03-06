"""MCP server for MigratOwl — exposes analysis tools via FastMCP."""

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastmcp import FastMCP


@asynccontextmanager
async def _lifespan(app: Any) -> AsyncIterator[None]:
    yield
    from migratowl.core.http import close_http_client

    await close_http_client()


mcp = FastMCP("migratowl", lifespan=_lifespan)


@mcp.tool()
async def analyze_dependencies(project_path: str, fix: bool = False, ignored_dependencies: str = "") -> dict[str, Any]:
    """Analyze project dependencies for breaking changes and generate migration report."""
    from migratowl.core.analyzer import analyze

    ignored = [d.strip() for d in ignored_dependencies.split(",") if d.strip()] if ignored_dependencies else None
    result = await analyze(project_path, fix_mode=fix, ignored_dependencies=ignored)
    parsed: dict[str, Any] = json.loads(result)
    return parsed


@mcp.tool()
async def get_impact_report(project_path: str, ignored_dependencies: str = "") -> dict[str, Any]:
    """Get a detailed impact report for outdated dependencies."""
    from migratowl.core.analyzer import analyze

    ignored = [d.strip() for d in ignored_dependencies.split(",") if d.strip()] if ignored_dependencies else None
    result = await analyze(project_path, fix_mode=False, ignored_dependencies=ignored)
    parsed: dict[str, Any] = json.loads(result)
    return parsed


@mcp.tool()
async def suggest_migration(project_path: str) -> dict[str, Any]:
    """Suggest migration patches for outdated dependencies."""
    from migratowl.core.analyzer import analyze

    result = await analyze(project_path, fix_mode=True)
    parsed: dict[str, Any] = json.loads(result)
    return parsed
