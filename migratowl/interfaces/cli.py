"""CLI interface for MigratOwl — entry point for the migratowl command."""

import asyncio
import json
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from migratowl.config import settings

app = typer.Typer(name="migratowl", help="AI-powered dependency migration analyzer")
console = Console()


# Lazy import to allow mocking in tests
def _get_run_analysis() -> Any:  # noqa: ANN202
    from migratowl.core.analyzer import analyze as _analyze

    return _analyze


# Module-level reference that tests can patch
run_analysis = None


_ENV_TEMPLATE = """\
# MigratOwl Configuration
# See docs for all available settings.

# Required for OpenAI (not needed if using Ollama)
# MIGRATOWL_OPENAI_API_KEY=sk-your-key-here

# Model to use (default: gpt-4o-mini)
# MIGRATOWL_OPENAI_MODEL=gpt-4o-mini

# Set to true to use local Ollama instead of OpenAI
# MIGRATOWL_USE_LOCAL_LLM=false

# Ollama base URL (default: http://localhost:11434/v1)
# MIGRATOWL_OLLAMA_BASE_URL=http://localhost:11434/v1

# Vectorstore path (default: .migratowl/vectorstore)
# MIGRATOWL_VECTORSTORE_PATH=.migratowl/vectorstore
"""


@app.command()
def analyze(
    project_path: str = typer.Argument(..., help="Path to the project to analyze"),
    fix: bool = typer.Option(False, "--fix", help="Generate migration patches"),
    output: str | None = typer.Option(None, "--output", "-o", help="Output file path"),
    model: str | None = typer.Option(None, "--model", "-m", help="Override the LLM model"),
) -> None:
    """Analyze project dependencies for breaking changes."""
    path = Path(project_path)
    if not path.exists():
        console.print(f"[red]Error: Path '{project_path}' does not exist[/red]")
        raise typer.Exit(code=1)

    if not settings.use_local_llm and not settings.openai_api_key:
        console.print(
            "[red]Error: MIGRATOWL_OPENAI_API_KEY is required (or set MIGRATOWL_USE_LOCAL_LLM=true for Ollama)[/red]"
        )
        raise typer.Exit(code=1)

    if model:
        settings.openai_model = model

    global run_analysis  # noqa: PLW0603
    if run_analysis is None:
        run_analysis = _get_run_analysis()

    result = asyncio.run(run_analysis(str(path), fix_mode=fix))

    if output:
        Path(output).write_text(result)
        console.print(f"Report written to {output}")
    else:
        from migratowl.core.report import render_report
        from migratowl.models.schemas import AnalysisReport

        report_data = json.loads(result)
        report = AnalysisReport.model_validate(report_data)
        render_report(report, console=console)


@app.command()
def init() -> None:
    """Create a .env template file with MigratOwl configuration."""
    env_path = Path(".env")
    if env_path.exists():
        console.print("[yellow].env file already exists — not overwriting[/yellow]")
        return

    env_path.write_text(_ENV_TEMPLATE)
    console.print("[green]Created .env template — edit it with your settings[/green]")


@app.command()
def serve() -> None:
    """Start the MCP server."""
    from migratowl.interfaces.mcp_server import mcp

    mcp.run()
