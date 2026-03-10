"""CLI interface for MigratOwl — entry point for the migratowl command."""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.logging import RichHandler

from migratowl.config import settings

app = typer.Typer(name="migratowl", help="AI-powered dependency migration analyzer")
console = Console()


def configure_logging(verbosity: int = 0) -> None:
    """Configure root logger with Rich handler."""
    if verbosity >= 2:
        level = "DEBUG"
    elif verbosity == 1:
        level = "INFO"
    else:
        level = settings.log_level.upper()

    logging.basicConfig(
        level=getattr(logging, level, logging.WARNING),
        format="%(message)s",
        handlers=[RichHandler(console=console, show_time=False, show_path=False, markup=True)],
        force=True,
    )


# Lazy import to allow mocking in tests
def _get_run_analysis() -> Any:  # noqa: ANN202
    from migratowl.core.analyzer import analyze as _analyze

    return _analyze


# Module-level reference that tests can patch
run_analysis = None


_VALID_FORMATS = {"json", "markdown"}
_EXTENSION_TO_FORMAT = {".json": "json", ".md": "markdown"}
_FORMAT_TO_EXTENSION = {"json": ".json", "markdown": ".md"}


def _resolve_output_format(output: str, format_flag: str | None) -> tuple[str, str]:
    """Resolve output path and format from --output extension and --format flag."""
    if format_flag is not None and format_flag not in _VALID_FORMATS:
        raise typer.BadParameter(f"Invalid format '{format_flag}'. Must be one of: {', '.join(sorted(_VALID_FORMATS))}")

    ext = Path(output).suffix.lower()
    ext_format = _EXTENSION_TO_FORMAT.get(ext)

    if ext_format:
        if format_flag is not None and format_flag != ext_format:
            raise typer.BadParameter(f"Output extension '{ext}' conflicts with --format '{format_flag}'")
        return output, ext_format

    if ext:
        raise typer.BadParameter(f"Unrecognized output extension '{ext}'. Use .json or .md, or omit the extension")

    resolved_format = format_flag if format_flag is not None else "json"
    resolved_path = output + _FORMAT_TO_EXTENSION[resolved_format]
    return resolved_path, resolved_format


_ENV_TEMPLATE = """\
# MigratOwl Configuration
# See docs for all available settings.

# Model to use (provider:model format)
# MIGRATOWL_MODEL=openai:gpt-4o-mini
# MIGRATOWL_MODEL=anthropic:claude-sonnet-4-5-20250929

# API keys — use standard provider env vars (LangChain reads these automatically)
# OPENAI_API_KEY=sk-your-key-here
# ANTHROPIC_API_KEY=sk-ant-your-key-here

# Vectorstore path (default: .migratowl/vectorstore)
# MIGRATOWL_VECTORSTORE_PATH=.migratowl/vectorstore

# Log level (default: WARNING, use INFO or DEBUG for more output)
# MIGRATOWL_LOG_LEVEL=WARNING
"""


def _render_or_write_report(result: str, output: str | None, format: str | None) -> None:
    """Write report to file or render to console."""
    if output:
        try:
            resolved_path, resolved_format = _resolve_output_format(output, format)
        except typer.BadParameter as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(code=1) from e

        if resolved_format == "markdown":
            from migratowl.core.report import export_markdown
            from migratowl.models.schemas import AnalysisReport

            report = AnalysisReport.model_validate(json.loads(result))
            Path(resolved_path).write_text(export_markdown(report))
        else:
            Path(resolved_path).write_text(result)
        console.print(f"Report written to {resolved_path}")
    else:
        from migratowl.core.report import render_report
        from migratowl.models.schemas import AnalysisReport

        report_data = json.loads(result)
        report = AnalysisReport.model_validate(report_data)
        render_report(report, console=console)


@app.command()
def analyze(
    project_path: str = typer.Argument(..., help="Path to the project to analyze"),
    # TODO: --fix is not fully implemented — patch quality from the LLM is unreliable.
    # fix: bool = typer.Option(False, "--fix", help="Generate migration patches"),
    output: str | None = typer.Option(None, "--output", "-o", help="Output file path"),
    model: str | None = typer.Option(None, "--model", "-m", help="Override the LLM model (provider:model format)"),
    format: str | None = typer.Option(None, "--format", "-f", help="Output format: json or markdown"),
    ignore: str | None = typer.Option(None, "--ignore", "-i", help="Comma-separated dependencies to skip"),
    verbose: int = typer.Option(
        0, "--verbose", "-v", count=True, help="Increase verbosity (-v for INFO, -vv for DEBUG)"
    ),
) -> None:
    """Analyze project dependencies for breaking changes."""
    configure_logging(verbose)

    path = Path(project_path)
    if not path.exists():
        console.print(f"[red]Error: Path '{project_path}' does not exist[/red]")
        raise typer.Exit(code=1)

    if model:
        settings.model = model

    ignored_deps = [d.strip() for d in ignore.split(",") if d.strip()] if ignore else None

    global run_analysis  # noqa: PLW0603
    if run_analysis is None:
        run_analysis = _get_run_analysis()

    async def _run() -> str:
        from migratowl.core.http import close_http_client

        try:
            result: str = await run_analysis(str(path), fix_mode=False, ignored_dependencies=ignored_deps)
            return result
        finally:
            await close_http_client()

    result = asyncio.run(_run())
    _render_or_write_report(result, output, format)


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
    configure_logging()

    from migratowl.interfaces.mcp_server import mcp

    mcp.run()
