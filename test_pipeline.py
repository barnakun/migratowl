"""Temporary script to manually test the pre-AI pipeline.

Usage:
    uv run python test_pipeline.py [project_path]

Tests the flow: scan_project -> find_outdated (registry) -> fetch_changelog -> chunk_changelog
without any LLM calls. Prints everything the AI would receive as input.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

from migratowl.core.changelog import (
    chunk_changelog_by_version,
    fetch_changelog,
    filter_chunks_by_version_range,
)
from migratowl.core.registry import find_outdated
from migratowl.core.scanner import scan_project

console = Console()


async def run(project_path: str) -> None:
    console.print(Rule("[bold cyan]MigratOwl Pipeline Test[/bold cyan]"))
    console.print(f"[dim]Project: {project_path}[/dim]\n")

    # ── 1. Scan ──────────────────────────────────────────────────────────────
    console.print(Rule("[yellow]Step 1: Scanning project[/yellow]"))
    deps = await scan_project(project_path)
    console.print(f"Found [bold]{len(deps)}[/bold] pinned dependencies\n")
    for dep in deps:
        console.print(f"  • [cyan]{dep.name}[/cyan] [dim]{dep.current_version}[/dim]  ({dep.ecosystem})")
    console.print()

    if not deps:
        console.print("[red]No pinned dependencies found — nothing to do.[/red]")
        return

    # ── 2. Registry ───────────────────────────────────────────────────────────
    console.print(Rule("[yellow]Step 2: Querying registries[/yellow]"))
    outdated, registry_errors = await find_outdated(deps)
    for err in registry_errors:
        console.print(f"  [red]⚠  {err}[/red]")
    console.print(f"[bold]{len(outdated)}[/bold] outdated dependencies\n")

    if not outdated:
        console.print("[green]All dependencies are up to date.[/green]")
        return

    for dep in outdated:
        console.print(
            Panel(
                Text.assemble(
                    ("Name:       ", "dim"),
                    (dep.name, "bold cyan"),
                    "\n",
                    ("Current:    ", "dim"),
                    (dep.current_version, "yellow"),
                    "\n",
                    ("Latest:     ", "dim"),
                    (dep.latest_version, "green"),
                    "\n",
                    ("Ecosystem:  ", "dim"),
                    (str(dep.ecosystem), ""),
                    "\n",
                    ("Homepage:   ", "dim"),
                    (dep.homepage_url or "—", "blue"),
                    "\n",
                    ("Repository: ", "dim"),
                    (dep.repository_url or "—", "blue"),
                    "\n",
                    ("Changelog:  ", "dim"),
                    (dep.changelog_url or "—", "blue"),
                ),
                title=f"[bold]{dep.name}[/bold]",
                expand=False,
            )
        )
    console.print()

    # ── 3. Changelog fetch + chunking ─────────────────────────────────────────
    console.print(Rule("[yellow]Step 3: Fetching changelogs[/yellow]"))
    fetch_results = await asyncio.gather(
        *[
            fetch_changelog(
                changelog_url=dep.changelog_url,
                repository_url=dep.repository_url,
                dep_name=dep.name,
            )
            for dep in outdated
        ],
        return_exceptions=True,
    )
    for dep, result in zip(outdated, fetch_results):
        console.print(f"\n[bold cyan]{dep.name}[/bold cyan]  {dep.current_version} → {dep.latest_version}")

        if isinstance(result, Exception):
            console.print(f"  [red]⚠  Error fetching changelog: {result}[/red]")
            continue

        changelog_text, warnings = result  # type: ignore[misc]

        if warnings:
            for w in warnings:
                console.print(f"  [red]⚠  {w}[/red]")
            continue

        console.print(f"  [dim]Fetched {len(changelog_text):,} characters of changelog[/dim]")

        # ── 4. Chunk ──────────────────────────────────────────────────────────
        all_chunks = chunk_changelog_by_version(changelog_text)
        console.print(f"  [dim]Parsed into {len(all_chunks)} version sections[/dim]")

        relevant = filter_chunks_by_version_range(
            all_chunks,
            current_version=dep.current_version,
            latest_version=dep.latest_version,
        )

        console.print(
            f"  [bold]{len(relevant)}[/bold] sections in upgrade range  "
            f"({dep.current_version} < version ≤ {dep.latest_version})\n"
        )

        if not relevant:
            console.print("  [dim]No changelog sections found for this version range.[/dim]")
            continue

        # ── 5. Print what the AI would receive ───────────────────────────────
        # console.print(Rule(f"[magenta]AI input for {dep.name}[/magenta]", style="magenta"))
        # for chunk in relevant:
        #     console.print(
        #         Panel(
        #             chunk["content"] if chunk["content"] else "[dim](empty)[/dim]",
        #             title=f"[bold green]{chunk['version']}[/bold green]",
        #             expand=False,
        #             border_style="green",
        #         )
        #     )

    console.print()
    console.print(Rule("[bold cyan]Done[/bold cyan]"))


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "."
    resolved = str(Path(path).resolve())
    asyncio.run(run(resolved))


if __name__ == "__main__":
    main()
