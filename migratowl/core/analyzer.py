"""LangGraph StateGraph — THE agent, all orchestration."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
from typing import Any

import httpx
import openai
from instructor.core import InstructorRetryException
from langgraph.graph import END, StateGraph
from langgraph.types import Command, Send
from openai import APIConnectionError, AuthenticationError

from migratowl.config import settings
from migratowl.core import cache, changelog, changelog_cache, code_parser, impact, llm, patcher, rag, registry, scanner

# Import report as a module reference so tests can patch it.
from migratowl.core import report as report  # noqa: PLW0127
from migratowl.models.schemas import (
    AnalysisState,
    BreakingChange,
    CodeUsage,
    DepAnalysisState,
    ImpactAssessment,
    PatchSet,
    Severity,
)

logger = logging.getLogger(__name__)


def _normalize_dep_name(name: str) -> str:
    """Normalize dependency name: lowercase and replace hyphens with underscores."""
    return name.lower().replace("-", "_")

# Lazily initialised semaphore that caps the number of deps running through the
# expensive pipeline (changelog fetch → embed → RAG → impact) concurrently.
# Cache hits in check_cache_node bypass this entirely.
_dep_semaphore: asyncio.Semaphore | None = None


def get_dep_semaphore() -> asyncio.Semaphore:
    """Return the module-level semaphore that gates concurrent dep analyses."""
    global _dep_semaphore
    if _dep_semaphore is None:
        _dep_semaphore = asyncio.Semaphore(settings.max_concurrent_deps)
    return _dep_semaphore


# ---------------------------------------------------------------------------
# Error message helpers
# ---------------------------------------------------------------------------

# Matches "Error code: NNN - {..." pattern from OpenAI SDK errors
_API_ERROR_RE = re.compile(r"Error code:\s*(\d+)\s*-\s*\{")
# Matches "N validation errors for ModelName" from Pydantic
_VALIDATION_ERROR_RE = re.compile(r"(\d+)\s+validation\s+errors?\s+for\s+(\w+)")


def _clean_error_message(exc: BaseException, *, max_length: int = 200) -> str:
    """Extract a concise, human-readable message from an exception.

    Handles verbose patterns from common libraries:
    - OpenAI SDK: 'Error code: 401 - {full JSON dict}' → 'Exception (HTTP 401)'
    - Pydantic: multi-line validation errors → '3 validation errors for Model'
    - General: truncates to max_length
    """
    raw = str(exc)

    # OpenAI SDK: strip the raw JSON dict
    match = _API_ERROR_RE.search(raw)
    if match:
        code = match.group(1)
        exc_type = type(exc).__name__
        return f"{exc_type} (HTTP {code})"

    # Pydantic ValidationError: extract just the summary line
    match = _VALIDATION_ERROR_RE.search(raw)
    if match:
        return f"{match.group(1)} validation errors for {match.group(2)}"

    # Strip newlines for any multi-line exception messages
    if "\n" in raw:
        raw = raw.split("\n")[0]

    if len(raw) > max_length:
        raw = raw[: max_length - 3] + "..."
    return raw


# ---------------------------------------------------------------------------
# Per-dependency worker nodes (DepAnalysisState)
# ---------------------------------------------------------------------------


def _make_degraded_assessment(state: DepAnalysisState, error_msg: str) -> dict:
    """Build a degraded ImpactAssessment dict when a node cannot complete."""
    return ImpactAssessment(
        dep_name=state["dep_name"],
        versions={"current": state["current_version"], "latest": state["latest_version"]},
        impacts=[],
        summary="Could not be fully analyzed",
        overall_severity=Severity.UNKNOWN,
        warnings=state.get("warnings", []),
        errors=[error_msg],
    ).model_dump()


async def check_cache_node(state: DepAnalysisState) -> Command:
    """Return cached impact assessment if available, skipping the full pipeline."""
    try:
        cached = cache.get_cached_assessment(
            state["project_path"],
            state["dep_name"],
            state["current_version"],
            state["latest_version"],
        )
    except (json.JSONDecodeError, OSError, KeyError, TypeError):
        logger.warning("Cache read failed for %s, continuing without cache", state["dep_name"])
        logger.debug("Cache read traceback for %s", state["dep_name"], exc_info=True)
        return Command(goto="fetch_changelog", update={})
    if cached is not None:
        return Command(goto=END, update={"impact_assessments": [cached]})
    return Command(goto="fetch_changelog", update={})


async def fetch_changelog_node(state: DepAnalysisState) -> Command:
    """Fetch changelog text for a dependency.

    Gated by get_dep_semaphore() to cap the number of deps running the full
    pipeline concurrently (default 20). Cache hits never reach this node.
    """
    async with get_dep_semaphore():
        # Cache read is non-fatal
        try:
            cached = changelog_cache.get_cached_changelog(state["dep_name"])
        except (json.JSONDecodeError, OSError, KeyError, TypeError):
            logger.warning("Changelog cache read failed for %s", state["dep_name"])
            logger.debug("Changelog cache read traceback for %s", state["dep_name"], exc_info=True)
            cached = None

        if cached is not None:
            text, warnings = cached
        else:
            try:
                text, warnings = await changelog.fetch_changelog(
                    changelog_url=state["changelog_url"] or None,
                    repository_url=state["repository_url"] or None,
                    dep_name=state["dep_name"],
                )
            except (httpx.HTTPStatusError, httpx.RequestError, ValueError, FileNotFoundError) as exc:
                error_msg = f"Changelog fetch failed for {state['dep_name']}: {_clean_error_message(exc)}"
                logger.warning(error_msg)
                logger.debug("Changelog fetch traceback for %s", state["dep_name"], exc_info=True)
                return Command(
                    goto=END,
                    update={"impact_assessments": [_make_degraded_assessment(state, error_msg)]},
                )

            # Cache write is non-fatal
            try:
                changelog_cache.set_cached_changelog(state["dep_name"], text, warnings)
            except (OSError, TypeError):
                logger.warning("Changelog cache write failed for %s", state["dep_name"])
                logger.debug("Changelog cache write traceback for %s", state["dep_name"], exc_info=True)

    update: dict = {"changelog": text}
    if warnings:
        update["warnings"] = warnings
    return Command(goto="embed_changelog", update=update)


async def embed_changelog_node(state: DepAnalysisState) -> Command:
    """Chunk and embed changelog into ChromaDB."""
    try:
        dep_name = state["dep_name"]
        chunks = changelog.chunk_changelog_by_version(state["changelog"])
        warnings: list[str] = []

        if not chunks:
            warnings.append(f"No parseable version headers found in {dep_name} changelog")
        else:
            chunks = changelog.filter_chunks_by_version_range(chunks, state["current_version"], state["latest_version"])
            if not chunks:
                warnings.append(
                    f"No changelog entries found for {dep_name} between "
                    f"{state['current_version']} and {state['latest_version']}"
                )

        await rag.embed_changelog(dep_name, chunks, state["project_path"])
    except Exception as exc:  # noqa: BLE001  # ChromaDB lacks stable exception types
        error_msg = f"Embed changelog failed for {state['dep_name']}: {_clean_error_message(exc)}"
        logger.warning(error_msg)
        logger.debug("Embed changelog traceback for %s", state["dep_name"], exc_info=True)
        return Command(
            goto=END,
            update={"impact_assessments": [_make_degraded_assessment(state, error_msg)]},
        )

    update: dict = {}
    if warnings:
        update["warnings"] = warnings
    return Command(goto="rag_analyze", update=update)


async def rag_analyze_node(state: DepAnalysisState) -> Command:
    """Query RAG for breaking changes."""
    query_text = (
        f"breaking changes in {state['dep_name']} between {state['current_version']} and {state['latest_version']}"
    )
    try:
        result = await rag.query(query_text, state["dep_name"], project_path=state["project_path"])
    except Exception as exc:  # noqa: BLE001  # ChromaDB lacks stable exception types
        warn_msg = f"RAG analysis failed for {state['dep_name']}: {_clean_error_message(exc)}"
        logger.warning(warn_msg)
        logger.debug("RAG analysis traceback for %s", state["dep_name"], exc_info=True)
        return Command(
            goto="parse_code",
            update={"rag_results": [], "rag_confidence": 0.0, "node_errors": [warn_msg]},
        )

    rag_results = [bc.model_dump() for bc in result.breaking_changes]

    return Command(
        goto="parse_code",
        update={
            "rag_results": rag_results,
            "rag_confidence": result.confidence,
        },
    )


async def parse_code_node(state: DepAnalysisState) -> Command:
    """Filter pre-parsed code usages for this dependency."""
    all_usages = [CodeUsage.model_validate(u) for u in state["all_code_usages"]]
    filtered = code_parser.filter_usages_for_dep(all_usages, state["dep_name"])
    return Command(
        goto="assess_impact",
        update={"code_usages": [u.model_dump() for u in filtered]},
    )


async def assess_impact_node(state: DepAnalysisState) -> Command:
    """Assess impact of breaking changes on code usages."""
    breaking_changes = [BreakingChange.model_validate(bc) for bc in state["rag_results"]]
    code_usages = [CodeUsage.model_validate(cu) for cu in state["code_usages"]]

    node_warnings: list[str] = []
    if not code_usages:
        node_warnings.append(f"No usages of {state['dep_name']} found in project code")

    try:
        assessment = await impact.assess_impact(
            dep_name=state["dep_name"],
            current_version=state["current_version"],
            latest_version=state["latest_version"],
            breaking_changes=breaking_changes,
            code_usages=code_usages,
        )
    except (
        openai.APIError,
        openai.APIConnectionError,
        httpx.RequestError,
        InstructorRetryException,
    ) as exc:
        error_msg = f"Impact assessment failed for {state['dep_name']}: {_clean_error_message(exc)}"
        logger.warning(error_msg)
        logger.debug("Impact assessment traceback for %s", state["dep_name"], exc_info=True)
        return Command(
            goto=END,
            update={"impact_assessments": [_make_degraded_assessment(state, error_msg)]},
        )

    all_warnings = state.get("warnings", []) + node_warnings
    all_errors = state.get("node_errors", [])
    assessment_dict = assessment.model_dump()
    assessment_dict["warnings"] = all_warnings
    assessment_dict["errors"] = all_errors

    # If upstream nodes recorded errors, the analysis is incomplete
    if all_errors and assessment_dict["overall_severity"] == Severity.INFO.value:
        assessment_dict["overall_severity"] = Severity.UNKNOWN.value
        assessment_dict["summary"] = "Could not be fully analyzed"

    # If warnings indicate incomplete data and no actual impacts found,
    # the "info" result is unreliable — mark as unknown
    elif all_warnings and not assessment_dict["impacts"] and assessment_dict["overall_severity"] == Severity.INFO.value:
        assessment_dict["overall_severity"] = Severity.UNKNOWN.value
        assessment_dict["summary"] = "Could not be fully analyzed"

    try:
        await cache.set_cached_assessment(
            state["project_path"],
            state["dep_name"],
            state["current_version"],
            state["latest_version"],
            assessment_dict,
        )
    except (OSError, TypeError):
        logger.warning("Cache write failed for %s", state["dep_name"])
        logger.debug("Cache write traceback for %s", state["dep_name"], exc_info=True)

    return Command(goto=END, update={"impact_assessments": [assessment_dict]})


# ---------------------------------------------------------------------------
# Parent graph nodes (AnalysisState)
# ---------------------------------------------------------------------------


async def scan_dependencies_node(state: AnalysisState) -> Command:
    """Scan project and find outdated dependencies."""
    deps = await scanner.scan_project(state["project_path"])
    outdated, registry_errors = await registry.find_outdated(deps)

    ignored = {_normalize_dep_name(d) for d in state.get("ignored_dependencies", [])}
    if ignored:
        before_count = len(outdated)
        outdated = [od for od in outdated if _normalize_dep_name(od.name) not in ignored]
        skipped = before_count - len(outdated)
        if skipped:
            logger.info("Skipped %d ignored dependenc%s", skipped, "y" if skipped == 1 else "ies")

    dep_dicts = [
        {
            "name": od.name,
            "current_version": od.current_version,
            "latest_version": od.latest_version,
            "project_path": state["project_path"],
            "changelog_url": od.changelog_url or "",
            "repository_url": od.repository_url or "",
        }
        for od in outdated
    ]

    update: dict = {"dependencies": dep_dicts, "total_dependencies": len(deps)}
    if registry_errors:
        update["errors"] = registry_errors
    return Command(goto="parse_all_code", update=update)


async def parse_all_code_node(state: AnalysisState) -> Command:
    """Parse all project source files once and store usages in state."""
    try:
        usages = await code_parser.find_all_usages(state["project_path"])
    except (OSError, UnicodeDecodeError) as exc:
        warn_msg = f"Project code parsing failed: {_clean_error_message(exc)}"
        logger.warning(warn_msg)
        return Command(goto="fan_out", update={"all_code_usages": [], "errors": [warn_msg]})
    return Command(goto="fan_out", update={"all_code_usages": [u.model_dump() for u in usages]})


def fan_out_node(state: AnalysisState) -> None:
    """Pass-through node; conditional edges dispatch Sends from here."""
    return None


def fan_out_deps(state: AnalysisState) -> list[Send]:
    """Create Send objects for parallel per-dependency analysis."""
    return [
        Send(
            "analyze_dep",
            {
                "dep_name": dep["name"],
                "current_version": dep["current_version"],
                "latest_version": dep["latest_version"],
                "project_path": dep["project_path"],
                "changelog_url": dep.get("changelog_url", ""),
                "repository_url": dep.get("repository_url", ""),
                "changelog": "",
                "rag_results": [],
                "rag_confidence": 0.0,
                "all_code_usages": state["all_code_usages"],
                "code_usages": [],
                "impact_assessments": [],
                "warnings": [],
                "node_errors": [],
            },
        )
        for dep in state["dependencies"]
    ]


async def cleanup_embeddings_node(state: AnalysisState) -> Command:
    """Remove stale ChromaDB embeddings for deps no longer in the project."""
    active_deps = {d["name"] for d in state["dependencies"]}
    purged = rag.purge_stale_embeddings(active_deps, state["project_path"])
    if purged:
        logger.info("Purged stale embeddings: %s", purged)
    return Command(goto="route_results", update={})


def route_after_fan_in(state: AnalysisState) -> Command:
    """Route to patches or directly to report based on fix_mode."""
    if state["fix_mode"]:
        return Command(goto="generate_patches")
    return Command(goto="generate_report")


async def generate_patches_node(state: AnalysisState) -> Command:
    """Generate patches for breaking changes using the patcher module."""
    assessments = [ImpactAssessment.model_validate(a) for a in state["impact_assessments"]]
    patch_sets = await patcher.generate_patches(assessments, state["project_path"])
    patches_json = [ps.model_dump_json() for ps in patch_sets]
    return Command(goto="generate_report", update={"patches": patches_json})


async def generate_report_node(state: AnalysisState) -> Command:
    """Build and export final report."""
    assessments = [ImpactAssessment.model_validate(a) for a in state["impact_assessments"]]

    patch_sets: list[PatchSet] = []
    for p in state["patches"]:
        if isinstance(p, str):
            patch_sets.append(PatchSet.model_validate_json(p))
        else:
            patch_sets.append(PatchSet.model_validate(p))

    analysis_report = report.build_report(
        project_path=state["project_path"],
        assessments=assessments,
        patches=patch_sets,
        errors=state["errors"],
        total_dependencies=state["total_dependencies"],
    )

    json_str = report.export_json(analysis_report)
    return Command(goto=END, update={"report": json_str})


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def _build_dep_worker_graph() -> StateGraph:
    """Build the per-dependency worker StateGraph (not compiled)."""
    builder = StateGraph(DepAnalysisState)

    builder.add_node("check_cache", check_cache_node)
    builder.add_node("fetch_changelog", fetch_changelog_node)
    builder.add_node("embed_changelog", embed_changelog_node)
    builder.add_node("rag_analyze", rag_analyze_node)
    builder.add_node("parse_code", parse_code_node)
    builder.add_node("assess_impact", assess_impact_node)

    builder.set_entry_point("check_cache")

    return builder


def _build_dep_worker_compiled() -> Any:
    """Build and compile the per-dependency worker subgraph."""
    return _build_dep_worker_graph().compile()


def build_analysis_graph() -> Any:
    """Build and compile the parent analysis StateGraph."""
    builder = StateGraph(AnalysisState)

    # Build the worker subgraph for per-dependency analysis.
    dep_worker = _build_dep_worker_compiled()

    builder.add_node("scan_dependencies", scan_dependencies_node)
    builder.add_node("parse_all_code", parse_all_code_node)
    builder.add_node("fan_out", fan_out_node)
    builder.add_node("analyze_dep", dep_worker)
    builder.add_node("cleanup_embeddings", cleanup_embeddings_node)
    builder.add_node("route_results", route_after_fan_in)
    builder.add_node("generate_patches", generate_patches_node)
    builder.add_node("generate_report", generate_report_node)

    builder.set_entry_point("scan_dependencies")

    builder.add_edge("scan_dependencies", "parse_all_code")
    builder.add_edge("parse_all_code", "fan_out")
    builder.add_conditional_edges("fan_out", fan_out_deps, ["analyze_dep"])
    builder.add_edge("analyze_dep", "cleanup_embeddings")
    builder.add_edge("cleanup_embeddings", "route_results")

    builder.add_edge("generate_patches", "generate_report")
    builder.add_edge("generate_report", END)

    return builder.compile()


async def _preflight_api_check() -> None:
    """Verify the OpenAI API key and connectivity before running the full pipeline.

    Makes one cheap embedding call. Fatal errors (auth, connection) cause an
    immediate exit with a clear message. Transient errors (rate limit, 500) are
    ignored — individual nodes handle those with degraded assessments.
    """
    if settings.use_local_llm:
        return
    try:
        await llm.get_embedding("preflight check")
    except AuthenticationError:
        logger.error("OpenAI API key is invalid — check MIGRATOWL_OPENAI_API_KEY")
        sys.exit(1)
    except APIConnectionError:
        logger.error("Cannot connect to OpenAI API — check your network or MIGRATOWL_OPENAI_API_KEY")
        sys.exit(1)
    except (openai.APIError, httpx.RequestError):
        # Transient errors (rate limit, server error) — don't block analysis
        logger.debug("Pre-flight API check failed with transient error, continuing", exc_info=True)


async def analyze(project_path: str, fix_mode: bool = False, ignored_dependencies: list[str] | None = None) -> str:
    """Run the full analysis pipeline and return report JSON."""
    await _preflight_api_check()
    graph = build_analysis_graph()

    merged_ignored = list(settings.parsed_ignored_dependencies)
    if ignored_dependencies:
        merged_ignored.extend(ignored_dependencies)

    initial_state: AnalysisState = {
        "project_path": project_path,
        "fix_mode": fix_mode,
        "total_dependencies": 0,
        "dependencies": [],
        "all_code_usages": [],
        "impact_assessments": [],
        "patches": [],
        "report": "",
        "errors": [],
        "ignored_dependencies": merged_ignored,
    }

    result = await graph.ainvoke(initial_state)
    report: str = result["report"]
    return report
