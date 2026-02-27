"""LangGraph StateGraph — THE agent, all orchestration."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph
from langgraph.types import Command, Send

from migratowl.config import settings
from migratowl.core import changelog, code_parser, impact, patcher, rag, registry, scanner

# Import report as a module reference so tests can patch it.
from migratowl.core import report as report  # noqa: PLW0127
from migratowl.models.schemas import (
    AnalysisState,
    BreakingChange,
    CodeUsage,
    DepAnalysisState,
    ImpactAssessment,
    PatchSet,
)

MAX_RAG_RETRIES = 3
CONFIDENCE_THRESHOLD = settings.confidence_threshold


# ---------------------------------------------------------------------------
# Per-dependency worker nodes (DepAnalysisState)
# ---------------------------------------------------------------------------


async def fetch_changelog_node(state: DepAnalysisState) -> Command:
    """Fetch changelog text for a dependency."""
    text, warnings = await changelog.fetch_changelog(
        changelog_url=state["changelog_url"] or None,
        repository_url=state["repository_url"] or None,
        dep_name=state["dep_name"],
    )
    update: dict = {"changelog": text}
    if warnings:
        update["warnings"] = warnings
    return Command(goto="embed_changelog", update=update)


async def embed_changelog_node(state: DepAnalysisState) -> Command:
    """Chunk and embed changelog into ChromaDB."""
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

    await rag.embed_changelog(dep_name, chunks)
    update: dict = {}
    if warnings:
        update["warnings"] = warnings
    return Command(goto="rag_analyze", update=update)


async def rag_analyze_node(state: DepAnalysisState) -> Command:
    """Query RAG for breaking changes; route based on confidence."""
    query_text = (
        f"breaking changes in {state['dep_name']} between {state['current_version']} and {state['latest_version']}"
    )
    try:
        result = await rag.query(query_text, state["dep_name"])
    except Exception:
        # LLM validation failures (e.g. small models producing bad JSON) — continue with empty results
        return Command(
            goto="parse_code",
            update={"rag_results": [], "rag_confidence": 0.0},
        )

    rag_results = [bc.model_dump() for bc in result.breaking_changes]

    if result.confidence < CONFIDENCE_THRESHOLD and state["retry_count"] < MAX_RAG_RETRIES:
        return Command(
            goto="refine_query",
            update={
                "rag_results": rag_results,
                "rag_confidence": result.confidence,
            },
        )

    return Command(
        goto="parse_code",
        update={
            "rag_results": rag_results,
            "rag_confidence": result.confidence,
        },
    )


async def refine_query_node(state: DepAnalysisState) -> Command:
    """Increment retry and loop back to rag_analyze."""
    return Command(
        goto="rag_analyze",
        update={"retry_count": state["retry_count"] + 1},
    )


async def parse_code_node(state: DepAnalysisState) -> Command:
    """Find code usages of the dependency in the project."""
    usages = await code_parser.find_usages(state["project_path"], state["dep_name"])
    return Command(
        goto="assess_impact",
        update={"code_usages": [u.model_dump() for u in usages]},
    )


async def assess_impact_node(state: DepAnalysisState) -> Command:
    """Assess impact of breaking changes on code usages."""
    breaking_changes = [BreakingChange.model_validate(bc) for bc in state["rag_results"]]
    code_usages = [CodeUsage.model_validate(cu) for cu in state["code_usages"]]

    node_warnings: list[str] = []
    if not code_usages:
        node_warnings.append(f"No usages of {state['dep_name']} found in project code")

    assessment = await impact.assess_impact(
        dep_name=state["dep_name"],
        current_version=state["current_version"],
        latest_version=state["latest_version"],
        breaking_changes=breaking_changes,
        code_usages=code_usages,
    )

    all_warnings = state.get("warnings", []) + node_warnings
    assessment_dict = assessment.model_dump()
    assessment_dict["warnings"] = all_warnings

    return Command(goto=END, update={"impact_assessments": [assessment_dict]})


# ---------------------------------------------------------------------------
# Parent graph nodes (AnalysisState)
# ---------------------------------------------------------------------------


async def scan_dependencies_node(state: AnalysisState) -> Command:
    """Scan project and find outdated dependencies."""
    deps = await scanner.scan_project(state["project_path"])
    outdated = await registry.find_outdated(deps)

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

    return Command(goto="fan_out", update={"dependencies": dep_dicts})


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
                "retry_count": 0,
                "code_usages": [],
                "impact": {},
                "impact_assessments": [],
                "warnings": [],
            },
        )
        for dep in state["dependencies"]
    ]


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
    )

    json_str = report.export_json(analysis_report)
    return Command(goto=END, update={"report": json_str})


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def _build_dep_worker_compiled() -> Any:
    """Build and compile the per-dependency worker subgraph."""
    builder = StateGraph(DepAnalysisState)

    builder.add_node("fetch_changelog", fetch_changelog_node)
    builder.add_node("embed_changelog", embed_changelog_node)
    builder.add_node("rag_analyze", rag_analyze_node)
    builder.add_node("refine_query", refine_query_node)
    builder.add_node("parse_code", parse_code_node)
    builder.add_node("assess_impact", assess_impact_node)

    builder.set_entry_point("fetch_changelog")

    return builder.compile()


def build_dep_worker_graph() -> StateGraph:
    """Build the per-dependency worker StateGraph (not compiled)."""
    builder = StateGraph(DepAnalysisState)

    builder.add_node("fetch_changelog", fetch_changelog_node)
    builder.add_node("embed_changelog", embed_changelog_node)
    builder.add_node("rag_analyze", rag_analyze_node)
    builder.add_node("refine_query", refine_query_node)
    builder.add_node("parse_code", parse_code_node)
    builder.add_node("assess_impact", assess_impact_node)

    builder.set_entry_point("fetch_changelog")

    return builder


def build_analysis_graph() -> Any:
    """Build and compile the parent analysis StateGraph."""
    builder = StateGraph(AnalysisState)

    # Build the worker subgraph for per-dependency analysis.
    dep_worker = _build_dep_worker_compiled()

    builder.add_node("scan_dependencies", scan_dependencies_node)
    builder.add_node("fan_out", fan_out_node)
    builder.add_node("analyze_dep", dep_worker)
    builder.add_node("route_results", route_after_fan_in)
    builder.add_node("generate_patches", generate_patches_node)
    builder.add_node("generate_report", generate_report_node)

    builder.set_entry_point("scan_dependencies")

    builder.add_edge("scan_dependencies", "fan_out")
    builder.add_conditional_edges("fan_out", fan_out_deps, ["analyze_dep"])
    builder.add_edge("analyze_dep", "route_results")

    builder.add_edge("generate_patches", "generate_report")
    builder.add_edge("generate_report", END)

    return builder.compile()


async def analyze(project_path: str, fix_mode: bool = False) -> str:
    """Run the full analysis pipeline and return report JSON."""
    graph = build_analysis_graph()

    initial_state: AnalysisState = {
        "project_path": project_path,
        "fix_mode": fix_mode,
        "dependencies": [],
        "impact_assessments": [],
        "patches": [],
        "report": "",
        "errors": [],
    }

    result = await graph.ainvoke(initial_state)
    report: str = result["report"]
    return report
