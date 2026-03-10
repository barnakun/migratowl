"""Centralized prompt templates for all LLM calls."""

from langchain_core.prompts import ChatPromptTemplate

IMPACT_ASSESSMENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a dependency migration expert. Analyze the breaking changes and code usages "
            "below to assess the impact on the project. Return a structured impact assessment.",
        ),
        (
            "human",
            "Dependency: {dep_name}\nCurrent version: {current_version}\nLatest version: {latest_version}\n\n{context}",
        ),
    ]
)

CHANGELOG_SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a dependency migration expert. Summarize the following changelog text, "
            "keeping only breaking changes, removals, renames, and deprecations. "
            "Discard new features and minor fixes.",
        ),
        ("human", "Summarize this {dep_name} changelog:\n\n{text}"),
    ]
)

CHANGELOG_ANALYSIS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a dependency migration expert. Analyze the following changelog excerpts "
            "and identify breaking changes, deprecations, and new features.",
        ),
        ("human", "Analyze these changelog excerpts for {dep_name}:\n\n{combined_text}"),
    ]
)

PATCH_GENERATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a code migration expert. Given the impact assessment below, "
            "generate concrete code patches that fix the breaking changes. "
            "Return a PatchSet with file-level patches showing original and patched code.\n\n"
            "IMPORTANT CONSTRAINTS:\n"
            "- Only patch executable code (imports, function calls, class definitions, configuration). "
            "Do NOT patch comments, docstrings, or version strings.\n"
            "- The original_code field MUST contain code that actually exists in the file. "
            "Do NOT invent or guess what the file contains.\n"
            "- If you are unsure what code exists in a file, omit that patch entirely.",
        ),
        (
            "human",
            "Project path: {project_path}\nDependency: {dep_name}\n"
            "Current version: {current_version}\nLatest version: {latest_version}\n\n"
            "{impacts_text}",
        ),
    ]
)
