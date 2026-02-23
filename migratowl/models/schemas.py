"""ALL Pydantic models and TypedDict graph states for MigratOwl."""

from __future__ import annotations

import operator
from enum import StrEnum
from typing import Annotated, TypedDict

from pydantic import BaseModel, Field, field_validator

# --- Enums ---


class Ecosystem(StrEnum):
    PYTHON = "python"
    NODEJS = "nodejs"


class Severity(StrEnum):
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


class ChangeType(StrEnum):
    REMOVED = "removed"
    RENAMED = "renamed"
    SIGNATURE_CHANGED = "signature_changed"
    BEHAVIOR_CHANGED = "behavior_changed"


# --- Data Models ---


class Dependency(BaseModel):
    name: str
    current_version: str
    ecosystem: Ecosystem
    manifest_path: str


class RegistryInfo(BaseModel):
    name: str
    latest_version: str
    homepage_url: str | None = None
    repository_url: str | None = None
    changelog_url: str | None = None


class OutdatedDependency(BaseModel):
    name: str
    current_version: str
    latest_version: str
    ecosystem: Ecosystem
    manifest_path: str
    homepage_url: str | None = None
    repository_url: str | None = None
    changelog_url: str | None = None


# --- LLM Response Models (used with Instructor response_model=) ---


class BreakingChange(BaseModel):
    api_name: str = Field(description="The function/class/method that changed")
    change_type: ChangeType = Field(description="Type of breaking change")
    description: str = Field(description="What changed and why")
    migration_hint: str = Field(description="How to update affected code")

    @field_validator("change_type", mode="before")
    @classmethod
    def coerce_change_type(cls, v: object) -> object:
        """Coerce unknown change_type values from small LLMs to behavior_changed."""
        valid = {ct.value for ct in ChangeType}
        if isinstance(v, str) and v not in valid:
            return ChangeType.BEHAVIOR_CHANGED
        return v


class ChangelogAnalysis(BaseModel):
    breaking_changes: list[BreakingChange] = Field(default_factory=list)
    deprecations: list[str] = Field(default_factory=list)
    new_features: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score 0-1")

    @field_validator("deprecations", "new_features", mode="before")
    @classmethod
    def coerce_string_list(cls, v: object) -> object:
        """Coerce list[dict] to list[str] — small LLMs often wrap strings in objects."""
        if not isinstance(v, list):
            return v
        result: list[str] = []
        for item in v:
            if isinstance(item, str):
                result.append(item)
            elif isinstance(item, dict):
                # Extract the first string value found, or stringify the dict
                for val in item.values():
                    if isinstance(val, str):
                        result.append(val)
                        break
                else:
                    result.append(str(item))
        return result


class RAGQueryResult(BaseModel):
    breaking_changes: list[BreakingChange] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    source_chunks: list[str] = Field(default_factory=list)


class ImpactItem(BaseModel):
    breaking_change: str
    affected_usages: list[str] = Field(default_factory=list)
    severity: Severity
    explanation: str
    suggested_fix: str


class ImpactAssessment(BaseModel):
    dep_name: str
    versions: dict[str, str] = Field(default_factory=dict)
    impacts: list[ImpactItem] = Field(default_factory=list)
    summary: str
    overall_severity: Severity
    warnings: list[str] = Field(default_factory=list)


class PatchSuggestion(BaseModel):
    file_path: str
    original_code: str
    patched_code: str
    explanation: str


class PatchSet(BaseModel):
    dep_name: str
    patches: list[PatchSuggestion] = Field(default_factory=list)
    unified_diff: str = ""


# --- Other Models ---


class CodeUsage(BaseModel):
    file_path: str
    line_number: int
    usage_type: str
    symbol: str
    code_snippet: str


class AnalysisReport(BaseModel):
    project_path: str
    timestamp: str
    total_dependencies: int
    outdated_count: int
    critical_count: int
    assessments: list[ImpactAssessment] = Field(default_factory=list)
    patches: list[PatchSet] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


# --- TypedDict Graph States ---


class AnalysisState(TypedDict):
    project_path: Annotated[str, lambda _old, new: new]
    fix_mode: bool
    dependencies: list[dict]
    impact_assessments: Annotated[list[dict], operator.add]
    patches: list[str]
    report: str
    errors: Annotated[list[str], operator.add]


class DepAnalysisState(TypedDict):
    dep_name: str
    current_version: str
    latest_version: str
    project_path: str
    changelog_url: str
    repository_url: str
    changelog: str
    rag_results: list[dict]
    rag_confidence: float
    retry_count: int
    code_usages: list[dict]
    impact: dict
    impact_assessments: list[dict]
    warnings: Annotated[list[str], operator.add]
