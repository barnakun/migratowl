"""ALL Pydantic models and TypedDict graph states for MigratOwl."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel

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
