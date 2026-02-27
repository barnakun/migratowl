"""Tests for migratowl.models.schemas — all Pydantic models and TypedDict states."""

import operator
from typing import get_type_hints

import pytest
from pydantic import ValidationError


class TestEnums:
    def test_ecosystem_values(self) -> None:
        from migratowl.models.schemas import Ecosystem

        assert Ecosystem.PYTHON == "python"
        assert Ecosystem.NODEJS == "nodejs"

    def test_severity_values(self) -> None:
        from migratowl.models.schemas import Severity

        assert Severity.CRITICAL == "critical"
        assert Severity.WARNING == "warning"
        assert Severity.INFO == "info"

    def test_change_type_values(self) -> None:
        from migratowl.models.schemas import ChangeType

        assert ChangeType.REMOVED == "removed"
        assert ChangeType.RENAMED == "renamed"
        assert ChangeType.SIGNATURE_CHANGED == "signature_changed"
        assert ChangeType.BEHAVIOR_CHANGED == "behavior_changed"


class TestDependencyModels:
    def test_dependency_creation(self) -> None:
        from migratowl.models.schemas import Dependency, Ecosystem

        dep = Dependency(
            name="requests", current_version="2.28.0", ecosystem=Ecosystem.PYTHON, manifest_path="requirements.txt"
        )
        assert dep.name == "requests"
        assert dep.ecosystem == Ecosystem.PYTHON

    def test_registry_info_creation(self) -> None:
        from migratowl.models.schemas import RegistryInfo

        info = RegistryInfo(
            name="requests",
            latest_version="2.31.0",
            homepage_url="https://requests.readthedocs.io",
            repository_url="https://github.com/psf/requests",
            changelog_url="https://github.com/psf/requests/blob/main/HISTORY.md",
        )
        assert info.latest_version == "2.31.0"

    def test_registry_info_optional_urls(self) -> None:
        from migratowl.models.schemas import RegistryInfo

        info = RegistryInfo(name="some-pkg", latest_version="1.0.0")
        assert info.homepage_url is None
        assert info.repository_url is None
        assert info.changelog_url is None

    def test_outdated_dependency(self) -> None:
        from migratowl.models.schemas import Ecosystem, OutdatedDependency

        od = OutdatedDependency(
            name="flask",
            current_version="2.3.0",
            latest_version="3.0.0",
            ecosystem=Ecosystem.PYTHON,
            manifest_path="requirements.txt",
        )
        assert od.current_version == "2.3.0"
        assert od.latest_version == "3.0.0"


class TestLLMResponseModels:
    def test_breaking_change(self) -> None:
        from migratowl.models.schemas import BreakingChange, ChangeType

        bc = BreakingChange(
            api_name="some_func",
            change_type=ChangeType.REMOVED,
            description="Function removed",
            migration_hint="Use new_func instead",
        )
        assert bc.api_name == "some_func"

    def test_changelog_analysis(self) -> None:
        from migratowl.models.schemas import BreakingChange, ChangelogAnalysis, ChangeType

        ca = ChangelogAnalysis(
            breaking_changes=[
                BreakingChange(api_name="f", change_type=ChangeType.REMOVED, description="gone", migration_hint="use g")
            ],
            deprecations=["old_func"],
            new_features=["new_func"],
            confidence=0.9,
        )
        assert len(ca.breaking_changes) == 1
        assert ca.confidence == 0.9

    def test_changelog_analysis_confidence_bounds(self) -> None:
        from migratowl.models.schemas import ChangelogAnalysis

        with pytest.raises(ValidationError):
            ChangelogAnalysis(breaking_changes=[], deprecations=[], new_features=[], confidence=1.5)

        with pytest.raises(ValidationError):
            ChangelogAnalysis(breaking_changes=[], deprecations=[], new_features=[], confidence=-0.1)

    def test_rag_query_result(self) -> None:
        from migratowl.models.schemas import RAGQueryResult

        r = RAGQueryResult(breaking_changes=[], confidence=0.5, source_chunks=["chunk1"])
        assert r.confidence == 0.5

    def test_impact_item(self) -> None:
        from migratowl.models.schemas import ImpactItem, Severity

        item = ImpactItem(
            breaking_change="removed func",
            affected_usages=["main.py:10"],
            severity=Severity.CRITICAL,
            explanation="This function was removed",
            suggested_fix="Use replacement",
        )
        assert item.severity == Severity.CRITICAL

    def test_impact_assessment(self) -> None:
        from migratowl.models.schemas import ImpactAssessment, Severity

        ia = ImpactAssessment(
            dep_name="requests",
            versions={"current": "2.28.0", "latest": "2.31.0"},
            impacts=[],
            summary="No breaking changes",
            overall_severity=Severity.INFO,
        )
        assert ia.dep_name == "requests"

    def test_patch_suggestion(self) -> None:
        from migratowl.models.schemas import PatchSuggestion

        ps = PatchSuggestion(
            file_path="main.py",
            original_code="import old",
            patched_code="import new",
            explanation="Renamed module",
        )
        assert ps.file_path == "main.py"

    def test_patch_set(self) -> None:
        from migratowl.models.schemas import PatchSet, PatchSuggestion

        patch = PatchSuggestion(file_path="a.py", original_code="x", patched_code="y", explanation="z")
        ps = PatchSet(dep_name="pkg", patches=[patch], unified_diff="--- a\n+++ b\n")
        assert len(ps.patches) == 1


class TestOtherModels:
    def test_code_usage(self) -> None:
        from migratowl.models.schemas import CodeUsage

        cu = CodeUsage(
            file_path="main.py",
            line_number=10,
            usage_type="import",
            symbol="requests",
            code_snippet="import requests",
        )
        assert cu.line_number == 10

    def test_analysis_report(self) -> None:
        from migratowl.models.schemas import AnalysisReport

        report = AnalysisReport(
            project_path="/tmp/project",
            timestamp="2026-01-01T00:00:00",
            total_dependencies=10,
            outdated_count=3,
            critical_count=1,
            assessments=[],
            patches=[],
            errors=[],
        )
        assert report.outdated_count == 3


class TestTypedDictStates:
    def test_analysis_state_as_dict(self) -> None:
        from migratowl.models.schemas import AnalysisState

        state: AnalysisState = {
            "project_path": "/tmp/project",
            "fix_mode": False,
            "dependencies": [],
            "impact_assessments": [],
            "patches": [],
            "report": "",
            "errors": [],
        }
        assert state["project_path"] == "/tmp/project"

    def test_dep_analysis_state_as_dict(self) -> None:
        from migratowl.models.schemas import DepAnalysisState

        state: DepAnalysisState = { # type: ignore[arg-type]
            "dep_name": "requests",
            "current_version": "2.28.0",
            "latest_version": "2.31.0",
            "project_path": "/tmp",
            "changelog_url": "",
            "repository_url": "",
            "changelog": "",
            "rag_results": [],
            "rag_confidence": 0.0,
            "retry_count": 0,
            "code_usages": [],
            "impact": {},
        }
        assert state["dep_name"] == "requests"

    def test_analysis_state_has_add_reducers(self) -> None:
        from migratowl.models.schemas import AnalysisState

        hints = get_type_hints(AnalysisState, include_extras=True)
        # impact_assessments and errors should have Annotated with operator.add
        ia_meta = hints["impact_assessments"].__metadata__
        assert operator.add in ia_meta
        errors_meta = hints["errors"].__metadata__
        assert operator.add in errors_meta


class TestChangelogAnalysisRobustness:
    """ChangelogAnalysis must tolerate imperfect LLM output (e.g. from llama3.2)."""

    def test_unknown_change_type_coerced_to_behavior_changed(self) -> None:
        """Unknown change_type values from the LLM must be coerced to behavior_changed."""
        from migratowl.models.schemas import BreakingChange, ChangeType

        bc = BreakingChange(
            api_name="RequestContext",
            change_type="DeprecatedAlias",  # type: ignore[arg-type]
            description="Has been deprecated",
            migration_hint="Use AppContext",
        )
        assert bc.change_type == ChangeType.BEHAVIOR_CHANGED

    def test_dict_items_in_deprecations_coerced_to_strings(self) -> None:
        """list[dict] in deprecations (llama3.2 habit) must be coerced to list[str]."""
        from migratowl.models.schemas import ChangelogAnalysis

        ca = ChangelogAnalysis(
            breaking_changes=[],
            deprecations=[{"comment": "Use new API instead"}],  # type: ignore[list-item]
            new_features=[],
            confidence=0.8,
        )
        assert len(ca.deprecations) == 1
        assert isinstance(ca.deprecations[0], str)

    def test_dict_items_in_new_features_coerced_to_strings(self) -> None:
        """list[dict] in new_features (llama3.2 habit) must be coerced to list[str]."""
        from migratowl.models.schemas import ChangelogAnalysis

        ca = ChangelogAnalysis(
            breaking_changes=[],
            deprecations=[],
            new_features=[{"comment": "Added async support"}],  # type: ignore[list-item]
            confidence=0.7,
        )
        assert len(ca.new_features) == 1
        assert isinstance(ca.new_features[0], str)


class TestWarningsFeature:
    def test_impact_assessment_has_warnings_field(self) -> None:
        from migratowl.models.schemas import ImpactAssessment, Severity

        ia = ImpactAssessment(
            dep_name="requests",
            versions={"current": "2.28.0", "latest": "2.31.0"},
            impacts=[],
            summary="No breaking changes",
            overall_severity=Severity.INFO,
        )
        assert hasattr(ia, "warnings")
        assert ia.warnings == []

    def test_impact_assessment_warnings_can_be_populated(self) -> None:
        from migratowl.models.schemas import ImpactAssessment, Severity

        ia = ImpactAssessment(
            dep_name="requests",
            versions={},
            impacts=[],
            summary="",
            overall_severity=Severity.INFO,
            warnings=["No changelog found", "No code usages found"],
        )
        assert ia.warnings == ["No changelog found", "No code usages found"]

    def test_dep_analysis_state_has_warnings_field_with_add_reducer(self) -> None:
        import operator
        from typing import get_type_hints

        from migratowl.models.schemas import DepAnalysisState

        hints = get_type_hints(DepAnalysisState, include_extras=True)
        assert "warnings" in hints
        meta = hints["warnings"].__metadata__
        assert operator.add in meta


class TestSerialization:
    def test_dependency_roundtrip(self) -> None:
        from migratowl.models.schemas import Dependency, Ecosystem

        dep = Dependency(name="flask", current_version="2.3.0", ecosystem=Ecosystem.PYTHON, manifest_path="req.txt")
        data = dep.model_dump()
        dep2 = Dependency.model_validate(data)
        assert dep == dep2

    def test_changelog_analysis_json_roundtrip(self) -> None:
        from migratowl.models.schemas import ChangelogAnalysis

        ca = ChangelogAnalysis(breaking_changes=[], deprecations=[], new_features=[], confidence=0.75)
        json_str = ca.model_dump_json()
        ca2 = ChangelogAnalysis.model_validate_json(json_str)
        assert ca == ca2
