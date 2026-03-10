"""Tests for impact assessment."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from migratowl.core.impact import _build_impact_context, assess_impact
from migratowl.models.schemas import (
    BreakingChange,
    ChangelogAnalysis,
    ChangeType,
    CodeUsage,
    ImpactAssessment,
    ImpactItem,
    PatchSet,
    RAGQueryResult,
    Severity,
)


def _make_breaking_change(**kwargs) -> BreakingChange:
    defaults = {
        "api_name": "old_func",
        "change_type": ChangeType.REMOVED,
        "description": "old_func was removed in v2.0",
        "migration_hint": "Use new_func instead",
    }
    defaults.update(kwargs)
    return BreakingChange(**defaults)


def _make_code_usage(**kwargs) -> CodeUsage:
    defaults = {
        "file_path": "src/app.py",
        "line_number": 42,
        "usage_type": "function_call",
        "symbol": "old_func",
        "code_snippet": "result = old_func(x)",
    }
    defaults.update(kwargs)
    return CodeUsage(**defaults)


class TestAssessImpactNoBreakingChanges:
    @pytest.mark.asyncio
    async def test_no_breaking_changes_returns_info(self) -> None:
        result = await assess_impact(
            dep_name="requests",
            current_version="1.0.0",
            latest_version="2.0.0",
            breaking_changes=[],
            code_usages=[_make_code_usage()],
        )
        assert isinstance(result, ImpactAssessment)
        assert result.overall_severity == Severity.INFO
        assert result.dep_name == "requests"
        assert result.impacts == []


class TestAssessImpactNoUsages:
    @pytest.mark.asyncio
    async def test_no_usages_returns_info(self) -> None:
        result = await assess_impact(
            dep_name="requests",
            current_version="1.0.0",
            latest_version="2.0.0",
            breaking_changes=[_make_breaking_change()],
            code_usages=[],
        )
        assert isinstance(result, ImpactAssessment)
        assert result.overall_severity == Severity.INFO
        assert result.dep_name == "requests"
        assert result.impacts == []


class TestAssessImpactCallsLLM:
    @pytest.mark.asyncio
    async def test_assess_impact_calls_llm(self) -> None:
        mock_response = ImpactAssessment(
            dep_name="requests",
            versions={"current": "1.0.0", "latest": "2.0.0"},
            impacts=[
                ImpactItem(
                    breaking_change="old_func removed",
                    affected_usages=["src/app.py:42"],
                    severity=Severity.CRITICAL,
                    explanation="old_func is used directly",
                    suggested_fix="Replace with new_func",
                )
            ],
            summary="1 critical impact found",
            overall_severity=Severity.CRITICAL,
        )

        mock_structured_llm = AsyncMock(return_value=mock_response)
        mock_structured_llm.ainvoke = AsyncMock(return_value=mock_response)

        with patch("migratowl.core.impact.get_structured_llm", return_value=mock_structured_llm):
            result = await assess_impact(
                dep_name="requests",
                current_version="1.0.0",
                latest_version="2.0.0",
                breaking_changes=[_make_breaking_change()],
                code_usages=[_make_code_usage()],
            )

            assert result.overall_severity == Severity.CRITICAL
            assert len(result.impacts) == 1


class TestAssessImpactVersionsPopulated:
    @pytest.mark.asyncio
    async def test_assess_impact_always_sets_versions_from_args(self) -> None:
        """LLM often omits 'versions'; assess_impact must always populate it from its own args."""
        mock_response = ImpactAssessment(
            dep_name="requests",
            versions={},  # LLM returned no versions
            impacts=[],
            summary="test",
            overall_severity=Severity.INFO,
        )

        mock_structured_llm = AsyncMock(return_value=mock_response)
        mock_structured_llm.ainvoke = AsyncMock(return_value=mock_response)

        with patch("migratowl.core.impact.get_structured_llm", return_value=mock_structured_llm):
            result = await assess_impact(
                dep_name="requests",
                current_version="1.0.0",
                latest_version="2.0.0",
                breaking_changes=[_make_breaking_change()],
                code_usages=[_make_code_usage()],
            )

        assert result.versions == {"current": "1.0.0", "latest": "2.0.0"}


class TestBuildImpactContext:
    def test_build_impact_context(self) -> None:
        breaking_changes = [
            _make_breaking_change(
                api_name="old_func",
                description="old_func was removed",
                migration_hint="Use new_func",
            ),
            _make_breaking_change(
                api_name="parse",
                change_type=ChangeType.SIGNATURE_CHANGED,
                description="parse() signature changed",
                migration_hint="Add encoding param",
            ),
        ]
        code_usages = [
            _make_code_usage(
                file_path="src/app.py",
                line_number=42,
                symbol="old_func",
                code_snippet="result = old_func(x)",
            ),
            _make_code_usage(
                file_path="src/utils.py",
                line_number=10,
                symbol="parse",
                code_snippet="data = parse(raw)",
            ),
        ]

        context = _build_impact_context(breaking_changes, code_usages)

        # Verify breaking change info is present
        assert "old_func" in context
        assert "old_func was removed" in context
        assert "Use new_func" in context
        assert "parse" in context
        assert "parse() signature changed" in context

        # Verify code usage info is present
        assert "src/app.py" in context
        assert "42" in context
        assert "result = old_func(x)" in context
        assert "src/utils.py" in context
        assert "data = parse(raw)" in context


class TestAssessImpactLLMSemaphore:
    @pytest.mark.asyncio
    async def test_assess_impact_acquires_llm_semaphore(self) -> None:
        """assess_impact must hold the LLM semaphore while making the LLM call."""
        mock_response = ImpactAssessment(
            dep_name="requests",
            versions={"current": "1.0.0", "latest": "2.0.0"},
            impacts=[],
            summary="test",
            overall_severity=Severity.INFO,
        )

        mock_sem = MagicMock()
        mock_sem.__aenter__ = AsyncMock(return_value=None)
        mock_sem.__aexit__ = AsyncMock(return_value=False)

        mock_structured_llm = AsyncMock(return_value=mock_response)
        mock_structured_llm.ainvoke = AsyncMock(return_value=mock_response)

        with (
            patch("migratowl.core.impact.get_llm_semaphore", return_value=mock_sem),
            patch("migratowl.core.impact.get_structured_llm", return_value=mock_structured_llm),
        ):
            await assess_impact(
                dep_name="requests",
                current_version="1.0.0",
                latest_version="2.0.0",
                breaking_changes=[_make_breaking_change()],
                code_usages=[_make_code_usage()],
            )

        mock_sem.__aenter__.assert_called_once()
        mock_sem.__aexit__.assert_called_once()


class TestNullCoercionValidators:
    """LLMs using function_calling often return null for collection fields.

    These tests verify that None values are coerced to empty collections
    instead of causing Pydantic validation errors.
    """

    def test_impact_item_null_affected_usages(self) -> None:
        item = ImpactItem(
            breaking_change="func removed",
            affected_usages=None,  # type: ignore[arg-type]
            severity=Severity.CRITICAL,
            explanation="removed",
            suggested_fix="use new_func",
        )
        assert item.affected_usages == []

    def test_impact_assessment_null_collections(self) -> None:
        assessment = ImpactAssessment(
            dep_name="requests",
            versions=None,  # type: ignore[arg-type]
            impacts=None,  # type: ignore[arg-type]
            summary="test",
            overall_severity=Severity.INFO,
            warnings=None,  # type: ignore[arg-type]
            errors=None,  # type: ignore[arg-type]
        )
        assert assessment.versions == {}
        assert assessment.impacts == []
        assert assessment.warnings == []
        assert assessment.errors == []

    def test_changelog_analysis_null_collections(self) -> None:
        analysis = ChangelogAnalysis(
            breaking_changes=None,  # type: ignore[arg-type]
            deprecations=None,  # type: ignore[arg-type]
            new_features=None,  # type: ignore[arg-type]
            confidence=0.8,
        )
        assert analysis.breaking_changes == []
        assert analysis.deprecations == []
        assert analysis.new_features == []

    def test_rag_query_result_null_collections(self) -> None:
        result = RAGQueryResult(
            breaking_changes=None,  # type: ignore[arg-type]
            confidence=0.5,
            source_chunks=None,  # type: ignore[arg-type]
        )
        assert result.breaking_changes == []
        assert result.source_chunks == []

    def test_patch_set_null_patches(self) -> None:
        ps = PatchSet(
            dep_name="requests",
            patches=None,  # type: ignore[arg-type]
        )
        assert ps.patches == []

    def test_non_null_values_pass_through(self) -> None:
        """Ensure validators don't interfere with valid non-null values."""
        item = ImpactItem(
            breaking_change="func removed",
            affected_usages=["src/app.py:10"],
            severity=Severity.CRITICAL,
            explanation="removed",
            suggested_fix="use new_func",
        )
        assert item.affected_usages == ["src/app.py:10"]
