"""Tests for the patcher module."""

from unittest.mock import AsyncMock, patch

import pytest

from migratowl.core.patcher import create_unified_diff, generate_patches
from migratowl.models.schemas import (
    ImpactAssessment,
    ImpactItem,
    PatchSet,
    PatchSuggestion,
    Severity,
)


def _make_assessment(**kwargs) -> ImpactAssessment:
    defaults = {
        "dep_name": "requests",
        "versions": {"current": "1.0.0", "latest": "2.0.0"},
        "impacts": [
            ImpactItem(
                breaking_change="old_func removed",
                affected_usages=["src/app.py:42"],
                severity=Severity.CRITICAL,
                explanation="old_func is used directly",
                suggested_fix="Replace with new_func",
            )
        ],
        "summary": "1 critical impact found",
        "overall_severity": Severity.CRITICAL,
    }
    defaults.update(kwargs)
    return ImpactAssessment(**defaults)


class TestCreateUnifiedDiff:
    def test_create_unified_diff_output_format(self) -> None:
        original = "import old_func\nresult = old_func(x)\n"
        patched = "import new_func\nresult = new_func(x)\n"

        diff = create_unified_diff("src/app.py", original, patched)

        assert "--- src/app.py" in diff
        assert "+++ src/app.py" in diff
        assert "-import old_func" in diff
        assert "+import new_func" in diff
        assert "-result = old_func(x)" in diff
        assert "+result = new_func(x)" in diff


class TestGeneratePatchesMockedLLM:
    @pytest.mark.asyncio
    async def test_generate_patches_with_mocked_llm(self) -> None:
        mock_patch_set = PatchSet(
            dep_name="requests",
            patches=[
                PatchSuggestion(
                    file_path="src/app.py",
                    original_code="import old_func",
                    patched_code="import new_func",
                    explanation="Renamed in v2.0",
                )
            ],
            unified_diff="",
        )

        with patch(
            "migratowl.core.patcher.client.chat.completions.create",
            new_callable=AsyncMock,
            return_value=mock_patch_set,
        ) as mock_create:
            assessments = [_make_assessment()]
            result = await generate_patches(assessments, "/tmp/project")

            mock_create.assert_called_once()
            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["response_model"] is PatchSet
            assert call_kwargs["max_retries"] == 2

            assert len(result) == 1
            assert isinstance(result[0], PatchSet)
            assert result[0].dep_name == "requests"


class TestGeneratePatchesEmpty:
    @pytest.mark.asyncio
    async def test_generate_patches_empty_assessments(self) -> None:
        result = await generate_patches([], "/tmp/project")
        assert result == []
