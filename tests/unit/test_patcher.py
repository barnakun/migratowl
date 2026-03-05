"""Tests for the patcher module."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from migratowl.core.patcher import (
    _build_impacts_context,
    _is_code_patch,
    _is_comment_only_change,
    _parse_file_line_ref,
    _read_code_context,
    _strip_line_comments,
    _validate_patch_against_file,
    create_unified_diff,
    generate_patches,
)
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

    def test_create_unified_diff_no_trailing_newline(self) -> None:
        """Lines without trailing newlines must still produce valid diffs."""
        original = "old_call()"
        patched = "new_call()"

        diff = create_unified_diff("src/app.py", original, patched)

        # Each diff line must be on its own line, not concatenated
        lines = diff.splitlines()
        minus_lines = [l for l in lines if l.startswith("-") and not l.startswith("---")]
        plus_lines = [l for l in lines if l.startswith("+") and not l.startswith("+++")]
        assert minus_lines == ["-old_call()"]
        assert plus_lines == ["+new_call()"]


class TestGeneratePatchesMockedLLM:
    @pytest.mark.asyncio
    async def test_generate_patches_with_mocked_llm(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        real_file = tmp_path / "app.py"
        real_file.write_text("import old_func\n")

        mock_patch_set = PatchSet(
            dep_name="requests",
            patches=[
                PatchSuggestion(
                    file_path=str(real_file),
                    original_code="import old_func",
                    patched_code="import new_func",
                    explanation="Renamed in v2.0",
                )
            ],
            unified_diff="",
        )

        mock_instructor_client = AsyncMock()
        mock_instructor_client.chat.completions.create = AsyncMock(return_value=mock_patch_set)

        with (
            patch("migratowl.core.patcher.get_client", return_value=mock_instructor_client),
            patch("migratowl.core.patcher.get_llm_semaphore", return_value=asyncio.Semaphore(1)),
        ):
            mock_create = mock_instructor_client.chat.completions.create
            assessments = [_make_assessment()]
            result = await generate_patches(assessments, str(tmp_path))

            mock_create.assert_called_once()
            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["response_model"] is PatchSet
            assert call_kwargs["max_retries"] == 2  # settings.max_retries default

            assert len(result) == 1
            assert isinstance(result[0], PatchSet)
            assert result[0].dep_name == "requests"


class TestUnifiedDiffPopulated:
    @pytest.mark.asyncio
    async def test_generate_patches_populates_unified_diff(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        """After LLM returns a PatchSet, unified_diff must be computed from patches."""
        real_file = tmp_path / "app.py"
        real_file.write_text("import old_func\n")

        mock_patch_set = PatchSet(
            dep_name="requests",
            patches=[
                PatchSuggestion(
                    file_path=str(real_file),
                    original_code="import old_func\n",
                    patched_code="import new_func\n",
                    explanation="Renamed in v2.0",
                )
            ],
            unified_diff="",  # LLM returns empty
        )

        mock_instructor_client = AsyncMock()
        mock_instructor_client.chat.completions.create = AsyncMock(return_value=mock_patch_set)

        with (
            patch("migratowl.core.patcher.get_client", return_value=mock_instructor_client),
            patch("migratowl.core.patcher.get_llm_semaphore", return_value=asyncio.Semaphore(1)),
        ):
            assessments = [_make_assessment()]
            result = await generate_patches(assessments, str(tmp_path))

            assert len(result) == 1
            assert result[0].unified_diff != ""
            assert "-import old_func" in result[0].unified_diff
            assert "+import new_func" in result[0].unified_diff

    @pytest.mark.asyncio
    async def test_generate_patches_skips_empty_code(self) -> None:
        """Patches with empty original/patched code should not produce diff."""
        mock_patch_set = PatchSet(
            dep_name="requests",
            patches=[
                PatchSuggestion(
                    file_path="src/app.py",
                    original_code="",
                    patched_code="",
                    explanation="No code change",
                )
            ],
            unified_diff="",
        )

        mock_instructor_client = AsyncMock()
        mock_instructor_client.chat.completions.create = AsyncMock(return_value=mock_patch_set)

        with (
            patch("migratowl.core.patcher.get_client", return_value=mock_instructor_client),
            patch("migratowl.core.patcher.get_llm_semaphore", return_value=asyncio.Semaphore(1)),
        ):
            assessments = [_make_assessment()]
            result = await generate_patches(assessments, "/tmp/project")

            assert result[0].unified_diff == ""


class TestPatchFiltering:
    def test_is_code_patch_rejects_comment_only(self) -> None:
        p = PatchSuggestion(
            file_path="src/app.py",
            original_code="# Python 3.7 required\n",
            patched_code="# Python 3.11 required\n",
            explanation="Version bump",
        )
        assert _is_code_patch(p) is False

    def test_is_code_patch_accepts_real_code(self) -> None:
        p = PatchSuggestion(
            file_path="src/app.py",
            original_code="import old_func\n",
            patched_code="import new_func\n",
            explanation="Renamed",
        )
        assert _is_code_patch(p) is True

    def test_is_code_patch_rejects_whitespace_only(self) -> None:
        p = PatchSuggestion(
            file_path="src/app.py",
            original_code="  \n\n",
            patched_code="  \n\n",
            explanation="Whitespace",
        )
        assert _is_code_patch(p) is False

    def test_is_code_patch_accepts_mixed_code_and_comments(self) -> None:
        p = PatchSuggestion(
            file_path="src/app.py",
            original_code="# comment\nimport foo\n",
            patched_code="# comment\nimport bar\n",
            explanation="Renamed",
        )
        assert _is_code_patch(p) is True

    def test_is_code_patch_rejects_js_comment_only(self) -> None:
        p = PatchSuggestion(
            file_path="src/app.js",
            original_code="// old version\n",
            patched_code="// new version\n",
            explanation="Comment update",
        )
        assert _is_code_patch(p) is False

    def test_validate_patch_against_file_rejects_missing_file(self) -> None:
        p = PatchSuggestion(
            file_path="/nonexistent/path/foo.py",
            original_code="import foo\n",
            patched_code="import bar\n",
            explanation="Renamed",
        )
        assert _validate_patch_against_file(p) is False

    def test_validate_patch_against_file_rejects_hallucinated_code(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        f = tmp_path / "app.py"
        f.write_text("import real_module\n")
        p = PatchSuggestion(
            file_path=str(f),
            original_code="import hallucinated_module\n",
            patched_code="import new_module\n",
            explanation="Not real",
        )
        assert _validate_patch_against_file(p) is False

    def test_validate_patch_against_file_accepts_matching_code(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        f = tmp_path / "app.py"
        f.write_text("import real_module\ndo_stuff()\n")
        p = PatchSuggestion(
            file_path=str(f),
            original_code="import real_module",
            patched_code="import new_module",
            explanation="Renamed",
        )
        assert _validate_patch_against_file(p) is True

    def test_validate_patch_resolves_relative_path(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        """Relative file_path + project_path should resolve to find the file."""
        f = tmp_path / "src" / "app.py"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("import old_func\n")
        p = PatchSuggestion(
            file_path="src/app.py",
            original_code="import old_func",
            patched_code="import new_func",
            explanation="Renamed",
        )
        assert _validate_patch_against_file(p, project_path=str(tmp_path)) is True

    def test_validate_patch_absolute_path_still_works(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        """Absolute file_path should work regardless of project_path."""
        f = tmp_path / "app.py"
        f.write_text("import old_func\n")
        p = PatchSuggestion(
            file_path=str(f),
            original_code="import old_func",
            patched_code="import new_func",
            explanation="Renamed",
        )
        assert _validate_patch_against_file(p, project_path="/some/other/path") is True

    def test_noop_patch_filtered(self) -> None:
        """Patch where original_code == patched_code should be rejected as no-op."""
        p = PatchSuggestion(
            file_path="src/app.py",
            original_code="import foo",
            patched_code="import foo",
            explanation="No actual change",
        )
        # No-op: original == patched, so even if it's "code", it's useless
        assert p.original_code.strip() == p.patched_code.strip()

    @pytest.mark.asyncio
    async def test_generate_patches_filters_non_code_patches(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        """End-to-end: LLM returns mix of code and comment patches, only code survives."""
        real_file = tmp_path / "app.py"
        real_file.write_text("import old_func\nresult = old_func(x)\n")

        mock_patch_set = PatchSet(
            dep_name="requests",
            patches=[
                PatchSuggestion(
                    file_path=str(real_file),
                    original_code="import old_func",
                    patched_code="import new_func",
                    explanation="Renamed in v2.0",
                ),
                PatchSuggestion(
                    file_path=str(real_file),
                    original_code="# Python 3.7 required",
                    patched_code="# Python 3.11 required",
                    explanation="Comment update",
                ),
                PatchSuggestion(
                    file_path="/nonexistent/hallucinated.py",
                    original_code="import something",
                    patched_code="import other",
                    explanation="Hallucinated file",
                ),
            ],
            unified_diff="",
        )

        mock_instructor_client = AsyncMock()
        mock_instructor_client.chat.completions.create = AsyncMock(
            return_value=mock_patch_set
        )

        with (
            patch(
                "migratowl.core.patcher.get_client",
                return_value=mock_instructor_client,
            ),
            patch(
                "migratowl.core.patcher.get_llm_semaphore",
                return_value=asyncio.Semaphore(1),
            ),
        ):
            assessments = [_make_assessment()]
            result = await generate_patches(assessments, str(tmp_path))

            assert len(result) == 1
            ps = result[0]
            assert len(ps.patches) == 1
            assert ps.patches[0].original_code == "import old_func"
            assert "--- " in ps.unified_diff
            assert "-import old_func" in ps.unified_diff
            assert "+import new_func" in ps.unified_diff


class TestGeneratePatchesEmpty:
    @pytest.mark.asyncio
    async def test_generate_patches_empty_assessments(self) -> None:
        result = await generate_patches([], "/tmp/project")
        assert result == []


class TestNoopPatchIntegration:
    @pytest.mark.asyncio
    async def test_generate_patches_filters_noop_patches(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        """No-op patches (original == patched) must be dropped by generate_patches."""
        real_file = tmp_path / "app.py"
        real_file.write_text("import foo\ndo_stuff()\n")

        mock_patch_set = PatchSet(
            dep_name="requests",
            patches=[
                PatchSuggestion(
                    file_path=str(real_file),
                    original_code="import foo",
                    patched_code="import foo",
                    explanation="No-op from LLM",
                ),
                PatchSuggestion(
                    file_path=str(real_file),
                    original_code="do_stuff()",
                    patched_code="do_new_stuff()",
                    explanation="Real change",
                ),
            ],
            unified_diff="",
        )

        mock_instructor_client = AsyncMock()
        mock_instructor_client.chat.completions.create = AsyncMock(
            return_value=mock_patch_set
        )

        with (
            patch("migratowl.core.patcher.get_client", return_value=mock_instructor_client),
            patch("migratowl.core.patcher.get_llm_semaphore", return_value=asyncio.Semaphore(1)),
        ):
            assessments = [_make_assessment()]
            result = await generate_patches(assessments, str(tmp_path))

            assert len(result) == 1
            ps = result[0]
            assert len(ps.patches) == 1
            assert ps.patches[0].patched_code == "do_new_stuff()"


class TestLLMRetryFailure:
    @pytest.mark.asyncio
    async def test_generate_patches_survives_llm_retry_failure(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        """When LLM fails for one dep, other deps should still get patches."""
        from instructor.core import InstructorRetryException

        real_file = tmp_path / "app.py"
        real_file.write_text("import old_func\n")

        good_patch_set = PatchSet(
            dep_name="good-dep",
            patches=[
                PatchSuggestion(
                    file_path=str(real_file),
                    original_code="import old_func",
                    patched_code="import new_func",
                    explanation="Renamed",
                )
            ],
            unified_diff="",
        )

        call_count = 0

        async def mock_create(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise InstructorRetryException(
                    Exception("validation failed"),
                    last_completion=None,
                    n_attempts=2,
                    total_usage=0,
                )
            return good_patch_set

        mock_instructor_client = AsyncMock()
        mock_instructor_client.chat.completions.create = AsyncMock(side_effect=mock_create)

        with (
            patch("migratowl.core.patcher.get_client", return_value=mock_instructor_client),
            patch("migratowl.core.patcher.get_llm_semaphore", return_value=asyncio.Semaphore(1)),
        ):
            assessments = [
                _make_assessment(dep_name="bad-dep"),
                _make_assessment(dep_name="good-dep"),
            ]
            result = await generate_patches(assessments, str(tmp_path))

            # bad-dep should produce an empty PatchSet, good-dep should succeed
            assert len(result) == 2
            bad = [r for r in result if r.dep_name == "bad-dep"]
            good = [r for r in result if r.dep_name == "good-dep"]
            assert len(bad) == 1
            assert bad[0].patches == []
            assert len(good) == 1
            assert len(good[0].patches) == 1


class TestCommentOnlyFilter:
    def test_rejects_appended_trailing_comment(self) -> None:
        """code → code # comment should be rejected."""
        p = PatchSuggestion(
            file_path="src/app.py",
            original_code="console = Console()",
            patched_code="console = Console()  # Ensure console uses color",
            explanation="Added comment",
        )
        assert _is_comment_only_change(p) is True

    def test_rejects_added_comment_lines(self) -> None:
        """code → code + comment line should be rejected."""
        p = PatchSuggestion(
            file_path="src/app.py",
            original_code="await close_http_client()",
            patched_code="await close_http_client()\n# Ensure to set necessary env vars",
            explanation="Added comment",
        )
        assert _is_comment_only_change(p) is True

    def test_accepts_real_code_change(self) -> None:
        p = PatchSuggestion(
            file_path="src/app.py",
            original_code="import old_func",
            patched_code="import new_func",
            explanation="Renamed",
        )
        assert _is_comment_only_change(p) is False

    def test_accepts_code_change_with_comment_change(self) -> None:
        p = PatchSuggestion(
            file_path="src/app.py",
            original_code="import old  # old comment",
            patched_code="import new  # new comment",
            explanation="Both changed",
        )
        assert _is_comment_only_change(p) is False

    def test_preserves_hash_in_string_literal(self) -> None:
        """'#' inside a string should not be treated as comment start."""
        p = PatchSuggestion(
            file_path="src/app.py",
            original_code='url = "http://example.com"',
            patched_code='url = "https://example.com"',
            explanation="Changed URL",
        )
        assert _is_comment_only_change(p) is False

    def test_preserves_double_slash_in_js_string(self) -> None:
        """'//' inside a JS string should not be treated as comment start."""
        p = PatchSuggestion(
            file_path="src/app.ts",
            original_code='const url = "http://old.com"',
            patched_code='const url = "http://new.com"',
            explanation="Changed URL",
        )
        assert _is_comment_only_change(p) is False

    def test_accepts_docstring_change(self) -> None:
        """Docstrings (triple-quoted) are real changes, not comment-only."""
        p = PatchSuggestion(
            file_path="src/app.py",
            original_code='def foo():\n    """Old docstring."""\n    pass',
            patched_code='def foo():\n    """New docstring."""\n    pass',
            explanation="Updated docstring",
        )
        assert _is_comment_only_change(p) is False

    def test_rejects_js_trailing_comment(self) -> None:
        """const x = 1; → const x = 1; // note should be rejected."""
        p = PatchSuggestion(
            file_path="src/app.js",
            original_code="const x = 1;",
            patched_code="const x = 1; // important note",
            explanation="Added comment",
        )
        assert _is_comment_only_change(p) is True

    def test_strip_line_comments_basic(self) -> None:
        assert _strip_line_comments("code  # comment", ("#",)).rstrip() == "code"

    def test_strip_line_comments_preserves_hash_in_string(self) -> None:
        assert _strip_line_comments('"http://x" # comment', ("#",)).rstrip() == '"http://x"'

    def test_strip_line_comments_no_comment(self) -> None:
        assert _strip_line_comments("just code", ("#",)) == "just code"

    def test_strip_line_comments_js_double_slash(self) -> None:
        assert _strip_line_comments("code // comment", ("//",)).rstrip() == "code"

    def test_strip_line_comments_js_preserves_string(self) -> None:
        assert _strip_line_comments('"http://x" // comment', ("//",)).rstrip() == '"http://x"'

    @pytest.mark.asyncio
    async def test_generate_patches_filters_comment_only_patches(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        """Full pipeline: comment-only patch filtered, real patch kept."""
        real_file = tmp_path / "app.py"
        real_file.write_text("console = Console()\nimport old_func\n")

        mock_patch_set = PatchSet(
            dep_name="requests",
            patches=[
                PatchSuggestion(
                    file_path=str(real_file),
                    original_code="console = Console()",
                    patched_code="console = Console()  # Ensure color support",
                    explanation="Comment only",
                ),
                PatchSuggestion(
                    file_path=str(real_file),
                    original_code="import old_func",
                    patched_code="import new_func",
                    explanation="Real change",
                ),
            ],
            unified_diff="",
        )

        mock_instructor_client = AsyncMock()
        mock_instructor_client.chat.completions.create = AsyncMock(
            return_value=mock_patch_set
        )

        with (
            patch("migratowl.core.patcher.get_client", return_value=mock_instructor_client),
            patch("migratowl.core.patcher.get_llm_semaphore", return_value=asyncio.Semaphore(1)),
        ):
            assessments = [_make_assessment()]
            result = await generate_patches(assessments, str(tmp_path))

            assert len(result) == 1
            ps = result[0]
            assert len(ps.patches) == 1
            assert ps.patches[0].patched_code == "import new_func"


class TestSnippetHelpers:
    def test_parse_file_line_ref_with_line(self) -> None:
        assert _parse_file_line_ref("src/app.py:42") == ("src/app.py", 42)

    def test_parse_file_line_ref_without_line(self) -> None:
        assert _parse_file_line_ref("src/app.py") == ("src/app.py", None)

    def test_parse_file_line_ref_absolute_path(self) -> None:
        assert _parse_file_line_ref("/abs/path/app.py:10") == ("/abs/path/app.py", 10)

    def test_read_code_context_returns_snippet(self, tmp_path: pytest.TempPathFactory) -> None:
        f = tmp_path / "code.py"
        lines = [f"line {i}\n" for i in range(1, 21)]
        f.write_text("".join(lines))

        snippet = _read_code_context(str(f), 10, context_lines=5)
        assert snippet is not None
        assert "line 10" in snippet
        assert "line 5" in snippet
        assert "line 15" in snippet
        # Should be ~11 lines (10-5=5 through 10+5=15)
        assert len(snippet.strip().splitlines()) == 11

    def test_read_code_context_missing_file(self) -> None:
        assert _read_code_context("/nonexistent/file.py", 10) is None

    def test_read_code_context_clamps_to_file_bounds(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        f = tmp_path / "short.py"
        f.write_text("line 1\nline 2\nline 3\n")

        snippet = _read_code_context(str(f), 1, context_lines=5)
        assert snippet is not None
        assert "line 1" in snippet
        # Should not crash, just return available lines
        assert len(snippet.strip().splitlines()) <= 6

    def test_read_code_context_relative_path(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        """Relative file_path + base_path should resolve to read the file."""
        f = tmp_path / "src" / "app.py"
        f.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"line {i}\n" for i in range(1, 11)]
        f.write_text("".join(lines))

        snippet = _read_code_context("src/app.py", 5, context_lines=2, base_path=str(tmp_path))
        assert snippet is not None
        assert "line 5" in snippet

    def test_build_impacts_context_embeds_snippets(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        f = tmp_path / "app.py"
        f.write_text("import os\nimport old_func\nfrom old_func import helper\nresult = old_func(x)\nprint(result)\n")

        assessment = _make_assessment(
            impacts=[
                ImpactItem(
                    breaking_change="old_func removed",
                    affected_usages=[f"{f}:3"],
                    severity=Severity.CRITICAL,
                    explanation="old_func is used directly",
                    suggested_fix="Replace with new_func",
                )
            ],
        )

        context = _build_impacts_context(assessment)

        # Should contain fenced code block with actual file content
        assert "```" in context
        assert "from old_func import helper" in context
