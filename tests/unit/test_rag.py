"""Tests for RAG module with mocked langchain-chroma vector store and LLM."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from migratowl.models.schemas import (
    BreakingChange,
    ChangelogAnalysis,
    ChangeType,
    RAGQueryResult,
)


def _mock_structured_llm(return_value: object) -> RunnableLambda:
    """Create a mock structured LLM that works with the PROMPT | structured_llm chain pattern.

    Uses RunnableLambda so it integrates properly with LangChain's pipe operator.
    """

    async def _afunc(_input: object) -> object:
        return return_value

    return RunnableLambda(func=lambda x: return_value, afunc=_afunc)


class TestGetVectorstore:
    def setup_method(self) -> None:
        from migratowl.core.rag import reset_vectorstore_cache

        reset_vectorstore_cache()

    def test_creates_vectorstore_with_cosine_similarity(self) -> None:
        mock_embeddings = MagicMock()

        with (
            patch("migratowl.core.rag.Chroma") as mock_chroma_cls,
            patch("migratowl.core.rag.get_embeddings", return_value=mock_embeddings),
            patch("migratowl.core.rag.settings") as mock_settings,
        ):
            mock_settings.vectorstore_path = "/tmp/vs"
            mock_settings.embedding_model = "openai:text-embedding-3-small"

            from migratowl.core.rag import get_vectorstore

            vs = get_vectorstore()

            mock_chroma_cls.assert_called_once()
            call_kwargs = mock_chroma_cls.call_args.kwargs
            assert call_kwargs["persist_directory"] == "/tmp/vs"
            assert call_kwargs["embedding_function"] is mock_embeddings
            assert call_kwargs["collection_name"].startswith("changelogs_")
            assert call_kwargs["collection_metadata"] == {"hnsw:space": "cosine"}
            assert vs is mock_chroma_cls.return_value

    def test_different_projects_use_different_collection_names(self) -> None:
        """Two different project paths must produce separate collections."""
        mock_embeddings = MagicMock()

        from migratowl.core.rag import get_vectorstore

        with (
            patch("migratowl.core.rag.Chroma") as mock_chroma_cls,
            patch("migratowl.core.rag.get_embeddings", return_value=mock_embeddings),
            patch("migratowl.core.rag.settings") as mock_settings,
        ):
            mock_settings.vectorstore_path = "/tmp/vs"
            mock_settings.embedding_model = "openai:text-embedding-3-small"

            get_vectorstore(project_path="/projects/app-a")
            name_a = mock_chroma_cls.call_args_list[0].kwargs["collection_name"]

            get_vectorstore(project_path="/projects/app-b")
            name_b = mock_chroma_cls.call_args_list[1].kwargs["collection_name"]

        assert name_a != name_b

    def test_same_project_path_uses_same_collection_name(self) -> None:
        """The same project path must always resolve to the same collection name —
        verified via the caching test (same instance returned)."""
        mock_embeddings = MagicMock()

        from migratowl.core.rag import get_vectorstore

        with (
            patch("migratowl.core.rag.Chroma") as mock_chroma_cls,
            patch("migratowl.core.rag.get_embeddings", return_value=mock_embeddings),
            patch("migratowl.core.rag.settings") as mock_settings,
        ):
            mock_settings.vectorstore_path = "/tmp/vs"
            mock_settings.embedding_model = "openai:text-embedding-3-small"

            vs1 = get_vectorstore(project_path="/projects/my-app")
            vs2 = get_vectorstore(project_path="/projects/my-app")

        # Cached: same instance, Chroma called only once
        assert vs1 is vs2
        mock_chroma_cls.assert_called_once()

    def test_uses_settings_path(self) -> None:
        mock_embeddings = MagicMock()

        with (
            patch("migratowl.core.rag.Chroma") as mock_chroma_cls,
            patch("migratowl.core.rag.get_embeddings", return_value=mock_embeddings),
            patch("migratowl.core.rag.settings") as mock_settings,
        ):
            mock_settings.vectorstore_path = "/tmp/test_vectorstore"
            mock_settings.embedding_model = "openai:text-embedding-3-small"

            from migratowl.core.rag import get_vectorstore

            get_vectorstore()

            assert mock_chroma_cls.call_args.kwargs["persist_directory"] == "/tmp/test_vectorstore"

    def test_caches_instance_per_project_path(self) -> None:
        """Same project_path returns the same Chroma instance (cached)."""
        mock_embeddings = MagicMock()

        from migratowl.core.rag import get_vectorstore

        with (
            patch("migratowl.core.rag.Chroma") as mock_chroma_cls,
            patch("migratowl.core.rag.get_embeddings", return_value=mock_embeddings),
            patch("migratowl.core.rag.settings") as mock_settings,
        ):
            mock_settings.vectorstore_path = "/tmp/vs"
            mock_settings.embedding_model = "openai:text-embedding-3-small"

            vs1 = get_vectorstore(project_path="/projects/app")
            vs2 = get_vectorstore(project_path="/projects/app")

        assert vs1 is vs2
        mock_chroma_cls.assert_called_once()

    def test_different_project_paths_not_cached_together(self) -> None:
        """Different project_paths return different Chroma instances."""
        mock_embeddings = MagicMock()

        from migratowl.core.rag import get_vectorstore

        with (
            patch("migratowl.core.rag.Chroma", side_effect=[MagicMock(), MagicMock()]) as mock_chroma_cls,
            patch("migratowl.core.rag.get_embeddings", return_value=mock_embeddings),
            patch("migratowl.core.rag.settings") as mock_settings,
        ):
            mock_settings.vectorstore_path = "/tmp/vs"
            mock_settings.embedding_model = "openai:text-embedding-3-small"

            vs1 = get_vectorstore(project_path="/projects/app-a")
            vs2 = get_vectorstore(project_path="/projects/app-b")

        assert vs1 is not vs2
        assert mock_chroma_cls.call_count == 2


class TestEmbedChangelog:
    @pytest.mark.asyncio
    async def test_adds_chunks_with_metadata(self) -> None:
        mock_vectorstore = MagicMock()
        mock_vectorstore.aadd_texts = AsyncMock()

        chunks = [
            {"version": "2.0.0", "content": "Breaking change"},
            {"version": "1.0.0", "content": "Initial release"},
        ]

        with patch("migratowl.core.rag.get_vectorstore", return_value=mock_vectorstore):
            from migratowl.core.rag import embed_changelog

            await embed_changelog("requests", chunks)

            assert mock_vectorstore.aadd_texts.call_count == 1

            call_kwargs = mock_vectorstore.aadd_texts.call_args.kwargs
            assert call_kwargs["texts"] == ["Breaking change", "Initial release"]
            assert call_kwargs["metadatas"] == [
                {"dep_name": "requests", "version": "2.0.0"},
                {"dep_name": "requests", "version": "1.0.0"},
            ]
            assert call_kwargs["ids"] == ["requests:2.0.0:0", "requests:1.0.0:0"]

    @pytest.mark.asyncio
    async def test_sub_chunks_oversized_version_sections(self) -> None:
        """Version sections larger than EMBED_CHUNK_CHARS must be split into multiple
        sub-chunks, each stored separately — no information is lost."""
        from migratowl.core.rag import EMBED_CHUNK_CHARS, embed_changelog

        mock_vectorstore = MagicMock()
        mock_vectorstore.aadd_texts = AsyncMock()
        # Content that is exactly 2.5x the chunk size -> should produce 3 sub-chunks
        oversized_content = "x" * (EMBED_CHUNK_CHARS * 2 + EMBED_CHUNK_CHARS // 2)
        chunks = [{"version": "1.0.0", "content": oversized_content}]

        with patch("migratowl.core.rag.get_vectorstore", return_value=mock_vectorstore):
            await embed_changelog("flask", chunks)

        call_kwargs = mock_vectorstore.aadd_texts.call_args.kwargs
        captured_texts = call_kwargs["texts"]
        assert len(captured_texts) == 3
        assert all(len(t) <= EMBED_CHUNK_CHARS for t in captured_texts)
        # Full content is preserved across sub-chunks
        assert "".join(captured_texts) == oversized_content

    @pytest.mark.asyncio
    async def test_small_chunk_not_split(self) -> None:
        """Chunks that fit within EMBED_CHUNK_CHARS must not be split."""
        from migratowl.core.rag import EMBED_CHUNK_CHARS, embed_changelog

        mock_vectorstore = MagicMock()
        mock_vectorstore.aadd_texts = AsyncMock()
        small_content = "x" * (EMBED_CHUNK_CHARS - 1)
        chunks = [{"version": "1.0.0", "content": small_content}]

        with patch("migratowl.core.rag.get_vectorstore", return_value=mock_vectorstore):
            await embed_changelog("flask", chunks)

        mock_vectorstore.aadd_texts.assert_called_once()
        assert len(mock_vectorstore.aadd_texts.call_args.kwargs["texts"]) == 1


class TestQueryNResults:
    @pytest.mark.asyncio
    async def test_none_n_results_defaults_to_settings_max_rag_results(self) -> None:
        """n_results=None must resolve to settings.max_rag_results at runtime."""
        mock_vectorstore = MagicMock()
        mock_vectorstore.asimilarity_search = AsyncMock(
            return_value=[
                Document(page_content="some text", metadata={"dep_name": "flask", "version": "2.0.0"}),
            ]
        )

        mock_analysis = ChangelogAnalysis(breaking_changes=[], deprecations=[], new_features=[], confidence=0.5)
        mock_structured_llm = _mock_structured_llm(mock_analysis)

        with (
            patch("migratowl.core.rag.get_vectorstore", return_value=mock_vectorstore),
            patch("migratowl.core.rag.get_structured_llm", return_value=mock_structured_llm),
            patch("migratowl.core.rag.settings") as mock_settings,
        ):
            mock_settings.max_rag_results = 15
            mock_settings.summarize_threshold = 32_000
            from migratowl.core.rag import query

            await query("breaking changes", "flask", n_results=None)

        call_kwargs = mock_vectorstore.asimilarity_search.call_args.kwargs
        assert call_kwargs["k"] == 15

    @pytest.mark.asyncio
    async def test_explicit_n_results_passed_directly(self) -> None:
        """An explicit int n_results must be passed to similarity_search unchanged."""
        mock_vectorstore = MagicMock()
        mock_vectorstore.asimilarity_search = AsyncMock(
            return_value=[
                Document(page_content="text", metadata={"dep_name": "flask", "version": "2.0.0"}),
            ]
        )

        mock_analysis = ChangelogAnalysis(breaking_changes=[], deprecations=[], new_features=[], confidence=0.5)
        mock_structured_llm = _mock_structured_llm(mock_analysis)

        with (
            patch("migratowl.core.rag.get_vectorstore", return_value=mock_vectorstore),
            patch("migratowl.core.rag.get_structured_llm", return_value=mock_structured_llm),
        ):
            from migratowl.core.rag import query

            await query("breaking changes", "flask", n_results=7)

        call_kwargs = mock_vectorstore.asimilarity_search.call_args.kwargs
        assert call_kwargs["k"] == 7

    def test_default_n_results_uses_settings(self) -> None:
        """Default n_results parameter is None, resolved to settings.max_rag_results at runtime."""
        import inspect

        from migratowl.core.rag import query

        sig = inspect.signature(query)
        assert sig.parameters["n_results"].default is None


class TestQuery:
    @pytest.mark.asyncio
    async def test_returns_rag_query_result(self) -> None:
        mock_vectorstore = MagicMock()
        mock_vectorstore.asimilarity_search = AsyncMock(
            return_value=[
                Document(
                    page_content="Breaking: removed old_func", metadata={"dep_name": "requests", "version": "2.0.0"}
                ),
                Document(page_content="Changed API", metadata={"dep_name": "requests", "version": "3.0.0"}),
            ]
        )

        mock_analysis = ChangelogAnalysis(
            breaking_changes=[
                BreakingChange(
                    api_name="old_func",
                    change_type=ChangeType.REMOVED,
                    description="Removed old_func",
                    migration_hint="Use new_func instead",
                )
            ],
            deprecations=[],
            new_features=[],
            confidence=0.9,
        )

        mock_structured_llm = _mock_structured_llm(mock_analysis)

        with (
            patch("migratowl.core.rag.get_vectorstore", return_value=mock_vectorstore),
            patch("migratowl.core.rag.get_structured_llm", return_value=mock_structured_llm),
        ):
            from migratowl.core.rag import query

            result = await query("breaking changes in requests", "requests")

            assert isinstance(result, RAGQueryResult)
            assert len(result.breaking_changes) == 1
            assert result.breaking_changes[0].api_name == "old_func"
            assert result.confidence == 0.9
            assert len(result.source_chunks) == 2

    @pytest.mark.asyncio
    async def test_query_filters_by_dep_name(self) -> None:
        mock_vectorstore = MagicMock()
        mock_vectorstore.asimilarity_search = AsyncMock(
            return_value=[
                Document(page_content="text", metadata={"dep_name": "flask", "version": "2.0.0"}),
            ]
        )

        mock_analysis = ChangelogAnalysis(breaking_changes=[], deprecations=[], new_features=[], confidence=0.5)
        mock_structured_llm = _mock_structured_llm(mock_analysis)

        with (
            patch("migratowl.core.rag.get_vectorstore", return_value=mock_vectorstore),
            patch("migratowl.core.rag.get_structured_llm", return_value=mock_structured_llm),
        ):
            from migratowl.core.rag import query

            await query("changes", "flask", n_results=3)

            call_kwargs = mock_vectorstore.asimilarity_search.call_args.kwargs
            assert call_kwargs["filter"] == {"dep_name": "flask"}
            assert call_kwargs["k"] == 3


class TestSummarization:
    @pytest.mark.asyncio
    async def test_small_text_skips_summarization(self) -> None:
        """Combined text under the threshold must not trigger a summarization call."""
        mock_vectorstore = MagicMock()
        mock_vectorstore.asimilarity_search = AsyncMock(
            return_value=[
                Document(page_content="short text", metadata={"dep_name": "flask", "version": "2.0.0"}),
            ]
        )
        mock_analysis = ChangelogAnalysis(breaking_changes=[], deprecations=[], new_features=[], confidence=0.5)
        mock_structured_llm = _mock_structured_llm(mock_analysis)

        with (
            patch("migratowl.core.rag.get_vectorstore", return_value=mock_vectorstore),
            patch("migratowl.core.rag.get_structured_llm", return_value=mock_structured_llm),
            patch("migratowl.core.rag._summarize_changelog", new_callable=AsyncMock) as mock_summarize,
        ):
            from migratowl.core.rag import query

            await query("breaking changes", "flask", n_results=3)

        mock_summarize.assert_not_called()

    @pytest.mark.asyncio
    async def test_large_text_triggers_summarization(self) -> None:
        """Combined text over the threshold must be summarized before analysis."""
        from migratowl.config import settings

        large_doc = "x" * (settings.summarize_threshold + 1)
        mock_vectorstore = MagicMock()
        mock_vectorstore.asimilarity_search = AsyncMock(
            return_value=[
                Document(page_content=large_doc, metadata={"dep_name": "requests", "version": "3.0.0"}),
            ]
        )
        mock_analysis = ChangelogAnalysis(breaking_changes=[], deprecations=[], new_features=[], confidence=0.8)
        mock_structured_llm = _mock_structured_llm(mock_analysis)

        with (
            patch("migratowl.core.rag.get_vectorstore", return_value=mock_vectorstore),
            patch("migratowl.core.rag.get_structured_llm", return_value=mock_structured_llm),
            patch(
                "migratowl.core.rag._summarize_changelog", new_callable=AsyncMock, return_value="concise summary"
            ) as mock_summarize,
        ):
            from migratowl.core.rag import query

            await query("breaking changes", "requests", n_results=3)

        mock_summarize.assert_called_once()
        call_args = mock_summarize.call_args
        assert call_args.args[0] == large_doc
        assert call_args.args[1] == "requests"

    @pytest.mark.asyncio
    async def test_summarized_text_passed_to_analysis_llm(self) -> None:
        """When summarization runs, the analysis LLM must receive the summary, not the raw text."""
        from migratowl.config import settings

        large_doc = "y" * (settings.summarize_threshold + 1)
        mock_vectorstore = MagicMock()
        mock_vectorstore.asimilarity_search = AsyncMock(
            return_value=[
                Document(page_content=large_doc, metadata={"dep_name": "django", "version": "5.0.0"}),
            ]
        )
        mock_analysis = ChangelogAnalysis(breaking_changes=[], deprecations=[], new_features=[], confidence=0.7)

        # Track what gets passed to the LLM
        invoke_args: list[object] = []

        async def capture_afunc(input_val: object) -> ChangelogAnalysis:
            invoke_args.append(input_val)
            return mock_analysis

        mock_structured_llm = RunnableLambda(func=lambda x: mock_analysis, afunc=capture_afunc)

        with (
            patch("migratowl.core.rag.get_vectorstore", return_value=mock_vectorstore),
            patch("migratowl.core.rag.get_structured_llm", return_value=mock_structured_llm),
            patch("migratowl.core.rag._summarize_changelog", new_callable=AsyncMock, return_value="summarized content"),
        ):
            from migratowl.core.rag import query

            await query("breaking changes", "django", n_results=3)

        # The chain passes a ChatPromptValue to the structured_llm; verify the summary is used
        assert len(invoke_args) == 1
        input_str = str(invoke_args[0])
        assert "summarized content" in input_str
        assert large_doc not in input_str


class TestQueryLLMSemaphore:
    @pytest.mark.asyncio
    async def test_query_acquires_llm_semaphore(self) -> None:
        """RAG query must hold the LLM semaphore while making the LLM call."""
        mock_vectorstore = MagicMock()
        mock_vectorstore.asimilarity_search = AsyncMock(
            return_value=[
                Document(page_content="some text", metadata={"dep_name": "flask", "version": "2.0.0"}),
            ]
        )

        mock_analysis = ChangelogAnalysis(breaking_changes=[], deprecations=[], new_features=[], confidence=0.5)
        mock_structured_llm = _mock_structured_llm(mock_analysis)

        mock_sem = MagicMock()
        mock_sem.__aenter__ = AsyncMock(return_value=None)
        mock_sem.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("migratowl.core.rag.get_vectorstore", return_value=mock_vectorstore),
            patch("migratowl.core.rag.get_structured_llm", return_value=mock_structured_llm),
            patch("migratowl.core.rag.get_llm_semaphore", return_value=mock_sem),
        ):
            from migratowl.core.rag import query

            await query("breaking changes", "flask", n_results=3)

        mock_sem.__aenter__.assert_called_once()
        mock_sem.__aexit__.assert_called_once()


class TestVerifyBreakingChanges:
    def test_verified_when_api_name_in_source(self) -> None:
        from migratowl.core.rag import verify_breaking_changes

        changes = [
            BreakingChange(
                api_name="old_func", change_type=ChangeType.REMOVED, description="Removed", migration_hint="Use new"
            )
        ]
        result, ratio = verify_breaking_changes(changes, ["REMOVED old_func in v2"])
        assert result[0].verified is True
        assert ratio == 1.0

    def test_unverified_when_api_name_not_in_source(self) -> None:
        from migratowl.core.rag import verify_breaking_changes

        changes = [
            BreakingChange(
                api_name="ghost_func", change_type=ChangeType.REMOVED, description="Gone", migration_hint="N/A"
            )
        ]
        result, ratio = verify_breaking_changes(changes, ["Nothing relevant here"])
        assert result[0].verified is False
        assert ratio == 0.0

    def test_dotted_name_any_part_matches(self) -> None:
        from migratowl.core.rag import verify_breaking_changes

        changes = [
            BreakingChange(
                api_name="SomeClass.do_thing",
                change_type=ChangeType.BEHAVIOR_CHANGED,
                description="Changed",
                migration_hint="Update",
            )
        ]
        result, ratio = verify_breaking_changes(changes, ["SomeClass.do_thing was removed"])
        assert result[0].verified is True

    def test_dotted_name_partial_match_sufficient(self) -> None:
        from migratowl.core.rag import verify_breaking_changes

        changes = [
            BreakingChange(
                api_name="SomeClass.obscure",
                change_type=ChangeType.REMOVED,
                description="Removed",
                migration_hint="N/A",
            )
        ]
        result, ratio = verify_breaking_changes(changes, ["SomeClass was refactored"])
        assert result[0].verified is True

    def test_case_insensitive(self) -> None:
        from migratowl.core.rag import verify_breaking_changes

        changes = [
            BreakingChange(
                api_name="Old_Func", change_type=ChangeType.REMOVED, description="Removed", migration_hint="N/A"
            )
        ]
        result, ratio = verify_breaking_changes(changes, ["REMOVED old_func in v2"])
        assert result[0].verified is True

    def test_mixed_verified_and_unverified(self) -> None:
        from migratowl.core.rag import verify_breaking_changes

        changes = [
            BreakingChange(
                api_name="real_func", change_type=ChangeType.REMOVED, description="Removed", migration_hint="N/A"
            ),
            BreakingChange(
                api_name="ghost_func", change_type=ChangeType.REMOVED, description="Gone", migration_hint="N/A"
            ),
        ]
        result, ratio = verify_breaking_changes(changes, ["real_func was removed"])
        assert result[0].verified is True
        assert result[1].verified is False
        assert ratio == 0.5

    def test_empty_breaking_changes(self) -> None:
        from migratowl.core.rag import verify_breaking_changes

        result, ratio = verify_breaking_changes([], ["some text"])
        assert result == []
        assert ratio == 1.0

    def test_short_name_parts_skipped(self) -> None:
        from migratowl.core.rag import verify_breaking_changes

        changes = [
            BreakingChange(
                api_name="X.do_thing",
                change_type=ChangeType.REMOVED,
                description="Removed",
                migration_hint="N/A",
            )
        ]
        # "X" is too short (<3 chars), but "do_thing" matches
        result, ratio = verify_breaking_changes(changes, ["do_thing was removed"])
        assert result[0].verified is True

    def test_parentheses_stripped(self) -> None:
        from migratowl.core.rag import verify_breaking_changes

        changes = [
            BreakingChange(
                api_name="some_func()",
                change_type=ChangeType.REMOVED,
                description="Removed",
                migration_hint="N/A",
            )
        ]
        result, ratio = verify_breaking_changes(changes, ["some_func was removed"])
        assert result[0].verified is True


class TestPurgeDepEmbeddings:
    def test_deletes_all_docs_for_dep(self) -> None:
        mock_vectorstore = MagicMock()
        mock_vectorstore.get.return_value = {"ids": ["flask:1.0.0:0", "flask:2.0.0:0"]}

        with patch("migratowl.core.rag.get_vectorstore", return_value=mock_vectorstore):
            from migratowl.core.rag import purge_dep_embeddings

            count = purge_dep_embeddings("flask")

        mock_vectorstore.get.assert_called_once_with(where={"dep_name": "flask"})
        mock_vectorstore.delete.assert_called_once_with(ids=["flask:1.0.0:0", "flask:2.0.0:0"])
        assert count == 2

    def test_noop_when_dep_has_no_docs(self) -> None:
        mock_vectorstore = MagicMock()
        mock_vectorstore.get.return_value = {"ids": []}

        with patch("migratowl.core.rag.get_vectorstore", return_value=mock_vectorstore):
            from migratowl.core.rag import purge_dep_embeddings

            count = purge_dep_embeddings("nonexistent")

        mock_vectorstore.delete.assert_not_called()
        assert count == 0

    def test_returns_deleted_count(self) -> None:
        mock_vectorstore = MagicMock()
        mock_vectorstore.get.return_value = {"ids": ["a:1:0", "a:2:0", "a:3:0"]}

        with patch("migratowl.core.rag.get_vectorstore", return_value=mock_vectorstore):
            from migratowl.core.rag import purge_dep_embeddings

            count = purge_dep_embeddings("a", project_path="/tmp/proj")

        assert count == 3


class TestPurgeStalEmbeddings:
    def test_removes_orphaned_deps(self) -> None:
        mock_vectorstore = MagicMock()
        mock_vectorstore.get.return_value = {
            "ids": ["flask:1:0", "requests:1:0", "django:1:0", "django:2:0", "django:3:0"],
            "metadatas": [
                {"dep_name": "flask"},
                {"dep_name": "requests"},
                {"dep_name": "django"},
                {"dep_name": "django"},
                {"dep_name": "django"},
            ],
        }

        with patch("migratowl.core.rag.get_vectorstore", return_value=mock_vectorstore):
            from migratowl.core.rag import purge_stale_embeddings

            with patch("migratowl.core.rag.purge_dep_embeddings", return_value=3) as mock_purge_dep:
                purged = purge_stale_embeddings({"flask", "requests"})

        mock_purge_dep.assert_called_once_with("django", "")
        assert purged == {"django": 3}

    def test_keeps_all_when_no_orphans(self) -> None:
        mock_vectorstore = MagicMock()
        mock_vectorstore.get.return_value = {
            "ids": ["flask:1:0", "requests:1:0"],
            "metadatas": [
                {"dep_name": "flask"},
                {"dep_name": "requests"},
            ],
        }

        with patch("migratowl.core.rag.get_vectorstore", return_value=mock_vectorstore):
            from migratowl.core.rag import purge_stale_embeddings

            with patch("migratowl.core.rag.purge_dep_embeddings") as mock_purge_dep:
                purged = purge_stale_embeddings({"flask", "requests"})

        mock_purge_dep.assert_not_called()
        assert purged == {}

    def test_handles_empty_collection(self) -> None:
        mock_vectorstore = MagicMock()
        mock_vectorstore.get.return_value = {"ids": [], "metadatas": []}

        with patch("migratowl.core.rag.get_vectorstore", return_value=mock_vectorstore):
            from migratowl.core.rag import purge_stale_embeddings

            purged = purge_stale_embeddings({"flask"})

        assert purged == {}

    def test_returns_purged_counts(self) -> None:
        mock_vectorstore = MagicMock()
        mock_vectorstore.get.return_value = {
            "ids": ["flask:1:0", "django:1:0", "django:2:0", "django:3:0", "celery:1:0"],
            "metadatas": [
                {"dep_name": "flask"},
                {"dep_name": "django"},
                {"dep_name": "django"},
                {"dep_name": "django"},
                {"dep_name": "celery"},
            ],
        }

        def fake_purge(dep_name: str, project_path: str = "") -> int:
            counts = {"django": 3, "celery": 1}
            return counts.get(dep_name, 0)

        with patch("migratowl.core.rag.get_vectorstore", return_value=mock_vectorstore):
            from migratowl.core.rag import purge_stale_embeddings

            with patch("migratowl.core.rag.purge_dep_embeddings", side_effect=fake_purge):
                purged = purge_stale_embeddings({"flask"})

        assert purged == {"django": 3, "celery": 1}


class TestVerificationConfidenceAdjustment:
    def test_confidence_unchanged_all_verified(self) -> None:
        """confidence * (1 - 0.5 * (1 - 1.0)) = confidence"""
        ratio = 1.0
        confidence = 0.9
        adjusted = confidence * (1 - 0.5 * (1 - ratio))
        assert adjusted == pytest.approx(0.9)

    def test_confidence_halved_none_verified(self) -> None:
        """confidence * (1 - 0.5 * (1 - 0.0)) = confidence * 0.5"""
        ratio = 0.0
        confidence = 0.8
        adjusted = confidence * (1 - 0.5 * (1 - ratio))
        assert adjusted == pytest.approx(0.4)

    def test_confidence_partial(self) -> None:
        """confidence * (1 - 0.5 * (1 - 0.5)) = confidence * 0.75"""
        ratio = 0.5
        confidence = 0.8
        adjusted = confidence * (1 - 0.5 * (1 - ratio))
        assert adjusted == pytest.approx(0.6)


class TestQueryVerification:
    @pytest.mark.asyncio
    async def test_query_calls_verify_and_adjusts_confidence(self) -> None:
        """query() must call verify_breaking_changes with LLM results and source chunks."""
        mock_vectorstore = MagicMock()
        raw_docs = ["Breaking: removed old_func", "Changed API"]
        mock_vectorstore.asimilarity_search = AsyncMock(
            return_value=[
                Document(page_content=text, metadata={"dep_name": "requests", "version": ver})
                for text, ver in zip(raw_docs, ["2.0.0", "3.0.0"])
            ]
        )

        bc = BreakingChange(
            api_name="old_func", change_type=ChangeType.REMOVED, description="Removed", migration_hint="Use new"
        )
        mock_analysis = ChangelogAnalysis(breaking_changes=[bc], deprecations=[], new_features=[], confidence=0.9)
        mock_structured_llm = _mock_structured_llm(mock_analysis)

        verified_bc = bc.model_copy(update={"verified": True})

        with (
            patch("migratowl.core.rag.get_vectorstore", return_value=mock_vectorstore),
            patch("migratowl.core.rag.get_structured_llm", return_value=mock_structured_llm),
            patch("migratowl.core.rag.verify_breaking_changes", return_value=([verified_bc], 1.0)) as mock_verify,
        ):
            from migratowl.core.rag import query

            result = await query("breaking changes in requests", "requests")

        mock_verify.assert_called_once_with(mock_analysis.breaking_changes, raw_docs)
        assert result.confidence == pytest.approx(0.9)
        assert result.breaking_changes[0].verified is True

    @pytest.mark.asyncio
    async def test_query_verifies_against_raw_chunks_not_summary(self) -> None:
        """When summarization triggers, verification must use raw documents, not the summary."""
        from migratowl.config import settings

        large_doc = "x" * (settings.summarize_threshold + 1)
        raw_docs = [large_doc]
        mock_vectorstore = MagicMock()
        mock_vectorstore.asimilarity_search = AsyncMock(
            return_value=[
                Document(page_content=large_doc, metadata={"dep_name": "flask", "version": "3.0.0"}),
            ]
        )

        bc = BreakingChange(
            api_name="old_func", change_type=ChangeType.REMOVED, description="Removed", migration_hint="Use new"
        )
        mock_analysis = ChangelogAnalysis(breaking_changes=[bc], deprecations=[], new_features=[], confidence=0.8)
        mock_structured_llm = _mock_structured_llm(mock_analysis)

        verified_bc = bc.model_copy(update={"verified": False})

        with (
            patch("migratowl.core.rag.get_vectorstore", return_value=mock_vectorstore),
            patch("migratowl.core.rag.get_structured_llm", return_value=mock_structured_llm),
            patch("migratowl.core.rag._summarize_changelog", new_callable=AsyncMock, return_value="concise summary"),
            patch("migratowl.core.rag.verify_breaking_changes", return_value=([verified_bc], 0.0)) as mock_verify,
        ):
            from migratowl.core.rag import query

            result = await query("breaking changes", "flask", n_results=3)

        # Verification must receive the raw documents, NOT "concise summary"
        mock_verify.assert_called_once_with(mock_analysis.breaking_changes, raw_docs)
        assert result.confidence == pytest.approx(0.4)
