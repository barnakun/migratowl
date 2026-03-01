"""Tests for RAG module with mocked ChromaDB and LLM."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from migratowl.models.schemas import (
    BreakingChange,
    ChangelogAnalysis,
    ChangeType,
    RAGQueryResult,
)


class TestGetCollection:
    def test_creates_collection_with_cosine_similarity(self) -> None:
        mock_chromadb = MagicMock()
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_chromadb.PersistentClient.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_collection

        with (
            patch("migratowl.core.rag._import_chromadb", return_value=mock_chromadb),
            patch("migratowl.core.rag.settings") as mock_settings,
        ):
            mock_settings.use_local_llm = False
            mock_settings.embedding_model = "text-embedding-3-small"
            mock_settings.local_embedding_model = "nomic-embed-text"
            mock_settings.vectorstore_path = "/tmp/vs"

            from migratowl.core.rag import get_collection

            collection = get_collection()

            mock_chromadb.PersistentClient.assert_called_once()
            call_args = mock_client.get_or_create_collection.call_args
            collection_name = call_args.args[0] if call_args.args else call_args.kwargs["name"]
            assert collection_name.startswith("changelogs_")
            assert call_args.kwargs["metadata"] == {"hnsw:space": "cosine"}
            assert collection is mock_collection

    def test_openai_and_ollama_use_different_collection_names(self) -> None:
        """Different embedding backends must use separate collections to avoid dimension conflicts."""
        mock_chromadb = MagicMock()
        mock_client = MagicMock()
        mock_chromadb.PersistentClient.return_value = mock_client
        mock_client.get_or_create_collection.return_value = MagicMock()

        from migratowl.core.rag import get_collection

        with (
            patch("migratowl.core.rag._import_chromadb", return_value=mock_chromadb),
            patch("migratowl.core.rag.settings") as mock_settings,
        ):
            mock_settings.vectorstore_path = "/tmp/vs"
            mock_settings.use_local_llm = False
            mock_settings.embedding_model = "text-embedding-3-small"
            mock_settings.local_embedding_model = "nomic-embed-text"
            get_collection()
            openai_name = mock_client.get_or_create_collection.call_args.args[0]

        mock_client.reset_mock()

        with (
            patch("migratowl.core.rag._import_chromadb", return_value=mock_chromadb),
            patch("migratowl.core.rag.settings") as mock_settings,
        ):
            mock_settings.vectorstore_path = "/tmp/vs"
            mock_settings.use_local_llm = True
            mock_settings.embedding_model = "text-embedding-3-small"
            mock_settings.local_embedding_model = "nomic-embed-text"
            get_collection()
            ollama_name = mock_client.get_or_create_collection.call_args.args[0]

        assert openai_name != ollama_name

    def test_uses_persistent_client_with_settings_path(self) -> None:
        mock_chromadb = MagicMock()
        mock_client = MagicMock()
        mock_chromadb.PersistentClient.return_value = mock_client
        mock_client.get_or_create_collection.return_value = MagicMock()

        with (
            patch("migratowl.core.rag._import_chromadb", return_value=mock_chromadb),
            patch("migratowl.core.rag.settings") as mock_settings,
        ):
            mock_settings.vectorstore_path = "/tmp/test_vectorstore"

            from migratowl.core.rag import get_collection

            get_collection()

            mock_chromadb.PersistentClient.assert_called_once_with(path="/tmp/test_vectorstore")


class TestEmbedChangelog:
    @pytest.mark.asyncio
    async def test_upserts_chunks_with_metadata(self) -> None:
        mock_collection = MagicMock()
        mock_embedding = [0.1] * 128

        chunks = [
            {"version": "2.0.0", "content": "Breaking change"},
            {"version": "1.0.0", "content": "Initial release"},
        ]

        with (
            patch(
                "migratowl.core.rag.get_collection",
                return_value=mock_collection,
            ),
            patch(
                "migratowl.core.rag.get_embedding",
                new_callable=AsyncMock,
                return_value=mock_embedding,
            ),
        ):
            from migratowl.core.rag import embed_changelog

            await embed_changelog("requests", chunks)

            assert mock_collection.upsert.call_count == 2

            first_call = mock_collection.upsert.call_args_list[0]
            assert first_call.kwargs["metadatas"] == [{"dep_name": "requests", "version": "2.0.0"}]
            assert first_call.kwargs["embeddings"] == [mock_embedding]
            assert first_call.kwargs["documents"] == ["Breaking change"]

    @pytest.mark.asyncio
    async def test_sub_chunks_oversized_version_sections(self) -> None:
        """Version sections larger than EMBED_CHUNK_CHARS must be split into multiple
        sub-chunks, each embedded separately — no information is lost and each sub-chunk
        stays within the embedding model's context window."""
        from migratowl.core.rag import EMBED_CHUNK_CHARS, embed_changelog

        mock_collection = MagicMock()
        mock_embedding = [0.1] * 128
        # Content that is exactly 2.5× the chunk size → should produce 3 sub-chunks
        oversized_content = "x" * (EMBED_CHUNK_CHARS * 2 + EMBED_CHUNK_CHARS // 2)
        chunks = [{"version": "1.0.0", "content": oversized_content}]

        captured: list[str] = []

        async def capture_embedding(text: str) -> list[float]:
            captured.append(text)
            return mock_embedding

        with (
            patch("migratowl.core.rag.get_collection", return_value=mock_collection),
            patch("migratowl.core.rag.get_embedding", side_effect=capture_embedding),
        ):
            await embed_changelog("flask", chunks)

        assert len(captured) == 3
        assert all(len(t) <= EMBED_CHUNK_CHARS for t in captured)
        # Full content is preserved across sub-chunks
        assert "".join(captured) == oversized_content

    @pytest.mark.asyncio
    async def test_small_chunk_not_split(self) -> None:
        """Chunks that fit within EMBED_CHUNK_CHARS must not be split."""
        from migratowl.core.rag import EMBED_CHUNK_CHARS, embed_changelog

        mock_collection = MagicMock()
        mock_embedding = [0.1] * 128
        small_content = "x" * (EMBED_CHUNK_CHARS - 1)
        chunks = [{"version": "1.0.0", "content": small_content}]

        captured: list[str] = []

        async def capture_embedding(text: str) -> list[float]:
            captured.append(text)
            return mock_embedding

        with (
            patch("migratowl.core.rag.get_collection", return_value=mock_collection),
            patch("migratowl.core.rag.get_embedding", side_effect=capture_embedding),
        ):
            await embed_changelog("flask", chunks)

        assert len(captured) == 1


class TestQueryNResults:
    @pytest.mark.asyncio
    async def test_none_n_results_queries_all_dep_chunks(self) -> None:
        """n_results=None must retrieve all chunks stored for that dep."""
        mock_collection = MagicMock()
        # Simulate 12 chunks stored for flask
        mock_collection.get.return_value = {"ids": [f"flask:1.0:{i}" for i in range(12)]}
        mock_collection.query.return_value = {
            "documents": [["some text"]],
            "distances": [[0.1]],
            "metadatas": [[{"dep_name": "flask", "version": "2.0.0"}]],
        }

        mock_analysis = ChangelogAnalysis(breaking_changes=[], deprecations=[], new_features=[], confidence=0.5)
        mock_instructor_client = MagicMock()
        mock_instructor_client.chat.completions.create = AsyncMock(return_value=mock_analysis)

        with (
            patch("migratowl.core.rag.get_collection", return_value=mock_collection),
            patch("migratowl.core.rag.get_embedding", new_callable=AsyncMock, return_value=[0.1] * 128),
            patch("migratowl.core.rag.get_client", return_value=mock_instructor_client),
        ):
            from migratowl.core.rag import query

            await query("breaking changes", "flask", n_results=None)

        query_kwargs = mock_collection.query.call_args.kwargs
        assert query_kwargs["n_results"] == 12

    @pytest.mark.asyncio
    async def test_explicit_n_results_passed_directly(self) -> None:
        """An explicit int n_results must be passed to ChromaDB unchanged."""
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["text"]],
            "distances": [[0.1]],
            "metadatas": [[{"dep_name": "flask", "version": "2.0.0"}]],
        }

        mock_analysis = ChangelogAnalysis(breaking_changes=[], deprecations=[], new_features=[], confidence=0.5)
        mock_instructor_client = MagicMock()
        mock_instructor_client.chat.completions.create = AsyncMock(return_value=mock_analysis)

        with (
            patch("migratowl.core.rag.get_collection", return_value=mock_collection),
            patch("migratowl.core.rag.get_embedding", new_callable=AsyncMock, return_value=[0.1] * 128),
            patch("migratowl.core.rag.get_client", return_value=mock_instructor_client),
        ):
            from migratowl.core.rag import query

            await query("breaking changes", "flask", n_results=7)

        query_kwargs = mock_collection.query.call_args.kwargs
        assert query_kwargs["n_results"] == 7

    def test_default_n_results_uses_settings(self) -> None:
        """RAG_N_RESULTS constant must use settings.max_rag_results (default 20)."""
        from migratowl.core.rag import RAG_N_RESULTS

        assert RAG_N_RESULTS == 20


class TestQuery:
    @pytest.mark.asyncio
    async def test_returns_rag_query_result(self) -> None:
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["Breaking: removed old_func", "Changed API"]],
            "distances": [[0.1, 0.2]],
            "metadatas": [
                [
                    {"dep_name": "requests", "version": "2.0.0"},
                    {"dep_name": "requests", "version": "3.0.0"},
                ]
            ],
        }

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

        mock_embedding = [0.1] * 128

        mock_instructor_client = MagicMock()
        mock_create = AsyncMock(return_value=mock_analysis)
        mock_instructor_client.chat.completions.create = mock_create

        with (
            patch(
                "migratowl.core.rag.get_collection",
                return_value=mock_collection,
            ),
            patch(
                "migratowl.core.rag.get_embedding",
                new_callable=AsyncMock,
                return_value=mock_embedding,
            ),
            patch(
                "migratowl.core.rag.get_client",
                return_value=mock_instructor_client,
            ),
        ):
            from migratowl.core.rag import query

            result = await query("breaking changes in requests", "requests")

            assert isinstance(result, RAGQueryResult)
            assert len(result.breaking_changes) == 1
            assert result.breaking_changes[0].api_name == "old_func"
            assert result.confidence == 0.9
            assert len(result.source_chunks) == 2

    @pytest.mark.asyncio
    async def test_llm_called_with_response_model(self) -> None:
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["some changelog text"]],
            "distances": [[0.1]],
            "metadatas": [[{"dep_name": "flask", "version": "2.0.0"}]],
        }

        mock_analysis = ChangelogAnalysis(
            breaking_changes=[],
            deprecations=[],
            new_features=[],
            confidence=0.5,
        )

        mock_instructor_client = MagicMock()
        mock_create = AsyncMock(return_value=mock_analysis)
        mock_instructor_client.chat.completions.create = mock_create

        mock_embedding = [0.1] * 128

        with (
            patch(
                "migratowl.core.rag.get_collection",
                return_value=mock_collection,
            ),
            patch(
                "migratowl.core.rag.get_embedding",
                new_callable=AsyncMock,
                return_value=mock_embedding,
            ),
            patch(
                "migratowl.core.rag.get_client",
                return_value=mock_instructor_client,
            ),
        ):
            from migratowl.core.rag import query

            await query("changes in flask", "flask")

            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["response_model"] is ChangelogAnalysis
            assert call_kwargs["max_retries"] == 2

    @pytest.mark.asyncio
    async def test_query_filters_by_dep_name(self) -> None:
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["text"]],
            "distances": [[0.1]],
            "metadatas": [[{"dep_name": "flask", "version": "2.0.0"}]],
        }

        mock_analysis = ChangelogAnalysis(breaking_changes=[], deprecations=[], new_features=[], confidence=0.5)

        mock_instructor_client = MagicMock()
        mock_instructor_client.chat.completions.create = AsyncMock(return_value=mock_analysis)

        mock_embedding = [0.1] * 128

        with (
            patch(
                "migratowl.core.rag.get_collection",
                return_value=mock_collection,
            ),
            patch(
                "migratowl.core.rag.get_embedding",
                new_callable=AsyncMock,
                return_value=mock_embedding,
            ),
            patch(
                "migratowl.core.rag.get_client",
                return_value=mock_instructor_client,
            ),
        ):
            from migratowl.core.rag import query

            await query("changes", "flask", n_results=3)

            query_kwargs = mock_collection.query.call_args.kwargs
            assert query_kwargs["where"] == {"dep_name": "flask"}
            assert query_kwargs["n_results"] == 3


class TestQueryLLMSemaphore:
    @pytest.mark.asyncio
    async def test_query_acquires_llm_semaphore(self) -> None:
        """RAG query must hold the LLM semaphore while making the LLM call."""
        from unittest.mock import AsyncMock, MagicMock, patch

        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["some text"]],
            "distances": [[0.1]],
            "metadatas": [[{"dep_name": "flask", "version": "2.0.0"}]],
        }

        mock_analysis = ChangelogAnalysis(
            breaking_changes=[], deprecations=[], new_features=[], confidence=0.5
        )
        mock_instructor_client = MagicMock()
        mock_instructor_client.chat.completions.create = AsyncMock(return_value=mock_analysis)

        mock_sem = MagicMock()
        mock_sem.__aenter__ = AsyncMock(return_value=None)
        mock_sem.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("migratowl.core.rag.get_collection", return_value=mock_collection),
            patch("migratowl.core.rag.get_embedding", new_callable=AsyncMock, return_value=[0.1] * 128),
            patch("migratowl.core.rag.get_client", return_value=mock_instructor_client),
            patch("migratowl.core.rag.get_llm_semaphore", return_value=mock_sem),
        ):
            from migratowl.core.rag import query

            await query("breaking changes", "flask", n_results=3)

        mock_sem.__aenter__.assert_called_once()
        mock_sem.__aexit__.assert_called_once()
