"""Tests for migratowl.core.llm — Instructor-wrapped LLM client."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestClientCreation:
    def setup_method(self) -> None:
        import migratowl.core.llm as llm_module

        llm_module._client = None
        llm_module._raw_client = None

    def teardown_method(self) -> None:
        import migratowl.core.llm as llm_module

        llm_module._client = None
        llm_module._raw_client = None

    def test_get_client_returns_instructor_client(self) -> None:
        from migratowl.core.llm import get_client

        with patch("migratowl.core.llm.settings") as mock_settings:
            mock_settings.openai_api_key = "sk-test"
            mock_settings.use_local_llm = False
            client = get_client()
            assert hasattr(client.chat.completions, "create")

    def test_get_client_ollama_mode(self) -> None:
        from migratowl.core.llm import get_client

        with patch("migratowl.core.llm.settings") as mock_settings:
            mock_settings.use_local_llm = True
            mock_settings.ollama_base_url = "http://localhost:11434/v1"
            mock_settings.openai_api_key = "ollama"
            client = get_client()
            assert hasattr(client.chat.completions, "create")

    def test_missing_api_key_raises_value_error(self) -> None:
        from migratowl.core.llm import get_client

        with patch("migratowl.core.llm.settings") as mock_settings:
            mock_settings.use_local_llm = False
            mock_settings.openai_api_key = ""
            with pytest.raises(ValueError, match="MIGRATOWL_OPENAI_API_KEY"):
                get_client()


class TestOpenAIClientRetries:
    def setup_method(self) -> None:
        import migratowl.core.llm as llm_module

        llm_module._client = None
        llm_module._raw_client = None

    def teardown_method(self) -> None:
        import migratowl.core.llm as llm_module

        llm_module._client = None
        llm_module._raw_client = None

    def test_openai_client_has_retry_budget_for_rate_limits(self) -> None:
        """AsyncOpenAI clients must be created with max_retries > 2 to handle 429 TPM errors."""
        with patch("migratowl.core.llm.settings") as mock_settings:
            mock_settings.use_local_llm = False
            mock_settings.openai_api_key = "sk-test"

            from migratowl.core.llm import _create_client

            instructor_client = _create_client()
            # Access the wrapped OpenAI client
            raw_client = instructor_client.client
            assert raw_client.max_retries >= 5, (
                f"Expected max_retries >= 5 for 429 backoff, got {raw_client.max_retries}"
            )

    def test_raw_openai_client_has_retry_budget(self) -> None:
        """Raw embedding client must also have elevated max_retries."""
        with patch("migratowl.core.llm.settings") as mock_settings:
            mock_settings.use_local_llm = False
            mock_settings.openai_api_key = "sk-test"

            from migratowl.core.llm import _get_raw_openai_client

            raw_client = _get_raw_openai_client()
            assert raw_client.max_retries >= 5


class TestGetEmbedding:
    @pytest.mark.asyncio
    async def test_get_embedding_returns_list_of_floats(self) -> None:
        from migratowl.core.llm import get_embedding

        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]

        with patch("migratowl.core.llm._get_raw_openai_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.embeddings.create.return_value = mock_response
            mock_get_client.return_value = mock_client

            result = await get_embedding("test text")
            assert isinstance(result, list)
            assert all(isinstance(x, float) for x in result)
            assert result == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_get_embedding_uses_correct_model(self) -> None:
        from migratowl.core.llm import get_embedding

        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1])]

        with (
            patch("migratowl.core.llm._get_raw_openai_client") as mock_get_client,
            patch("migratowl.core.llm.active_embedding_model", return_value="text-embedding-3-small"),
            patch("migratowl.core.llm.settings") as mock_settings,
        ):
            mock_settings.use_local_llm = False
            mock_client = AsyncMock()
            mock_client.embeddings.create.return_value = mock_response
            mock_get_client.return_value = mock_client

            await get_embedding("test")
            mock_client.embeddings.create.assert_called_once_with(
                model="text-embedding-3-small",
                input="test",
            )

    @pytest.mark.asyncio
    async def test_local_llm_serializes_concurrent_embedding_calls(self) -> None:
        """Concurrent get_embedding calls with local LLM must not exceed 1 at a time."""
        import asyncio

        from migratowl.core.llm import get_embedding

        active = 0
        concurrency_log: list[int] = []

        async def slow_create(model: str, input: str) -> MagicMock:  # noqa: A002
            nonlocal active
            active += 1
            concurrency_log.append(active)
            await asyncio.sleep(0.02)
            active -= 1
            response = MagicMock()
            response.data = [MagicMock(embedding=[0.1])]
            return response

        with (
            patch("migratowl.core.llm._get_raw_openai_client") as mock_get_client,
            patch("migratowl.core.llm.active_embedding_model", return_value="nomic-embed-text"),
            patch("migratowl.core.llm.settings") as mock_settings,
        ):
            mock_settings.use_local_llm = True
            mock_client = MagicMock()
            mock_client.embeddings.create = slow_create
            mock_get_client.return_value = mock_client

            await asyncio.gather(*[get_embedding(f"text{i}") for i in range(5)])

        assert max(concurrency_log) == 1, f"Expected max 1 concurrent call, got {max(concurrency_log)}"

    @pytest.mark.asyncio
    async def test_get_embedding_ollama_uses_local_model(self) -> None:
        from migratowl.core.llm import get_embedding

        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.5])]

        with (
            patch("migratowl.core.llm._get_raw_openai_client") as mock_get_client,
            patch("migratowl.core.llm.active_embedding_model", return_value="nomic-embed-text"),
            patch("migratowl.core.llm.settings") as mock_settings,
        ):
            mock_settings.use_local_llm = True
            mock_client = AsyncMock()
            mock_client.embeddings.create.return_value = mock_response
            mock_get_client.return_value = mock_client

            await get_embedding("test")
            mock_client.embeddings.create.assert_called_once_with(
                model="nomic-embed-text",
                input="test",
            )


class TestGetEmbeddingOpenAISemaphore:
    @pytest.mark.asyncio
    async def test_openai_embedding_caps_concurrent_calls(self) -> None:
        """Concurrent get_embedding calls with OpenAI must be gated by get_llm_semaphore."""
        import asyncio

        import migratowl.core.llm as llm_module
        from migratowl.core.llm import get_embedding

        max_concurrent = 3
        llm_module._llm_semaphore = None  # reset
        active = 0
        peak = 0

        async def slow_create(model: str, input: str) -> MagicMock:  # noqa: A002
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.02)
            active -= 1
            response = MagicMock()
            response.data = [MagicMock(embedding=[0.1])]
            return response

        with (
            patch("migratowl.core.llm._get_raw_openai_client") as mock_get_client,
            patch("migratowl.core.llm.active_embedding_model", return_value="text-embedding-3-small"),
            patch("migratowl.core.llm.settings") as mock_settings,
        ):
            mock_settings.use_local_llm = False
            mock_settings.max_concurrent_llm_calls = max_concurrent
            mock_client = MagicMock()
            mock_client.embeddings.create = slow_create
            mock_get_client.return_value = mock_client

            await asyncio.gather(*[get_embedding(f"text{i}") for i in range(10)])

        assert peak <= max_concurrent, f"Expected peak ≤ {max_concurrent}, got {peak}"
        llm_module._llm_semaphore = None  # cleanup


class TestClientSingleton:
    def setup_method(self) -> None:
        import migratowl.core.llm as llm_module

        llm_module._client = None
        llm_module._raw_client = None

    def teardown_method(self) -> None:
        import migratowl.core.llm as llm_module

        llm_module._client = None
        llm_module._raw_client = None

    def test_get_client_returns_same_instance(self) -> None:
        from migratowl.core.llm import get_client

        with patch("migratowl.core.llm.settings") as mock_settings:
            mock_settings.use_local_llm = False
            mock_settings.openai_api_key = "sk-test"
            c1 = get_client()
            c2 = get_client()
            assert c1 is c2

    def test_get_raw_client_returns_same_instance(self) -> None:
        from migratowl.core.llm import _get_raw_openai_client

        with patch("migratowl.core.llm.settings") as mock_settings:
            mock_settings.use_local_llm = False
            mock_settings.openai_api_key = "sk-test"
            c1 = _get_raw_openai_client()
            c2 = _get_raw_openai_client()
            assert c1 is c2

    def test_reset_clients_clears_singletons(self) -> None:
        from migratowl.core.llm import get_client, reset_clients

        with patch("migratowl.core.llm.settings") as mock_settings:
            mock_settings.use_local_llm = False
            mock_settings.openai_api_key = "sk-test"
            c1 = get_client()
            reset_clients()
            c2 = get_client()
            assert c1 is not c2


class TestLLMSemaphore:
    def test_get_llm_semaphore_returns_asyncio_semaphore(self) -> None:
        """get_llm_semaphore must return an asyncio.Semaphore."""
        import asyncio

        import migratowl.core.llm as llm_module
        from migratowl.core.llm import get_llm_semaphore

        llm_module._llm_semaphore = None  # reset for clean test
        with patch("migratowl.core.llm.settings") as mock_settings:
            mock_settings.max_concurrent_llm_calls = 3
            sem = get_llm_semaphore()
            assert isinstance(sem, asyncio.Semaphore)
        llm_module._llm_semaphore = None  # cleanup

    @pytest.mark.asyncio
    async def test_llm_semaphore_caps_concurrent_calls(self) -> None:
        """At most max_concurrent_llm_calls coroutines hold the semaphore simultaneously."""
        import asyncio

        import migratowl.core.llm as llm_module
        from migratowl.core.llm import get_llm_semaphore

        max_concurrent = 3
        llm_module._llm_semaphore = None  # reset
        concurrent_count = 0
        peak_concurrent = 0

        async def slow_task() -> None:
            nonlocal concurrent_count, peak_concurrent
            async with get_llm_semaphore():
                concurrent_count += 1
                peak_concurrent = max(peak_concurrent, concurrent_count)
                await asyncio.sleep(0.01)
                concurrent_count -= 1

        with patch("migratowl.core.llm.settings") as mock_settings:
            mock_settings.max_concurrent_llm_calls = max_concurrent
            await asyncio.gather(*[slow_task() for _ in range(10)])

        assert peak_concurrent <= max_concurrent
        llm_module._llm_semaphore = None  # cleanup

    def test_get_llm_semaphore_returns_same_instance(self) -> None:
        """Successive calls must return the same semaphore (singleton)."""
        import migratowl.core.llm as llm_module
        from migratowl.core.llm import get_llm_semaphore

        llm_module._llm_semaphore = None
        with patch("migratowl.core.llm.settings") as mock_settings:
            mock_settings.max_concurrent_llm_calls = 5
            s1 = get_llm_semaphore()
            s2 = get_llm_semaphore()
            assert s1 is s2
        llm_module._llm_semaphore = None
