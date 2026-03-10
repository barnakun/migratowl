"""Tests for migratowl.core.llm — LangChain-based LLM factory."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestLLMCreation:
    def setup_method(self) -> None:
        import migratowl.core.llm as llm_module

        llm_module._llm_instance = None
        llm_module._embeddings_instance = None

    def teardown_method(self) -> None:
        import migratowl.core.llm as llm_module

        llm_module._llm_instance = None
        llm_module._embeddings_instance = None

    def test_get_llm_openai_via_init_chat_model(self) -> None:
        from langchain_core.language_models import BaseChatModel

        from migratowl.core.llm import get_llm

        with (
            patch("migratowl.core.llm.settings") as mock_settings,
            patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}),
        ):
            mock_settings.model = "openai:gpt-4o-mini"
            llm = get_llm()
            assert isinstance(llm, BaseChatModel)

    def test_missing_openai_api_key_raises_value_error(self) -> None:
        from migratowl.core.llm import get_llm

        with (
            patch("migratowl.core.llm.settings") as mock_settings,
            patch.dict("os.environ", {}, clear=True),
        ):
            mock_settings.model = "openai:gpt-4o-mini"
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                get_llm()


class TestCreateLLMProviderKwargs:
    def setup_method(self) -> None:
        import migratowl.core.llm as llm_module

        llm_module._llm_instance = None
        llm_module._embeddings_instance = None

    def teardown_method(self) -> None:
        import migratowl.core.llm as llm_module

        llm_module._llm_instance = None
        llm_module._embeddings_instance = None

    def test_passes_max_retries(self) -> None:
        with (
            patch("migratowl.core.llm.settings") as mock_settings,
            patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}),
        ):
            mock_settings.model = "openai:gpt-4o-mini"

            from migratowl.core.llm import _create_llm

            llm = _create_llm()
            assert llm.max_retries >= 5

    def test_init_chat_model_called_with_correct_args_for_anthropic(self) -> None:
        from migratowl.core.llm import _create_llm

        with (
            patch("migratowl.core.llm.settings") as mock_settings,
            patch("migratowl.core.llm.init_chat_model") as mock_init,
        ):
            mock_settings.model = "anthropic:claude-sonnet-4-5-20250929"
            mock_init.return_value = MagicMock()
            _create_llm()
            mock_init.assert_called_once_with(
                "claude-sonnet-4-5-20250929",
                model_provider="anthropic",
                max_retries=5,
            )


class TestStructuredLLM:
    def setup_method(self) -> None:
        import migratowl.core.llm as llm_module

        llm_module._llm_instance = None
        llm_module._embeddings_instance = None

    def teardown_method(self) -> None:
        import migratowl.core.llm as llm_module

        llm_module._llm_instance = None
        llm_module._embeddings_instance = None

    def test_get_structured_llm_returns_runnable(self) -> None:
        from langchain_core.runnables import Runnable
        from pydantic import BaseModel, Field

        from migratowl.core.llm import get_structured_llm

        class TestSchema(BaseModel):
            answer: str = Field(description="test")

        with (
            patch("migratowl.core.llm.settings") as mock_settings,
            patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}),
        ):
            mock_settings.model = "openai:gpt-4o-mini"
            structured = get_structured_llm(TestSchema)
            assert isinstance(structured, Runnable)

    def test_uses_function_calling_method(self) -> None:
        from pydantic import BaseModel, Field

        from migratowl.core.llm import get_structured_llm

        class TestSchema(BaseModel):
            answer: str = Field(description="test")

        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = MagicMock()

        import migratowl.core.llm as llm_module

        llm_module._llm_instance = mock_llm

        with patch("migratowl.core.llm.settings") as mock_settings:
            mock_settings.model = "openai:gpt-4o-mini"
            get_structured_llm(TestSchema)
            mock_llm.with_structured_output.assert_called_once_with(TestSchema, method="function_calling")



class TestGetEmbeddings:
    def setup_method(self) -> None:
        import migratowl.core.llm as llm_module

        llm_module._embeddings_instance = None

    def teardown_method(self) -> None:
        import migratowl.core.llm as llm_module

        llm_module._embeddings_instance = None

    def test_calls_init_embeddings_with_settings_model(self) -> None:
        """get_embeddings() must call init_embeddings with the configured embedding model."""
        from migratowl.core.llm import get_embeddings

        mock_embeddings = MagicMock()

        with (
            patch("migratowl.core.llm.settings") as mock_settings,
            patch("migratowl.core.llm.init_embeddings", return_value=mock_embeddings) as mock_init,
        ):
            mock_settings.embedding_model = "openai:text-embedding-3-small"
            result = get_embeddings()

        mock_init.assert_called_once_with("openai:text-embedding-3-small")
        assert result is mock_embeddings


class TestGetEmbedding:
    @pytest.mark.asyncio
    async def test_get_embedding_returns_list_of_floats(self) -> None:
        from migratowl.core.llm import get_embedding

        mock_embeddings = AsyncMock()
        mock_embeddings.aembed_query = AsyncMock(return_value=[0.1, 0.2, 0.3])

        with patch("migratowl.core.llm.get_embeddings", return_value=mock_embeddings):
            result = await get_embedding("test text")
            assert isinstance(result, list)
            assert all(isinstance(x, float) for x in result)
            assert result == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_get_embedding_uses_correct_model(self) -> None:
        from migratowl.core.llm import get_embedding

        mock_embeddings = AsyncMock()
        mock_embeddings.aembed_query = AsyncMock(return_value=[0.1])

        with (
            patch("migratowl.core.llm.get_embeddings", return_value=mock_embeddings),
            patch("migratowl.core.llm.settings") as mock_settings,
        ):
            mock_settings.model = "openai:gpt-4o-mini"
            mock_settings.max_concurrent_llm_calls = 5

            await get_embedding("test")
            mock_embeddings.aembed_query.assert_called_once_with("test")


class TestGetEmbeddingOpenAISemaphore:
    @pytest.mark.asyncio
    async def test_openai_embedding_caps_concurrent_calls(self) -> None:
        """Concurrent get_embedding calls must be gated by get_llm_semaphore."""
        import asyncio

        import migratowl.core.llm as llm_module
        from migratowl.core.llm import get_embedding

        max_concurrent = 3
        llm_module._llm_semaphore = None  # reset
        active = 0
        peak = 0

        async def slow_embed(text: str) -> list[float]:
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.02)
            active -= 1
            return [0.1]

        mock_embeddings = MagicMock()
        mock_embeddings.aembed_query = slow_embed

        with (
            patch("migratowl.core.llm.get_embeddings", return_value=mock_embeddings),
            patch("migratowl.core.llm.settings") as mock_settings,
        ):
            mock_settings.max_concurrent_llm_calls = max_concurrent

            await asyncio.gather(*[get_embedding(f"text{i}") for i in range(10)])

        assert peak <= max_concurrent, f"Expected peak <= {max_concurrent}, got {peak}"
        llm_module._llm_semaphore = None  # cleanup


class TestClientSingleton:
    def setup_method(self) -> None:
        import migratowl.core.llm as llm_module

        llm_module._llm_instance = None
        llm_module._embeddings_instance = None

    def teardown_method(self) -> None:
        import migratowl.core.llm as llm_module

        llm_module._llm_instance = None
        llm_module._embeddings_instance = None

    def test_get_llm_returns_same_instance(self) -> None:
        from migratowl.core.llm import get_llm

        with (
            patch("migratowl.core.llm.settings") as mock_settings,
            patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}),
        ):
            mock_settings.model = "openai:gpt-4o-mini"
            c1 = get_llm()
            c2 = get_llm()
            assert c1 is c2

    def test_reset_clients_clears_singletons(self) -> None:
        from migratowl.core.llm import get_llm, reset_clients

        with (
            patch("migratowl.core.llm.settings") as mock_settings,
            patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}),
        ):
            mock_settings.model = "openai:gpt-4o-mini"
            c1 = get_llm()
            reset_clients()
            c2 = get_llm()
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
