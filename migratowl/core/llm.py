"""LangChain-based LLM factory — ALL LLM calls go through here."""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING

from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable

from migratowl.config import settings

if TYPE_CHECKING:
    from pydantic import BaseModel

# Gate all LLM chat completion calls to prevent 429 TPM errors when many deps
# are analysed concurrently. Configurable via MIGRATOWL_MAX_CONCURRENT_LLM_CALLS.
# Lazily initialised so it reads settings at first use, not at import time.
_llm_semaphore: asyncio.Semaphore | None = None
_llm_instance: BaseChatModel | None = None
_embeddings_instance: Embeddings | None = None

# Providers that need an API key, with env var names to check.
_PROVIDERS_NEEDING_KEY: dict[str, tuple[str, ...]] = {
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "google_genai": ("GOOGLE_API_KEY",),
    "mistral": ("MISTRAL_API_KEY",),
    "groq": ("GROQ_API_KEY",),
}


def _get_provider() -> str:
    """Extract provider from settings.model string."""
    model = settings.model
    if ":" in model:
        return model.split(":", 1)[0]
    return "openai"


def _get_model_name() -> str:
    """Extract model name from settings.model string."""
    model = settings.model
    if ":" in model:
        return model.split(":", 1)[1]
    return model


def get_llm_semaphore() -> asyncio.Semaphore:
    """Return the module-level semaphore that gates concurrent LLM calls."""
    global _llm_semaphore
    if _llm_semaphore is None:
        _llm_semaphore = asyncio.Semaphore(settings.max_concurrent_llm_calls)
    return _llm_semaphore


def _create_llm() -> BaseChatModel:
    """Create a BaseChatModel instance via init_chat_model()."""
    provider = _get_provider()
    model_name = _get_model_name()

    result: BaseChatModel = init_chat_model(
        model_name, model_provider=provider, max_retries=5
    )
    return result


def _validate_api_key() -> None:
    """Validate that the required API key is available for the current provider."""
    provider = _get_provider()

    env_vars = _PROVIDERS_NEEDING_KEY.get(provider)
    if env_vars is None:
        return

    for var in env_vars:
        if os.environ.get(var):
            return

    var_names = " or ".join(env_vars)
    msg = f"{var_names} is required for provider '{provider}'. Set the env var."
    raise ValueError(msg)


def get_llm() -> BaseChatModel:
    """Return a cached BaseChatModel instance, creating it on first call."""
    global _llm_instance
    if _llm_instance is None:
        _validate_api_key()
        _llm_instance = _create_llm()
    return _llm_instance


def get_structured_llm(schema: type[BaseModel]) -> Runnable:
    """Return an LLM bound to a Pydantic schema via with_structured_output()."""
    llm = get_llm()
    return llm.with_structured_output(schema, method="function_calling")


def get_embeddings() -> Embeddings:
    """Return a cached Embeddings instance via init_embeddings().

    Uses the same provider:model pattern as init_chat_model().
    Configured via MIGRATOWL_EMBEDDING_MODEL (default: openai:text-embedding-3-small).
    """
    global _embeddings_instance
    if _embeddings_instance is None:
        _embeddings_instance = init_embeddings(settings.embedding_model)
    return _embeddings_instance


def reset_clients() -> None:
    """Clear cached clients so next call creates fresh instances."""
    global _llm_instance, _embeddings_instance
    _llm_instance = None
    _embeddings_instance = None


async def get_embedding(text: str) -> list[float]:
    """Get embedding vector for a single text string.

    Used for preflight checks and direct embedding needs.
    For vector store operations, use get_embeddings() with langchain-chroma instead.
    """
    embeddings_model = get_embeddings()
    async with get_llm_semaphore():
        result: list[float] = await embeddings_model.aembed_query(text)
    return result
