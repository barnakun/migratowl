"""Instructor-wrapped OpenAI client — ALL LLM calls go through here."""

import asyncio

import instructor
from openai import AsyncOpenAI

from migratowl.config import active_embedding_model, settings

# Ollama is single-threaded; serialise all embedding requests to avoid
# "connection refused" / overload errors when multiple deps are analysed in parallel.
_ollama_semaphore = asyncio.Semaphore(1)

# Gate all LLM chat completion calls to prevent 429 TPM errors when many deps
# are analysed concurrently. Configurable via MIGRATOWL_MAX_CONCURRENT_LLM_CALLS.
# Lazily initialised so it reads settings at first use, not at import time.
_llm_semaphore: asyncio.Semaphore | None = None
_client: instructor.AsyncInstructor | None = None
_raw_client: AsyncOpenAI | None = None


def get_llm_semaphore() -> asyncio.Semaphore:
    """Return the module-level semaphore that gates concurrent LLM calls."""
    global _llm_semaphore
    if _llm_semaphore is None:
        _llm_semaphore = asyncio.Semaphore(settings.max_concurrent_llm_calls)
    return _llm_semaphore


def _create_client() -> instructor.AsyncInstructor:
    """Create an Instructor-wrapped async OpenAI client."""
    if settings.use_local_llm:
        return instructor.from_openai(
            AsyncOpenAI(
                base_url=settings.ollama_base_url,
                api_key="ollama",
            ),
            mode=instructor.Mode.JSON,
        )
    return instructor.from_openai(
        AsyncOpenAI(api_key=settings.openai_api_key, max_retries=5),
    )


def get_client() -> instructor.AsyncInstructor:
    """Return a cached Instructor client, creating it on first call."""
    global _client
    if not settings.use_local_llm and not settings.openai_api_key:
        msg = (
            "MIGRATOWL_OPENAI_API_KEY is required when not using local LLM. "
            "Set the env var or use MIGRATOWL_USE_LOCAL_LLM=true for Ollama."
        )
        raise ValueError(msg)
    if _client is None:
        _client = _create_client()
    return _client


def _get_raw_openai_client() -> AsyncOpenAI:
    """Return a cached raw AsyncOpenAI client for embeddings."""
    global _raw_client
    if _raw_client is None:
        if settings.use_local_llm:
            _raw_client = AsyncOpenAI(
                base_url=settings.ollama_base_url,
                api_key="ollama",
            )
        else:
            _raw_client = AsyncOpenAI(api_key=settings.openai_api_key, max_retries=5)
    return _raw_client


def reset_clients() -> None:
    """Clear cached clients so next call creates fresh instances."""
    global _client, _raw_client
    _client = None
    _raw_client = None


async def get_embedding(text: str) -> list[float]:
    """Get embedding vector for text using OpenAI or Ollama."""
    raw_client = _get_raw_openai_client()
    model = active_embedding_model()
    if settings.use_local_llm:
        async with _ollama_semaphore:
            response = await raw_client.embeddings.create(model=model, input=text)
    else:
        async with get_llm_semaphore():
            response = await raw_client.embeddings.create(model=model, input=text)
    return response.data[0].embedding
