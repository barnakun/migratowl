"""Instructor-wrapped OpenAI client — ALL LLM calls go through here."""

import asyncio

import instructor
from openai import AsyncOpenAI

from migratowl.config import settings

# Ollama is single-threaded; serialise all embedding requests to avoid
# "connection refused" / overload errors when multiple deps are analysed in parallel.
_ollama_semaphore = asyncio.Semaphore(1)


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
        AsyncOpenAI(api_key=settings.openai_api_key),
    )


def get_client() -> instructor.AsyncInstructor:
    """Create a fresh client with API key validation."""
    if not settings.use_local_llm and not settings.openai_api_key:
        msg = (
            "MIGRATOWL_OPENAI_API_KEY is required when not using local LLM. "
            "Set the env var or use MIGRATOWL_USE_LOCAL_LLM=true for Ollama."
        )
        raise ValueError(msg)
    return _create_client()


def _get_raw_openai_client() -> AsyncOpenAI:
    """Get a raw AsyncOpenAI client for embeddings."""
    if settings.use_local_llm:
        return AsyncOpenAI(
            base_url=settings.ollama_base_url,
            api_key="ollama",
        )
    return AsyncOpenAI(api_key=settings.openai_api_key)


async def get_embedding(text: str) -> list[float]:
    """Get embedding vector for text using OpenAI or Ollama."""
    raw_client = _get_raw_openai_client()
    model = settings.local_embedding_model if settings.use_local_llm else settings.embedding_model
    if settings.use_local_llm:
        async with _ollama_semaphore:
            response = await raw_client.embeddings.create(model=model, input=text)
    else:
        response = await raw_client.embeddings.create(model=model, input=text)
    return response.data[0].embedding


client = _create_client()
