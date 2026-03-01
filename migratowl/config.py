"""Application configuration via pydantic-settings with MIGRATOWL_ env prefix."""

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Load .env into os.environ so third-party SDKs (LangSmith, OpenAI) can read it.
load_dotenv(override=False)


class Settings(BaseSettings):
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    github_token: str = ""
    use_local_llm: bool = False
    ollama_base_url: str = "http://localhost:11434/v1"
    local_llm_model: str = "llama3.2"
    vectorstore_path: str = ".migratowl/vectorstore"
    max_retries: int = 2
    confidence_threshold: float = 0.6
    embedding_model: str = "text-embedding-3-small"
    local_embedding_model: str = "nomic-embed-text"
    max_concurrent_deps: int = 20
    max_concurrent_registry_queries: int = 20
    max_rag_results: int = 20
    max_concurrent_llm_calls: int = 5

    model_config = {"env_prefix": "MIGRATOWL_", "env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()


def active_model() -> str:
    """Return the model name to use for completions based on current config."""
    return settings.local_llm_model if settings.use_local_llm else settings.openai_model


def active_embedding_model() -> str:
    """Return the embedding model name based on current config."""
    return settings.local_embedding_model if settings.use_local_llm else settings.embedding_model
