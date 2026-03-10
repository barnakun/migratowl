"""Application configuration via pydantic-settings with MIGRATOWL_ env prefix."""

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Load .env into os.environ so third-party SDKs (LangSmith, OpenAI) can read it.
load_dotenv(override=False)

_MODEL_DEFAULT = "openai:gpt-4o-mini"


class Settings(BaseSettings):
    model: str = _MODEL_DEFAULT
    github_token: str = ""
    vectorstore_path: str = ".migratowl/vectorstore"
    confidence_threshold: float = 0.6
    embedding_model: str = "openai:text-embedding-3-small"
    max_concurrent_deps: int = 20
    max_concurrent_registry_queries: int = 20
    max_rag_results: int = 20
    max_concurrent_llm_calls: int = 5
    summarize_threshold: int = 32_000
    cache_path: str = ".migratowl/cache"
    changelog_cache_path: str = ".migratowl/changelog-cache"
    changelog_cache_ttl_minutes: int = 1440
    http_timeout: float = 30.0
    http_retry_count: int = 3
    http_retry_backoff_base: float = 0.5
    ignored_dependencies: str = ""
    log_level: str = "WARNING"

    @property
    def parsed_ignored_dependencies(self) -> list[str]:
        """Parse comma-separated ignored_dependencies string into a list."""
        if not self.ignored_dependencies:
            return []
        return [d.strip() for d in self.ignored_dependencies.split(",") if d.strip()]

    model_config = {"env_prefix": "MIGRATOWL_", "env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
