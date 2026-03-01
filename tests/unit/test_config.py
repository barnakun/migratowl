"""Tests for migratowl.config — Settings with MIGRATOWL_ env prefix."""

import pytest

from migratowl.config import Settings


class TestSettingsDefaults:
    def test_default_openai_model(self) -> None:
        s = Settings(openai_api_key="test-key")
        assert s.openai_model == "gpt-4o-mini"

    def test_default_use_local_llm(self) -> None:
        s = Settings(openai_api_key="test-key")
        assert s.use_local_llm is False

    def test_default_ollama_base_url(self) -> None:
        s = Settings(openai_api_key="test-key")
        assert s.ollama_base_url == "http://localhost:11434/v1"

    def test_default_vectorstore_path(self) -> None:
        s = Settings(openai_api_key="test-key")
        assert s.vectorstore_path == ".migratowl/vectorstore"

    def test_default_max_retries(self) -> None:
        s = Settings(openai_api_key="test-key")
        assert s.max_retries == 2

    def test_default_confidence_threshold(self) -> None:
        s = Settings(openai_api_key="test-key")
        assert s.confidence_threshold == 0.6

    def test_default_embedding_model(self) -> None:
        s = Settings(openai_api_key="test-key")
        assert s.embedding_model == "text-embedding-3-small"

    def test_default_local_embedding_model(self) -> None:
        s = Settings(openai_api_key="test-key")
        assert s.local_embedding_model == "nomic-embed-text"


class TestSettingsEnvOverride:
    def test_env_prefix_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MIGRATOWL_OPENAI_API_KEY", "sk-test-123")
        monkeypatch.setenv("MIGRATOWL_OPENAI_MODEL", "gpt-4o")
        monkeypatch.setenv("MIGRATOWL_USE_LOCAL_LLM", "true")
        s = Settings()
        assert s.openai_api_key == "sk-test-123"
        assert s.openai_model == "gpt-4o"
        assert s.use_local_llm is True

    def test_confidence_threshold_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MIGRATOWL_OPENAI_API_KEY", "test")
        monkeypatch.setenv("MIGRATOWL_CONFIDENCE_THRESHOLD", "0.8")
        s = Settings()
        assert s.confidence_threshold == 0.8


class TestEnvFileConfig:
    def test_env_file_configured(self) -> None:
        """Settings should be configured to load from .env file."""
        config = Settings.model_config
        assert config.get("env_file") == ".env"
        assert config.get("env_file_encoding") == "utf-8"


class TestSettingsSingleton:
    def test_module_level_settings_exists(self) -> None:
        from migratowl.config import settings

        assert isinstance(settings, Settings)


class TestScalabilitySettings:
    def test_default_max_concurrent_deps(self) -> None:
        s = Settings(openai_api_key="test")
        assert s.max_concurrent_deps == 20

    def test_default_max_concurrent_registry_queries(self) -> None:
        s = Settings(openai_api_key="test")
        assert s.max_concurrent_registry_queries == 20

    def test_default_max_rag_results(self) -> None:
        s = Settings(openai_api_key="test")
        assert s.max_rag_results == 20

    def test_max_concurrent_deps_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MIGRATOWL_MAX_CONCURRENT_DEPS", "50")
        s = Settings()
        assert s.max_concurrent_deps == 50

    def test_max_rag_results_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MIGRATOWL_MAX_RAG_RESULTS", "10")
        s = Settings()
        assert s.max_rag_results == 10

    def test_default_max_concurrent_llm_calls(self) -> None:
        s = Settings(openai_api_key="test")
        assert s.max_concurrent_llm_calls == 5

    def test_max_concurrent_llm_calls_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MIGRATOWL_MAX_CONCURRENT_LLM_CALLS", "2")
        s = Settings()
        assert s.max_concurrent_llm_calls == 2
