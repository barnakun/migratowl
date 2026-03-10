"""Tests for migratowl.config — Settings with MIGRATOWL_ env prefix."""

import pytest

from migratowl.config import Settings


class TestSettingsDefaults:
    def test_default_vectorstore_path(self) -> None:
        s = Settings()
        assert s.vectorstore_path == ".migratowl/vectorstore"

    def test_default_confidence_threshold(self) -> None:
        s = Settings()
        assert s.confidence_threshold == 0.6

    def test_default_embedding_model(self) -> None:
        s = Settings()
        assert s.embedding_model == "openai:text-embedding-3-small"


class TestSettingsEnvOverride:
    def test_confidence_threshold_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
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
        s = Settings()
        assert s.max_concurrent_deps == 20

    def test_default_max_concurrent_registry_queries(self) -> None:
        s = Settings()
        assert s.max_concurrent_registry_queries == 20

    def test_default_max_rag_results(self) -> None:
        s = Settings()
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
        s = Settings()
        assert s.max_concurrent_llm_calls == 5

    def test_max_concurrent_llm_calls_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MIGRATOWL_MAX_CONCURRENT_LLM_CALLS", "2")
        s = Settings()
        assert s.max_concurrent_llm_calls == 2

    def test_default_summarize_threshold(self) -> None:
        s = Settings()
        assert s.summarize_threshold == 32_000

    def test_summarize_threshold_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MIGRATOWL_SUMMARIZE_THRESHOLD", "16000")
        s = Settings()
        assert s.summarize_threshold == 16_000


class TestIgnoredDependencies:
    def test_default_empty_string(self) -> None:
        s = Settings()
        assert s.ignored_dependencies == ""

    def test_parsed_empty_string_returns_empty_list(self) -> None:
        s = Settings(ignored_dependencies="")
        assert s.parsed_ignored_dependencies == []

    def test_parsed_comma_separated(self) -> None:
        s = Settings(ignored_dependencies="requests, flask, boto3")
        assert s.parsed_ignored_dependencies == ["requests", "flask", "boto3"]

    def test_parsed_strips_whitespace(self) -> None:
        s = Settings(ignored_dependencies="  requests , flask  ")
        assert s.parsed_ignored_dependencies == ["requests", "flask"]

    def test_parsed_skips_empty_entries(self) -> None:
        s = Settings(ignored_dependencies="requests,,flask,")
        assert s.parsed_ignored_dependencies == ["requests", "flask"]

    def test_env_var_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MIGRATOWL_IGNORED_DEPENDENCIES", "numpy,pandas")
        s = Settings()
        assert s.parsed_ignored_dependencies == ["numpy", "pandas"]


class TestModelField:
    def test_default_model_is_openai_gpt4o_mini(self) -> None:
        s = Settings()
        assert s.model == "openai:gpt-4o-mini"

    def test_model_env_var_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MIGRATOWL_MODEL", "anthropic:claude-sonnet-4-5-20250929")
        s = Settings()
        assert s.model == "anthropic:claude-sonnet-4-5-20250929"
