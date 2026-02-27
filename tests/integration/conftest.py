"""Integration test fixtures — requires Ollama running with llama3.2 + nomic-embed-text."""

import pytest


@pytest.fixture(autouse=True)
def _integration_env(monkeypatch, tmp_path):
    """Set environment for local LLM integration tests."""
    monkeypatch.setenv("MIGRATOWL_USE_LOCAL_LLM", "true")
    monkeypatch.setenv("MIGRATOWL_OPENAI_API_KEY", "")
    monkeypatch.setenv("MIGRATOWL_OPENAI_MODEL", "llama3.2")
    monkeypatch.setenv("MIGRATOWL_VECTORSTORE_PATH", str(tmp_path / "vectorstore"))


@pytest.fixture
def fixture_project(tmp_path):
    """Create a minimal Python project with an outdated dependency."""
    project_dir = tmp_path / "sample_project"
    project_dir.mkdir()

    (project_dir / "requirements.txt").write_text("requests==2.28.0\n")
    (project_dir / "app.py").write_text("import requests\n\nresponse = requests.get('https://example.com')\n")

    return project_dir
