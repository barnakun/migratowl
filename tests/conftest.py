from pathlib import Path

import pytest


@pytest.fixture
def sample_project_path(tmp_path: Path) -> Path:
    """Create a minimal sample project for testing."""
    project = tmp_path / "sample-project"
    project.mkdir()
    (project / "requirements.txt").write_text("requests==2.28.0\nflask==2.3.0\n")
    (project / "main.py").write_text("import requests\nfrom flask import Flask\n")
    return project


@pytest.fixture
def sample_pyproject_path(tmp_path: Path) -> Path:
    """Create a sample project with pyproject.toml."""
    project = tmp_path / "pyproject-project"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        '[project]\nname = "test"\ndependencies = ["httpx>=0.24", "pydantic>=2.0"]\n'
    )
    return project
