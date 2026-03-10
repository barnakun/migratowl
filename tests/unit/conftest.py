from pathlib import Path

import pytest
from dotenv import dotenv_values

# Keys that .env would inject — computed once at import time.
_DOTENV_KEYS = set(dotenv_values(Path(__file__).resolve().parents[2] / ".env").keys())


@pytest.fixture(autouse=True)
def _clean_dotenv(monkeypatch, tmp_path):
    """Undo every env var that .env injected and prevent pydantic-settings from re-reading it."""
    for key in _DOTENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    # Move away from project root so pydantic-settings' env_file=".env" finds nothing.
    monkeypatch.chdir(tmp_path)
