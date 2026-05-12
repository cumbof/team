"""Shared fixtures for the team integration test suite.

Integration tests require a live Ollama instance and Docker.  They are
skipped automatically unless the ``TEAM_INTEGRATION_OLLAMA_URL`` environment
variable is set (which ``scripts/integration-test.sh`` does for you).

Set it manually to run the tests against an existing Ollama:

    TEAM_INTEGRATION_OLLAMA_URL=http://localhost:11434 pytest tests/integration/
"""

from __future__ import annotations

import os

import pytest
import requests


# ------------------------------------------------------------------
# Guard: skip the whole suite when no Ollama URL has been provided.
# ------------------------------------------------------------------

_OLLAMA_URL = os.getenv("TEAM_INTEGRATION_OLLAMA_URL", "")
_MODEL = os.getenv("TEAM_INTEGRATION_MODEL", "smollm2:135m")

if not _OLLAMA_URL:
    # Apply a module-level skip so pytest still collects the files
    # (and shows them as skipped) rather than erroring on import.
    collect_ignore_glob = ["test_*.py"]


def pytest_collection_modifyitems(config, items):
    """Skip every integration test when TEAM_INTEGRATION_OLLAMA_URL is unset."""
    if _OLLAMA_URL:
        return
    skip = pytest.mark.skip(reason="TEAM_INTEGRATION_OLLAMA_URL not set — run via scripts/integration-test.sh")
    for item in items:
        # Only touch items that live under tests/integration/.
        if "integration" in str(item.fspath):
            item.add_marker(skip)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture(scope="session")
def ollama_url() -> str:
    """The base URL of the Ollama instance used for this test session."""
    if not _OLLAMA_URL:
        pytest.skip("TEAM_INTEGRATION_OLLAMA_URL not set")
    # Quick reachability check so failures surface early with a clear message.
    try:
        r = requests.get(f"{_OLLAMA_URL}/api/tags", timeout=5)
        r.raise_for_status()
    except Exception as exc:
        pytest.fail(f"Ollama at {_OLLAMA_URL} is not reachable: {exc}")
    return _OLLAMA_URL


@pytest.fixture(scope="session")
def tiny_model() -> str:
    """The model tag used for integration tests (e.g. 'smollm2:135m')."""
    return _MODEL
