"""Tests for tool_sandbox security guardrails (orchestrator-level after MCP cutover).

Covers:
* Config parsing: tool_sandbox field in Defaults and MemberConfig.
* Orchestrator._build_member_toolset: security warning fires when code tools are
  enabled in host-Ollama mode without a sandbox; suppressed in Docker mode or
  when a sandbox is set; unknown sandbox values fall back to 'none'.

(The _build_sandboxed_cmd command-wrapping tests live in test_builtin_servers.py,
against team.mcp.builtin.code.)
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from team.config import Defaults, MemberConfig, TeamConfig, WorkflowConfig
from team.mcp.builtin.code import VALID_TOOL_SANDBOXES
from team.orchestrator import Orchestrator


# --------------------------------------------------------------------------- #
# Config defaults
# --------------------------------------------------------------------------- #


def test_defaults_tool_sandbox_is_none_string():
    assert Defaults().tool_sandbox == "none"


def test_member_tool_sandbox_defaults_to_none():
    cfg = MemberConfig(name="bob", role="R", model="m", persona="p")
    assert cfg.tool_sandbox is None


def test_valid_sandboxes_set():
    assert VALID_TOOL_SANDBOXES == {"none", "firejail", "bubblewrap"}


# --------------------------------------------------------------------------- #
# Orchestrator security warning
# --------------------------------------------------------------------------- #


def _build_toolset(
    tmp_path: Path,
    *,
    tools: list[str],
    ollama_url: str | None = None,
    default_ollama_url: str | None = None,
    tool_sandbox: str | None = None,
    default_tool_sandbox: str = "none",
):
    """Build a member's toolset through a real (bus-backed) Orchestrator."""
    cfg = MemberConfig(
        name="agent", role="Coder", model="m", persona="a coder",
        tools=tools, ollama_url=ollama_url, tool_sandbox=tool_sandbox,
    )
    defaults = Defaults(ollama_url=default_ollama_url, tool_sandbox=default_tool_sandbox)
    team = TeamConfig(
        name="t", goal="g", workspace=tmp_path,
        workflow=WorkflowConfig(), defaults=defaults, members=[cfg],
    )
    orch = Orchestrator(team, container_manager=MagicMock())
    orch.tool_bus.start()
    try:
        return orch._build_member_toolset(cfg, None)
    finally:
        orch.tool_bus.stop()


def test_warning_fires_for_code_tools_in_host_ollama_mode(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="team.orchestrator"):
        _build_toolset(tmp_path, tools=["code/run_python"], ollama_url="http://localhost:11434")
    assert "SECURITY" in caplog.text


def test_warning_fires_for_bash_tool_in_host_ollama_mode(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="team.orchestrator"):
        _build_toolset(tmp_path, tools=["code/run_bash"], default_ollama_url="http://localhost:11434")
    assert "SECURITY" in caplog.text


def test_no_warning_in_docker_mode(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="team.orchestrator"):
        _build_toolset(tmp_path, tools=["code/*"], ollama_url=None, default_ollama_url=None)
    assert "SECURITY" not in caplog.text


def test_no_warning_when_sandbox_is_set(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="team.orchestrator"):
        _build_toolset(
            tmp_path, tools=["code/run_python"],
            ollama_url="http://localhost:11434", tool_sandbox="firejail",
        )
    assert "SECURITY" not in caplog.text


def test_no_warning_for_safe_tools_in_host_ollama_mode(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="team.orchestrator"):
        _build_toolset(
            tmp_path, tools=["web/web_search", "workspace/read_file"],
            ollama_url="http://localhost:11434",
        )
    assert "SECURITY" not in caplog.text


def test_invalid_sandbox_value_logs_warning_and_falls_back(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="team.orchestrator"):
        _build_toolset(tmp_path, tools=["web/*"], tool_sandbox="invalid_value")
    assert "unknown tool_sandbox" in caplog.text
