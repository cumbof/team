"""Tests for tool_sandbox security guardrails.

Covers:
* Config parsing: tool_sandbox field in Defaults and MemberConfig.
* _build_sandboxed_cmd: correct command wrapping for firejail and bubblewrap;
  fallback to bare command when binary is absent.
* Member.__init__: security warning fires when run_python/run_bash are enabled
  in host-Ollama mode without a sandbox; suppressed in Docker mode or when a
  sandbox is set.
* execute_tool: sandbox kwarg is forwarded to _run_python / _run_bash.
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from team.config import Defaults, MemberConfig, TeamConfig, WorkflowConfig
from team.tools import _VALID_TOOL_SANDBOXES, _build_sandboxed_cmd, execute_tool


# --------------------------------------------------------------------------- #
# Config defaults
# --------------------------------------------------------------------------- #


def test_defaults_tool_sandbox_is_none_string():
    assert Defaults().tool_sandbox == "none"


def test_member_tool_sandbox_defaults_to_none():
    cfg = MemberConfig(name="bob", role="R", model="m", persona="p")
    assert cfg.tool_sandbox is None


def test_valid_sandboxes_set():
    assert "none" in _VALID_TOOL_SANDBOXES
    assert "firejail" in _VALID_TOOL_SANDBOXES
    assert "bubblewrap" in _VALID_TOOL_SANDBOXES


# --------------------------------------------------------------------------- #
# _build_sandboxed_cmd
# --------------------------------------------------------------------------- #


def test_none_sandbox_returns_cmd_unchanged(tmp_path):
    cmd = ["python", "script.py"]
    result = _build_sandboxed_cmd(cmd, tmp_path, "none")
    assert result == cmd


def test_firejail_prepends_when_available(tmp_path):
    cmd = ["python", "script.py"]
    with patch("team.tools._sandbox_available", return_value=True):
        result = _build_sandboxed_cmd(cmd, tmp_path, "firejail")
    assert result[0] == "firejail"
    assert "python" in result
    assert "script.py" in result


def test_firejail_includes_workspace_whitelist(tmp_path):
    cmd = ["python", "script.py"]
    with patch("team.tools._sandbox_available", return_value=True):
        result = _build_sandboxed_cmd(cmd, tmp_path, "firejail")
    assert any(str(tmp_path) in arg for arg in result)


def test_firejail_fallback_when_binary_missing(tmp_path, caplog):
    cmd = ["python", "script.py"]
    with patch("team.tools._sandbox_available", return_value=False):
        with caplog.at_level(logging.WARNING, logger="team.tools"):
            result = _build_sandboxed_cmd(cmd, tmp_path, "firejail")
    assert result == cmd
    assert "firejail" in caplog.text.lower()


def test_bubblewrap_prepends_bwrap_when_available(tmp_path):
    cmd = ["python", "script.py"]
    with patch("team.tools._sandbox_available", return_value=True):
        result = _build_sandboxed_cmd(cmd, tmp_path, "bubblewrap")
    assert result[0] == "bwrap"
    assert "python" in result
    assert "script.py" in result


def test_bubblewrap_includes_workspace_bind(tmp_path):
    cmd = ["python", "script.py"]
    with patch("team.tools._sandbox_available", return_value=True):
        result = _build_sandboxed_cmd(cmd, tmp_path, "bubblewrap")
    joined = " ".join(result)
    assert str(tmp_path) in joined


def test_bubblewrap_fallback_when_binary_missing(tmp_path, caplog):
    cmd = ["python", "script.py"]
    with patch("team.tools._sandbox_available", return_value=False):
        with caplog.at_level(logging.WARNING, logger="team.tools"):
            result = _build_sandboxed_cmd(cmd, tmp_path, "bubblewrap")
    assert result == cmd
    assert "bubblewrap" in caplog.text.lower()


def test_unknown_sandbox_falls_back(tmp_path, caplog):
    cmd = ["python", "script.py"]
    with caplog.at_level(logging.WARNING, logger="team.tools"):
        result = _build_sandboxed_cmd(cmd, tmp_path, "nonexistent_sandbox")
    assert result == cmd


# --------------------------------------------------------------------------- #
# execute_tool passes sandbox to _run_python / _run_bash
# --------------------------------------------------------------------------- #


def test_execute_run_python_receives_sandbox(tmp_path):
    """sandbox kwarg must reach the _run_python implementation."""
    captured: list[str] = []

    def _fake_run_python(body, *, sandbox="none", **kwargs):
        captured.append(sandbox)
        return "ok"

    fake_tools = {"run_python": _fake_run_python}
    execute_tool("run_python", "print(1)", tools=fake_tools, workspace_path=tmp_path, sandbox="firejail")
    assert captured == ["firejail"]


def test_execute_run_bash_receives_sandbox(tmp_path):
    captured: list[str] = []

    def _fake_run_bash(body, *, sandbox="none", **kwargs):
        captured.append(sandbox)
        return "ok"

    fake_tools = {"run_bash": _fake_run_bash}
    execute_tool("run_bash", "echo hi", tools=fake_tools, workspace_path=tmp_path, sandbox="bubblewrap")
    assert captured == ["bubblewrap"]


# --------------------------------------------------------------------------- #
# Member.__init__ security warning
# --------------------------------------------------------------------------- #


def _make_member_with_tools(
    tmp_path: Path,
    *,
    tools: list[str],
    ollama_url: str | None = None,
    default_ollama_url: str | None = None,
    tool_sandbox: str | None = None,
    default_tool_sandbox: str = "none",
):
    """Build a Member instance for warning tests."""
    from team.member import Member

    cfg = MemberConfig(
        name="agent",
        role="Coder",
        model="m",
        persona="a coder",
        tools=tools,
        ollama_url=ollama_url,
        tool_sandbox=tool_sandbox,
    )
    defaults = Defaults(
        ollama_url=default_ollama_url,
        tool_sandbox=default_tool_sandbox,
    )
    team = TeamConfig(
        name="t",
        goal="g",
        workspace=tmp_path,
        workflow=WorkflowConfig(),
        defaults=defaults,
        members=[cfg],
    )
    runtime = MagicMock()
    runtime.base_url = "http://localhost:11434"

    with (
        patch("team.member.OllamaClient"),
        patch("team.member.OpenAICompatClient"),
        patch("team.member.load_skills", return_value=({}, {}, [])),
        patch("team.member.render_system_prompt", return_value="sys"),
    ):
        return Member(team, cfg, runtime)


def test_warning_fires_for_code_tools_in_host_ollama_mode(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="team.member"):
        _make_member_with_tools(
            tmp_path,
            tools=["run_python"],
            ollama_url="http://localhost:11434",
        )
    assert "SECURITY" in caplog.text
    assert "run_python" in caplog.text


def test_warning_fires_for_bash_tool_in_host_ollama_mode(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="team.member"):
        _make_member_with_tools(
            tmp_path,
            tools=["run_bash"],
            default_ollama_url="http://localhost:11434",
        )
    assert "SECURITY" in caplog.text


def test_no_warning_in_docker_mode(tmp_path, caplog):
    """When ollama_url is not set (Docker mode), no warning should fire."""
    with caplog.at_level(logging.WARNING, logger="team.member"):
        _make_member_with_tools(
            tmp_path,
            tools=["run_python", "run_bash"],
            ollama_url=None,
            default_ollama_url=None,
        )
    assert "SECURITY" not in caplog.text


def test_no_warning_when_sandbox_is_set(tmp_path, caplog):
    """A sandbox setting suppresses the advisory warning."""
    with caplog.at_level(logging.WARNING, logger="team.member"):
        _make_member_with_tools(
            tmp_path,
            tools=["run_python"],
            ollama_url="http://localhost:11434",
            tool_sandbox="firejail",
        )
    assert "SECURITY" not in caplog.text


def test_no_warning_for_safe_tools_in_host_ollama_mode(tmp_path, caplog):
    """Non-code tools (web_search, read_file, …) do not trigger the warning."""
    with caplog.at_level(logging.WARNING, logger="team.member"):
        _make_member_with_tools(
            tmp_path,
            tools=["web_search", "read_file"],
            ollama_url="http://localhost:11434",
        )
    assert "SECURITY" not in caplog.text


def test_invalid_sandbox_value_logs_warning_and_falls_back(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="team.member"):
        m = _make_member_with_tools(
            tmp_path,
            tools=[],
            tool_sandbox="invalid_value",
        )
    assert "unknown tool_sandbox" in caplog.text
    assert m._tool_sandbox == "none"


def test_member_tool_sandbox_stored(tmp_path):
    m = _make_member_with_tools(
        tmp_path,
        tools=[],
        tool_sandbox="firejail",
    )
    assert m._tool_sandbox == "firejail"


def test_default_sandbox_inherited_when_member_not_set(tmp_path):
    m = _make_member_with_tools(
        tmp_path,
        tools=[],
        tool_sandbox=None,
        default_tool_sandbox="bubblewrap",
    )
    assert m._tool_sandbox == "bubblewrap"
