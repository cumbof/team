"""Tests for the shared context file (context.md) injection feature.

The context file lives at ``{workspace_root}/context.md`` and is automatically
injected into every member's turn context when present.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from team.config import Defaults, MemberConfig, TeamConfig, WorkflowConfig
from team.member import Member


def _make_team(workspace: Path) -> TeamConfig:
    member_cfg = MemberConfig(
        name="alice",
        role="Researcher",
        model="llama3.1:8b",
        persona="You are a researcher.",
    )
    return TeamConfig(
        name="test-team",
        goal="Do science",
        workspace=workspace,
        workflow=WorkflowConfig(type="round_robin", max_rounds=1),
        defaults=Defaults(),
        members=[member_cfg],
    )


def _make_member(team: TeamConfig) -> Member:
    cfg = team.members[0]
    runtime = MagicMock()
    runtime.base_url = "http://localhost:11434"
    runtime.member = cfg

    with patch("team.member.OllamaClient"), \
         patch("team.member.OpenAICompatClient"), \
         patch("team.member.load_skills", return_value=({}, {})), \
         patch("team.member.render_system_prompt", return_value="system"):
        m = Member(team, cfg, runtime)
    return m


# --------------------------------------------------------------------------- #
# _load_context_file
# --------------------------------------------------------------------------- #


def test_no_context_file_returns_none(tmp_path):
    team = _make_team(tmp_path)
    member = _make_member(team)
    assert member._load_context_file() is None


def test_context_file_content_returned(tmp_path):
    ctx = tmp_path / "context.md"
    ctx.write_text("# Lab context\nWe use pandas for all data wrangling.", encoding="utf-8")
    team = _make_team(tmp_path)
    member = _make_member(team)
    result = member._load_context_file()
    assert result is not None
    assert "pandas" in result
    assert "Lab context" in result


def test_context_file_truncated_at_limit(tmp_path):
    ctx = tmp_path / "context.md"
    # Write more than 8192 chars
    ctx.write_text("x" * 10_000, encoding="utf-8")
    team = _make_team(tmp_path)
    member = _make_member(team)
    result = member._load_context_file()
    assert result is not None
    assert "truncated" in result
    assert len(result) < 10_000


def test_context_file_strips_whitespace(tmp_path):
    ctx = tmp_path / "context.md"
    ctx.write_text("\n\nHello context\n\n", encoding="utf-8")
    team = _make_team(tmp_path)
    member = _make_member(team)
    result = member._load_context_file()
    assert result == "Hello context"


# --------------------------------------------------------------------------- #
# _build_messages injects context file
# --------------------------------------------------------------------------- #


def test_build_messages_includes_context_section(tmp_path):
    ctx = tmp_path / "context.md"
    ctx.write_text("Domain expertise: proteomics", encoding="utf-8")
    team = _make_team(tmp_path)
    member = _make_member(team)

    transcript = MagicMock()
    transcript.turns = []
    transcript.render = MagicMock(return_value="")

    workspace = MagicMock()
    workspace.list_files.return_value = []
    workspace.recent_changes.return_value = []

    messages = member._build_messages(transcript, workspace, prompt=None)
    # The user message (index 1) should contain context.md content
    user_msg = messages[1].content
    assert "proteomics" in user_msg
    assert "institutional context" in user_msg.lower()


def test_build_messages_no_context_section_when_file_absent(tmp_path):
    team = _make_team(tmp_path)
    member = _make_member(team)

    transcript = MagicMock()
    transcript.turns = []
    transcript.render = MagicMock(return_value="")

    workspace = MagicMock()
    workspace.list_files.return_value = []
    workspace.recent_changes.return_value = []

    messages = member._build_messages(transcript, workspace, prompt=None)
    user_msg = messages[1].content
    assert "institutional context" not in user_msg.lower()


def test_context_file_appears_before_workspace_listing(tmp_path):
    ctx = tmp_path / "context.md"
    ctx.write_text("INSTITUTIONAL_MARKER", encoding="utf-8")
    team = _make_team(tmp_path)
    member = _make_member(team)

    transcript = MagicMock()
    transcript.turns = []
    transcript.render = MagicMock(return_value="")

    workspace = MagicMock()
    workspace.list_files.return_value = ["data.csv"]
    workspace.recent_changes.return_value = []

    messages = member._build_messages(transcript, workspace, prompt=None)
    user_msg = messages[1].content
    idx_ctx = user_msg.find("INSTITUTIONAL_MARKER")
    idx_ws = user_msg.find("data.csv")
    assert idx_ctx != -1
    assert idx_ws != -1
    assert idx_ctx < idx_ws, "context.md content should appear before workspace listing"
