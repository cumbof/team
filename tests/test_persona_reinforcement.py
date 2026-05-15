"""Tests for the persona reinforcement feature.

When ``persona_reinforcement`` is enabled (via member config or team defaults),
a ``## Persona reminder`` block is injected into the user message on every turn.
This prevents smaller models from drifting away from their assigned identity
over the course of long conversations.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from team.config import Defaults, MemberConfig, TeamConfig, WorkflowConfig


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _make_team(
    workspace: Path,
    *,
    member_reinforcement: bool | None = None,
    default_reinforcement: bool = False,
) -> TeamConfig:
    member_cfg = MemberConfig(
        name="alice",
        role="Researcher",
        model="llama3.1:8b",
        persona="You are a meticulous researcher who always cites sources.",
        persona_reinforcement=member_reinforcement,
    )
    defaults = Defaults(persona_reinforcement=default_reinforcement)
    return TeamConfig(
        name="test-team",
        goal="Do science",
        workspace=workspace,
        workflow=WorkflowConfig(type="round_robin", max_rounds=1),
        defaults=defaults,
        members=[member_cfg],
    )


def _make_member(team: TeamConfig):
    from team.member import Member

    cfg = team.members[0]
    runtime = MagicMock()
    runtime.base_url = "http://localhost:11434"
    runtime.member = cfg

    with (
        patch("team.member.OllamaClient"),
        patch("team.member.OpenAICompatClient"),
        patch("team.member.load_skills", return_value=({}, {}, [])),
        patch("team.member.render_system_prompt", return_value="system"),
    ):
        return Member(team, cfg, runtime)


def _build_user_message(member, tmp_path: Path) -> str:
    transcript = MagicMock()
    transcript.turns = []
    transcript.render = MagicMock(return_value="")

    workspace = MagicMock()
    workspace.list_files.return_value = []
    workspace.recent_changes.return_value = []

    messages = member._build_messages(transcript, workspace, prompt=None)
    return messages[1].content


# --------------------------------------------------------------------------- #
# Config parsing
# --------------------------------------------------------------------------- #


def test_default_persona_reinforcement_is_false():
    assert Defaults().persona_reinforcement is False


def test_member_persona_reinforcement_defaults_to_none():
    cfg = MemberConfig(name="bob", role="R", model="m", persona="p")
    assert cfg.persona_reinforcement is None


def test_member_reinforcement_overrides_default(tmp_path):
    team = _make_team(tmp_path, member_reinforcement=True, default_reinforcement=False)
    member = _make_member(team)
    user_msg = _build_user_message(member, tmp_path)
    assert "Persona reminder" in user_msg


def test_default_reinforcement_applies_when_member_not_set(tmp_path):
    team = _make_team(tmp_path, member_reinforcement=None, default_reinforcement=True)
    member = _make_member(team)
    user_msg = _build_user_message(member, tmp_path)
    assert "Persona reminder" in user_msg


# --------------------------------------------------------------------------- #
# Content of the reminder block
# --------------------------------------------------------------------------- #


def test_persona_reminder_includes_name_and_role(tmp_path):
    team = _make_team(tmp_path, member_reinforcement=True)
    member = _make_member(team)
    user_msg = _build_user_message(member, tmp_path)
    assert "@alice" in user_msg
    assert "Researcher" in user_msg


def test_persona_reminder_includes_persona_text(tmp_path):
    team = _make_team(tmp_path, member_reinforcement=True)
    member = _make_member(team)
    user_msg = _build_user_message(member, tmp_path)
    assert "meticulous researcher" in user_msg


def test_persona_reminder_appears_before_your_turn(tmp_path):
    team = _make_team(tmp_path, member_reinforcement=True)
    member = _make_member(team)
    user_msg = _build_user_message(member, tmp_path)
    idx_reminder = user_msg.find("Persona reminder")
    idx_turn = user_msg.find("## Your turn")
    assert idx_reminder != -1
    assert idx_turn != -1
    assert idx_reminder < idx_turn, "Persona reminder should appear before '## Your turn'"


# --------------------------------------------------------------------------- #
# Feature disabled
# --------------------------------------------------------------------------- #


def test_no_reminder_when_disabled_by_default(tmp_path):
    team = _make_team(tmp_path, member_reinforcement=None, default_reinforcement=False)
    member = _make_member(team)
    user_msg = _build_user_message(member, tmp_path)
    assert "Persona reminder" not in user_msg


def test_no_reminder_when_member_explicitly_disables(tmp_path):
    team = _make_team(tmp_path, member_reinforcement=False, default_reinforcement=True)
    member = _make_member(team)
    user_msg = _build_user_message(member, tmp_path)
    assert "Persona reminder" not in user_msg
