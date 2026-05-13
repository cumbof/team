"""Tests for orchestrator diagnostic warnings.

Covers:
- Context-window saturation warning emitted by run_turn() when
  prompt_tokens >= 90 % of the member's effective context_window.
- End-of-run warnings emitted by _check_run_outcome() when the workflow
  finishes without [[TEAM_DONE]] and/or without any files being written.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from team.config import Defaults, MemberConfig, TeamConfig, WorkflowConfig
from team.member import TurnResult
from team.orchestrator import Orchestrator


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _make_team(tmp_path, *, context_window: int = 1024, member_ctx: int | None = None):
    defaults = Defaults()
    defaults.context_window = context_window

    member = MemberConfig(
        name="worker",
        role="Worker",
        model="m",
        persona="You work.",
        context_window=member_ctx,
    )
    return TeamConfig(
        name="t",
        goal="g",
        workspace=tmp_path / "ws",
        workflow=WorkflowConfig(type="round_robin", max_rounds=1),
        defaults=defaults,
        members=[member],
    )


def _make_orch(team) -> Orchestrator:
    mock_cm = MagicMock()
    orch = Orchestrator(team, container_manager=mock_cm)
    orch.workspace = MagicMock()
    orch.workspace.touch = MagicMock()
    orch.workspace.list_files.return_value = []
    orch.workspace.recent_changes.return_value = []
    orch.workspace.shared_dir = team.workspace
    orch.checkpoints = MagicMock()
    return orch


def _mock_member(name: str, *, prompt_tokens: int, completion_tokens: int = 5,
                 files_written: list | None = None, declared_done: bool = False,
                 context_window: int | None = None) -> MagicMock:
    m = MagicMock()
    m.name = name
    m.config = MagicMock()
    m.config.name = name
    m.config.role = name.capitalize()
    m.config.turn_timeout = None
    m.config.token_budget = None
    m.config.context_window = context_window
    m.take_turn.return_value = TurnResult(
        content="[[TEAM_DONE]]" if declared_done else "reply",
        declared_done=declared_done,
        files_written=files_written or [],
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    return m


# --------------------------------------------------------------------------- #
# Context-window saturation warning
# --------------------------------------------------------------------------- #


class TestContextSaturationWarning:
    def test_warning_at_90_percent(self, tmp_path, caplog):
        """Warning fires when prompt_tokens >= 90 % of context_window."""
        team = _make_team(tmp_path, context_window=1024)
        orch = _make_orch(team)
        worker = _mock_member("worker", prompt_tokens=922)  # 922/1024 ≈ 90 %
        orch.members["worker"] = worker

        with caplog.at_level(logging.WARNING, logger="team.orchestrator"):
            orch.run_turn("worker")

        assert "context window near-full" in caplog.text
        assert "922" in caplog.text

    def test_no_warning_below_threshold(self, tmp_path, caplog):
        """No warning when prompt_tokens < 90 % of context_window."""
        team = _make_team(tmp_path, context_window=1024)
        orch = _make_orch(team)
        worker = _mock_member("worker", prompt_tokens=800)  # 800/1024 ≈ 78 %
        orch.members["worker"] = worker

        with caplog.at_level(logging.WARNING, logger="team.orchestrator"):
            orch.run_turn("worker")

        assert "context window near-full" not in caplog.text

    def test_warning_uses_member_override(self, tmp_path, caplog):
        """Per-member context_window override is used instead of the default."""
        # default=4096 but member override=512 — 470/512 ≈ 91 % → should warn
        team = _make_team(tmp_path, context_window=4096, member_ctx=512)
        orch = _make_orch(team)
        worker = _mock_member("worker", prompt_tokens=470, context_window=512)
        orch.members["worker"] = worker

        with caplog.at_level(logging.WARNING, logger="team.orchestrator"):
            orch.run_turn("worker")

        assert "context window near-full" in caplog.text

    def test_no_warning_when_context_window_zero(self, tmp_path, caplog):
        """No division-by-zero / spurious warning when context_window is 0."""
        team = _make_team(tmp_path, context_window=0)
        orch = _make_orch(team)
        worker = _mock_member("worker", prompt_tokens=100)
        orch.members["worker"] = worker

        with caplog.at_level(logging.WARNING, logger="team.orchestrator"):
            orch.run_turn("worker")

        assert "context window near-full" not in caplog.text


# --------------------------------------------------------------------------- #
# End-of-run outcome warning
# --------------------------------------------------------------------------- #


class TestCheckRunOutcome:
    def _run_outcome(self, orch, declared_done: bool, files_written: list) -> None:
        """Directly populate a minimal transcript and call _check_run_outcome."""
        from team.bus import Turn
        orch.transcript.turns = [
            Turn(
                index=1,
                speaker="worker",
                role="Worker",
                content="[[TEAM_DONE]]" if declared_done else "reply",
                files_written=files_written,
            )
        ]
        orch._check_run_outcome()

    def test_no_done_no_files_warns(self, tmp_path, caplog):
        """Both warnings fire when neither TEAM_DONE nor files are present."""
        team = _make_team(tmp_path)
        orch = _make_orch(team)

        with caplog.at_level(logging.WARNING, logger="team.orchestrator"):
            self._run_outcome(orch, declared_done=False, files_written=[])

        assert "[[TEAM_DONE]]" in caplog.text
        assert "files" in caplog.text.lower()

    def test_no_done_but_files_warns_only_done(self, tmp_path, caplog):
        """Only the no-TEAM_DONE warning fires when files exist but done is absent."""
        team = _make_team(tmp_path)
        orch = _make_orch(team)

        with caplog.at_level(logging.WARNING, logger="team.orchestrator"):
            self._run_outcome(orch, declared_done=False, files_written=["output.py"])

        assert "[[TEAM_DONE]]" in caplog.text
        # The "no files" part of the message should NOT appear
        assert "without any workspace files" not in caplog.text

    def test_done_emitted_no_warning(self, tmp_path, caplog):
        """No warning fires when TEAM_DONE was emitted."""
        team = _make_team(tmp_path)
        orch = _make_orch(team)

        with caplog.at_level(logging.WARNING, logger="team.orchestrator"):
            self._run_outcome(orch, declared_done=True, files_written=[])

        assert "Run ended without" not in caplog.text

    def test_empty_transcript_no_warning(self, tmp_path, caplog):
        """No warning fires when there are no live turns (e.g. dry-run)."""
        team = _make_team(tmp_path)
        orch = _make_orch(team)
        orch.transcript.turns = []

        with caplog.at_level(logging.WARNING, logger="team.orchestrator"):
            orch._check_run_outcome()

        assert "Run ended without" not in caplog.text
