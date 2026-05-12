"""Tests for per-turn timeout (F12).

Covers:
- TurnTimeoutError is raised when take_turn exceeds turn_timeout
- Workflow completes normally when turn finishes within the timeout
- turn_timeout in Defaults flows through resolve_member_setting
- Member-level turn_timeout overrides Defaults
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from team.orchestrator import Orchestrator, TurnTimeoutError


def _make_team(tmp_path, turn_timeout_default=None, member_turn_timeout=None):
    """Build a minimal TeamConfig with turn_timeout settings."""
    from team.config import Defaults, MemberConfig, TeamConfig, WorkflowConfig

    defaults = Defaults()
    if turn_timeout_default is not None:
        defaults.turn_timeout = turn_timeout_default

    member = MemberConfig(
        name="worker",
        role="Worker",
        model="llama3",
        persona="You work.",
    )
    if member_turn_timeout is not None:
        member.turn_timeout = member_turn_timeout

    return TeamConfig(
        name="test-team",
        goal="test goal",
        workspace=tmp_path / "runs",
        workflow=WorkflowConfig(type="round_robin", max_rounds=1),
        defaults=defaults,
        members=[member],
    )


def _fast_take_turn(*args, **kwargs):
    from team.member import TurnResult
    return TurnResult(content="done", declared_done=True, files_written=[])


def _slow_take_turn(*args, **kwargs):
    time.sleep(5)
    from team.member import TurnResult
    return TurnResult(content="done", declared_done=True, files_written=[])


def _make_orch(team):
    """Create an Orchestrator with a mocked ContainerManager (no Docker required)."""
    mock_cm = MagicMock()
    orch = Orchestrator(team, container_manager=mock_cm)
    orch.workspace = MagicMock()
    orch.transcript = MagicMock()
    orch.transcript.turns = []
    orch.checkpoints = MagicMock()
    return orch


class TestTurnTimeout:
    def test_timeout_raises(self, tmp_path):
        team = _make_team(tmp_path, turn_timeout_default=1)
        orch = _make_orch(team)

        mock_member = MagicMock()
        mock_member.name = "worker"
        mock_member.config = team.members[0]
        mock_member.take_turn.side_effect = _slow_take_turn
        orch.members["worker"] = mock_member

        with pytest.raises(TurnTimeoutError, match="timed out"):
            orch.run_turn("worker")

    def test_no_timeout_when_zero(self, tmp_path):
        """turn_timeout=0 means disabled — fast turn should complete fine."""
        team = _make_team(tmp_path, turn_timeout_default=0)
        orch = _make_orch(team)

        mock_member = MagicMock()
        mock_member.name = "worker"
        mock_member.config = team.members[0]
        mock_member.take_turn.side_effect = _fast_take_turn
        orch.members["worker"] = mock_member

        result = orch.run_turn("worker")
        assert result.content == "done"

    def test_no_timeout_when_none(self, tmp_path):
        """turn_timeout=None means disabled — fast turn completes normally."""
        team = _make_team(tmp_path)  # no timeout set
        orch = _make_orch(team)

        mock_member = MagicMock()
        mock_member.name = "worker"
        mock_member.config = team.members[0]
        mock_member.take_turn.side_effect = _fast_take_turn
        orch.members["worker"] = mock_member

        result = orch.run_turn("worker")
        assert result.content == "done"

    def test_member_override_wins(self, tmp_path):
        """Member-level turn_timeout = 1 overrides default = 999."""
        team = _make_team(tmp_path, turn_timeout_default=999, member_turn_timeout=1)
        orch = _make_orch(team)

        mock_member = MagicMock()
        mock_member.name = "worker"
        mock_member.config = team.members[0]
        mock_member.take_turn.side_effect = _slow_take_turn
        orch.members["worker"] = mock_member

        with pytest.raises(TurnTimeoutError):
            orch.run_turn("worker")

    def test_fast_turn_within_generous_timeout(self, tmp_path):
        """Fast turn completes well within a generous timeout."""
        team = _make_team(tmp_path, turn_timeout_default=30)
        orch = _make_orch(team)

        mock_member = MagicMock()
        mock_member.name = "worker"
        mock_member.config = team.members[0]
        mock_member.take_turn.side_effect = _fast_take_turn
        orch.members["worker"] = mock_member

        result = orch.run_turn("worker")
        assert result.content == "done"
        assert result.declared_done is True
