"""Tests for the parallel workflow (F14).

Covers:
- run_parallel_round() returns results in declaration order
- run_parallel_round() actually runs members concurrently (timing test)
- run_parallel_round() raises TurnTimeoutError when a member exceeds turn_timeout
- run_parallel_round() replays cached turns (resume support)
- run_parallel_round() fires _on_parallel_round_start / _on_parallel_round_end hooks
- parallel_rounds workflow drives the correct number of rounds
- parallel_rounds workflow stops early on [[TEAM_DONE]]
- parallel_rounds workflow respects max_rounds
- "parallel" is accepted as a valid workflow type in YAML
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from team.member import TurnResult
from team.orchestrator import Orchestrator, TurnTimeoutError


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _make_team(tmp_path, max_rounds=2, member_names=("alpha", "beta"), turn_timeout=None):
    from team.config import Defaults, MemberConfig, TeamConfig, WorkflowConfig

    defaults = Defaults()
    if turn_timeout is not None:
        defaults.turn_timeout = turn_timeout

    members = [
        MemberConfig(name=n, role=n.capitalize(), model="m", persona=f"You are {n}.")
        for n in member_names
    ]
    return TeamConfig(
        name="test-parallel",
        goal="parallel test",
        workspace=tmp_path / "runs",
        workflow=WorkflowConfig(type="parallel", max_rounds=max_rounds),
        defaults=defaults,
        members=members,
    )


def _make_orch(team):
    mock_cm = MagicMock()
    orch = Orchestrator(team, container_manager=mock_cm)
    orch.workspace = MagicMock()
    orch.workspace.touch = MagicMock()
    orch.checkpoints = MagicMock()
    return orch


def _mock_member(name, content="ok", declared_done=False, delay=0.0):
    """Build a mock Member whose take_turn sleeps *delay* seconds then returns."""
    m = MagicMock()
    m.name = name
    m.config = MagicMock()
    m.config.name = name
    m.config.role = name.capitalize()
    m.config.turn_timeout = None

    def _take_turn(*args, **kwargs):
        if delay:
            time.sleep(delay)
        return TurnResult(
            content=content,
            declared_done=declared_done,
            files_written=[],
            prompt_tokens=10,
            completion_tokens=5,
        )

    m.take_turn.side_effect = _take_turn
    return m


def _attach_members(orch, members):
    for m in members:
        orch.members[m.name] = m


# --------------------------------------------------------------------------- #
# run_parallel_round — basic behaviour
# --------------------------------------------------------------------------- #


class TestRunParallelRound:
    def test_results_in_declaration_order(self, tmp_path):
        team = _make_team(tmp_path)
        orch = _make_orch(team)
        alpha = _mock_member("alpha", content="alpha reply")
        beta = _mock_member("beta", content="beta reply")
        _attach_members(orch, [alpha, beta])

        results = orch.run_parallel_round(["alpha", "beta"])

        assert len(results) == 2
        assert results[0][0] == "alpha"
        assert results[1][0] == "beta"
        assert results[0][1].content == "alpha reply"
        assert results[1][1].content == "beta reply"

    def test_both_members_actually_called(self, tmp_path):
        team = _make_team(tmp_path)
        orch = _make_orch(team)
        alpha = _mock_member("alpha")
        beta = _mock_member("beta")
        _attach_members(orch, [alpha, beta])

        orch.run_parallel_round(["alpha", "beta"])

        assert alpha.take_turn.call_count == 1
        assert beta.take_turn.call_count == 1

    def test_single_member(self, tmp_path):
        team = _make_team(tmp_path, member_names=("solo",))
        orch = _make_orch(team)
        solo = _mock_member("solo", content="alone")
        _attach_members(orch, [solo])

        results = orch.run_parallel_round(["solo"])
        assert len(results) == 1
        assert results[0][1].content == "alone"

    def test_transcript_appended_in_order(self, tmp_path):
        team = _make_team(tmp_path)
        orch = _make_orch(team)
        appended = []
        orch.transcript.append = lambda **kw: appended.append(kw["speaker"])  # type: ignore
        alpha = _mock_member("alpha")
        beta = _mock_member("beta")
        _attach_members(orch, [alpha, beta])

        orch.run_parallel_round(["alpha", "beta"])
        assert appended == ["alpha", "beta"]

    def test_token_totals_accumulated(self, tmp_path):
        team = _make_team(tmp_path)
        orch = _make_orch(team)
        alpha = _mock_member("alpha")
        beta = _mock_member("beta")
        _attach_members(orch, [alpha, beta])

        orch.run_parallel_round(["alpha", "beta"])

        assert orch._token_totals["alpha"]["prompt"] == 10
        assert orch._token_totals["alpha"]["completion"] == 5
        assert orch._token_totals["beta"]["prompt"] == 10
        assert orch._token_totals["beta"]["completion"] == 5


# --------------------------------------------------------------------------- #
# run_parallel_round — concurrency
# --------------------------------------------------------------------------- #


class TestParallelConcurrency:
    def test_runs_concurrently(self, tmp_path):
        """Two members each sleeping 0.3 s should finish in < 0.55 s total."""
        team = _make_team(tmp_path)
        orch = _make_orch(team)
        alpha = _mock_member("alpha", delay=0.3)
        beta = _mock_member("beta", delay=0.3)
        _attach_members(orch, [alpha, beta])

        start = time.monotonic()
        orch.run_parallel_round(["alpha", "beta"])
        elapsed = time.monotonic() - start

        # Sequential would take ~0.6 s; parallel should be ~0.3 s.
        assert elapsed < 0.55, f"Expected parallel execution but took {elapsed:.2f}s"

    def test_three_members_run_concurrently(self, tmp_path):
        """Three members sleeping 0.2 s each should finish in < 0.45 s."""
        team = _make_team(tmp_path, member_names=("a", "b", "c"))
        orch = _make_orch(team)
        members = [_mock_member(n, delay=0.2) for n in ("a", "b", "c")]
        _attach_members(orch, members)

        start = time.monotonic()
        orch.run_parallel_round(["a", "b", "c"])
        elapsed = time.monotonic() - start

        assert elapsed < 0.45, f"Expected parallel execution but took {elapsed:.2f}s"


# --------------------------------------------------------------------------- #
# run_parallel_round — timeout
# --------------------------------------------------------------------------- #


class TestParallelTimeout:
    def test_slow_member_raises_timeout_error(self, tmp_path):
        team = _make_team(tmp_path, turn_timeout=1)
        orch = _make_orch(team)
        alpha = _mock_member("alpha", delay=5.0)   # will time out
        beta = _mock_member("beta", content="fast")
        _attach_members(orch, [alpha, beta])

        with pytest.raises(TurnTimeoutError, match="timed out"):
            orch.run_parallel_round(["alpha", "beta"])

    def test_fast_member_no_timeout(self, tmp_path):
        team = _make_team(tmp_path, turn_timeout=30)
        orch = _make_orch(team)
        alpha = _mock_member("alpha", delay=0.05)
        beta = _mock_member("beta", delay=0.05)
        _attach_members(orch, [alpha, beta])

        results = orch.run_parallel_round(["alpha", "beta"])
        assert len(results) == 2


# --------------------------------------------------------------------------- #
# run_parallel_round — hooks
# --------------------------------------------------------------------------- #


class TestParallelHooks:
    def test_start_hook_fires_with_member_names(self, tmp_path):
        team = _make_team(tmp_path)
        orch = _make_orch(team)
        alpha = _mock_member("alpha")
        beta = _mock_member("beta")
        _attach_members(orch, [alpha, beta])

        fired_names = []
        orch._on_parallel_round_start = lambda names: fired_names.extend(names)

        orch.run_parallel_round(["alpha", "beta"])
        assert fired_names == ["alpha", "beta"]

    def test_end_hook_fires_with_results(self, tmp_path):
        team = _make_team(tmp_path)
        orch = _make_orch(team)
        alpha = _mock_member("alpha", content="A")
        beta = _mock_member("beta", content="B")
        _attach_members(orch, [alpha, beta])

        end_results = []
        orch._on_parallel_round_end = lambda r: end_results.extend(r)

        orch.run_parallel_round(["alpha", "beta"])

        assert len(end_results) == 2
        names = [n for n, _ in end_results]
        assert names == ["alpha", "beta"]


# --------------------------------------------------------------------------- #
# run_parallel_round — resume / replay
# --------------------------------------------------------------------------- #


class TestParallelReplay:
    def test_cached_members_replayed_not_re_run(self, tmp_path):
        from team.bus import Turn

        team = _make_team(tmp_path)
        orch = _make_orch(team)
        alpha = _mock_member("alpha")
        beta = _mock_member("beta")
        _attach_members(orch, [alpha, beta])

        # Pre-populate the replay queue as if coming from a resumed transcript.
        orch._replay_queue = [
            Turn(index=0, speaker="alpha", role="Alpha", content="cached alpha", files_written=[]),
            Turn(index=1, speaker="beta", role="Beta", content="cached beta", files_written=[]),
        ]

        results = orch.run_parallel_round(["alpha", "beta"])

        # Neither member's LLM should have been called.
        assert alpha.take_turn.call_count == 0
        assert beta.take_turn.call_count == 0
        assert results[0][1].content == "cached alpha"
        assert results[1][1].content == "cached beta"
        assert orch._replay_queue == []  # queue fully consumed

    def test_partial_replay_then_live(self, tmp_path):
        """alpha cached, beta runs live."""
        from team.bus import Turn

        team = _make_team(tmp_path)
        orch = _make_orch(team)
        alpha = _mock_member("alpha", content="live alpha")
        beta = _mock_member("beta", content="live beta")
        _attach_members(orch, [alpha, beta])

        orch._replay_queue = [
            Turn(index=0, speaker="alpha", role="Alpha", content="cached alpha", files_written=[]),
        ]

        results = orch.run_parallel_round(["alpha", "beta"])

        # alpha replayed, beta run live.
        assert alpha.take_turn.call_count == 0
        assert beta.take_turn.call_count == 1
        assert results[0][1].content == "cached alpha"
        assert results[1][1].content == "live beta"


# --------------------------------------------------------------------------- #
# parallel_rounds workflow
# --------------------------------------------------------------------------- #


class TestParallelRoundsWorkflow:
    def _build_orch(self, tmp_path, max_rounds=2, contents=None):
        """Build an orch with two mock members for workflow tests."""
        team = _make_team(tmp_path, max_rounds=max_rounds)
        orch = _make_orch(team)
        contents = contents or {"alpha": "ok", "beta": "ok"}
        alpha = _mock_member("alpha", content=contents["alpha"])
        beta = _mock_member("beta", content=contents["beta"])
        _attach_members(orch, [alpha, beta])
        return orch, alpha, beta

    def test_runs_all_members_each_round(self, tmp_path):
        orch, alpha, beta = self._build_orch(tmp_path, max_rounds=2)
        from team.workflows import parallel_rounds
        parallel_rounds(orch)
        assert alpha.take_turn.call_count == 2
        assert beta.take_turn.call_count == 2

    def test_stops_on_declared_done(self, tmp_path):
        orch, alpha, beta = self._build_orch(
            tmp_path, max_rounds=10,
            contents={"alpha": "ok", "beta": "[[TEAM_DONE]]"},
        )
        beta._mock_member_done = True
        # Reconfigure beta to declare done.
        beta.take_turn.side_effect = lambda *a, **kw: TurnResult(
            content="[[TEAM_DONE]]", declared_done=True, files_written=[],
        )

        from team.workflows import parallel_rounds
        parallel_rounds(orch)

        # Should stop after the first round (beta declared done).
        assert alpha.take_turn.call_count == 1
        assert beta.take_turn.call_count == 1

    def test_round_end_hook_fires(self, tmp_path):
        orch, _, _ = self._build_orch(tmp_path, max_rounds=3)
        round_indices = []
        orch._on_round_end = lambda idx: round_indices.append(idx)

        from team.workflows import parallel_rounds
        parallel_rounds(orch)

        assert round_indices == [0, 1, 2]

    def test_respects_max_rounds(self, tmp_path):
        orch, alpha, beta = self._build_orch(tmp_path, max_rounds=1)
        from team.workflows import parallel_rounds
        parallel_rounds(orch)
        assert alpha.take_turn.call_count == 1
        assert beta.take_turn.call_count == 1


# --------------------------------------------------------------------------- #
# Config: "parallel" accepted as valid workflow type
# --------------------------------------------------------------------------- #


class TestParallelWorkflowConfig:
    def test_parallel_workflow_type_valid(self, tmp_path):
        from team.config import load_team
        y = tmp_path / "team.yaml"
        y.write_text(f"""\
name: parallel-team
goal: parallel test
workspace: {tmp_path}/runs

workflow:
  type: parallel
  max_rounds: 3

members:
  - name: thinker
    role: Thinker
    model: llama3
    persona: You think.
  - name: builder
    role: Builder
    model: llama3
    persona: You build.
""")
        cfg = load_team(str(y))
        assert cfg.workflow.type == "parallel"
        assert cfg.workflow.max_rounds == 3

    def test_parallel_registered_in_workflows(self):
        from team.workflows import WORKFLOWS
        assert "parallel" in WORKFLOWS
