"""Tests for the new CLI commands: team rollback and team beliefs."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner

from team.beliefs import BeliefBoard
from team.cli import cli
from team.workspace import CheckpointManager


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _make_team_yaml(tmp_path: Path, name: str = "test-team") -> Path:
    """Write a minimal valid team YAML and return its path."""
    workspace = tmp_path / "runs" / name
    workspace.mkdir(parents=True, exist_ok=True)
    yaml_content = f"""\
name: {name}
goal: test goal
workspace: {workspace}
workflow:
  type: round_robin
  max_rounds: 2
members:
  - name: alice
    role: Researcher
    model: llama3.2:3b
    persona: You are Alice.
  - name: bob
    role: Analyst
    model: llama3.2:3b
    persona: You are Bob.
beliefs:
  enabled: true
  consensus_threshold: 0.5
"""
    p = tmp_path / f"{name}.yaml"
    p.write_text(yaml_content, encoding="utf-8")
    return p


def _make_checkpoints(workspace: Path, count: int = 3) -> list[str]:
    """Create fake checkpoint directories and return their IDs."""
    shared = workspace / "shared"
    shared.mkdir(parents=True, exist_ok=True)
    (shared / "sample.txt").write_text("content", encoding="utf-8")

    mgr = CheckpointManager(workspace)
    ids = []
    for i in range(count):
        cp = mgr.create(turn_index=i, member_name="alice")
        if cp:
            ids.append(cp.name)
    return ids


# --------------------------------------------------------------------------- #
# team rollback — list mode
# --------------------------------------------------------------------------- #


def test_rollback_list_no_checkpoints(tmp_path):
    p = _make_team_yaml(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli, ["rollback", str(p)])
    assert result.exit_code == 0
    assert "no checkpoints" in result.output.lower()


def test_rollback_list_shows_table(tmp_path):
    p = _make_team_yaml(tmp_path)
    workspace = tmp_path / "runs" / "test-team"
    ids = _make_checkpoints(workspace, count=2)

    runner = CliRunner()
    result = runner.invoke(cli, ["rollback", str(p)])
    assert result.exit_code == 0
    # The table should show the member name and checkpoint count.
    assert "alice" in result.output
    assert "Checkpoints" in result.output


# --------------------------------------------------------------------------- #
# team rollback — restore mode
# --------------------------------------------------------------------------- #


def test_rollback_to_with_yes(tmp_path):
    p = _make_team_yaml(tmp_path)
    workspace = tmp_path / "runs" / "test-team"
    ids = _make_checkpoints(workspace, count=2)

    # Write a new file to the shared workspace to distinguish from checkpoint.
    (workspace / "shared" / "new_file.txt").write_text("new", encoding="utf-8")

    checkpoint_id = ids[0]
    runner = CliRunner()
    result = runner.invoke(cli, ["rollback", str(p), "--to", checkpoint_id, "--yes"])
    assert result.exit_code == 0
    assert "Rolled back" in result.output or "rolled back" in result.output.lower()


def test_rollback_to_nonexistent_id(tmp_path):
    p = _make_team_yaml(tmp_path)
    workspace = tmp_path / "runs" / "test-team"
    _make_checkpoints(workspace, count=1)

    runner = CliRunner()
    result = runner.invoke(cli, ["rollback", str(p), "--to", "9999_ghost_20250101T000000", "--yes"])
    assert result.exit_code != 0
    assert "not found" in result.output.lower() or "error" in result.output.lower()


def test_rollback_to_with_confirmation_yes(tmp_path):
    p = _make_team_yaml(tmp_path)
    workspace = tmp_path / "runs" / "test-team"
    ids = _make_checkpoints(workspace, count=1)

    runner = CliRunner()
    # Simulate user typing "y" at the confirmation prompt.
    result = runner.invoke(cli, ["rollback", str(p), "--to", ids[0]], input="y\n")
    assert result.exit_code == 0
    assert "Rolled back" in result.output or "rolled back" in result.output.lower()


def test_rollback_to_with_confirmation_no(tmp_path):
    p = _make_team_yaml(tmp_path)
    workspace = tmp_path / "runs" / "test-team"
    ids = _make_checkpoints(workspace, count=1)

    runner = CliRunner()
    result = runner.invoke(cli, ["rollback", str(p), "--to", ids[0]], input="n\n")
    assert result.exit_code == 0
    assert "Cancelled" in result.output


# --------------------------------------------------------------------------- #
# team beliefs
# --------------------------------------------------------------------------- #


def test_beliefs_no_file(tmp_path):
    p = _make_team_yaml(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli, ["beliefs", str(p)])
    assert result.exit_code == 0
    assert "No belief board" in result.output or "belief board found" in result.output.lower()


def test_beliefs_shows_table(tmp_path):
    p = _make_team_yaml(tmp_path)
    workspace = tmp_path / "runs" / "test-team"
    beliefs_path = workspace / "beliefs.json"

    board = BeliefBoard(beliefs_path, member_names=["alice", "bob"], consensus_threshold=0.5)
    board.assert_belief("RNA polymerase is key", author="alice", confidence=0.8)
    board.assert_belief("CRISPR beats TALENs", author="bob", confidence=0.6)

    runner = CliRunner()
    result = runner.invoke(cli, ["beliefs", str(p)])
    assert result.exit_code == 0
    assert "RNA polymerase" in result.output
    assert "CRISPR" in result.output


def test_beliefs_filter_by_status(tmp_path):
    p = _make_team_yaml(tmp_path)
    workspace = tmp_path / "runs" / "test-team"
    beliefs_path = workspace / "beliefs.json"

    # 3 members, threshold 0.5 → need 2/3 (67%) votes to accept, 1/3 stays pending.
    board = BeliefBoard(beliefs_path, member_names=["alice", "bob", "carol"], consensus_threshold=0.5)
    b = board.assert_belief("X is accepted", author="alice")  # 1/3 = 33% < 50% → pending
    board.accept_belief(b.id, voter="bob")  # 2/3 = 67% ≥ 50% → accepted
    board.assert_belief("Y is pending", author="carol")  # 1/3 < 50% → pending

    runner = CliRunner()
    result = runner.invoke(cli, ["beliefs", str(p), "--status", "accepted"])
    assert result.exit_code == 0
    assert "X is accepted" in result.output
    assert "Y is pending" not in result.output


def test_beliefs_empty_after_filter(tmp_path):
    p = _make_team_yaml(tmp_path)
    workspace = tmp_path / "runs" / "test-team"
    beliefs_path = workspace / "beliefs.json"

    # 1-member team: author's vote (1/1 = 100%) always accepts.
    board = BeliefBoard(beliefs_path, member_names=["alice"], consensus_threshold=0.5)
    board.assert_belief("X", author="alice")  # auto-accepted (100%)

    runner = CliRunner()
    result = runner.invoke(cli, ["beliefs", str(p), "--status", "contested"])
    assert result.exit_code == 0
    assert "No beliefs" in result.output or "empty" in result.output.lower()


def test_beliefs_contested_warning(tmp_path):
    p = _make_team_yaml(tmp_path)
    workspace = tmp_path / "runs" / "test-team"
    beliefs_path = workspace / "beliefs.json"

    board = BeliefBoard(beliefs_path, member_names=["alice", "bob"], consensus_threshold=0.5)
    b = board.assert_belief("X", author="alice")
    board.contest_belief(b.id, voter="bob", reason="not enough data")

    runner = CliRunner()
    result = runner.invoke(cli, ["beliefs", str(p)])
    assert result.exit_code == 0
    # Should show a warning about contested beliefs.
    assert "contested" in result.output.lower()
