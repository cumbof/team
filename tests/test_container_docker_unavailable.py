"""Tests for ContainerManager graceful degradation when Docker is unavailable.

These tests verify that `team status` and `team down` work correctly even when
the Docker daemon is not running (no DockerException traceback).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from team.cli import cli
from team.config import Defaults, MemberConfig, TeamConfig, WorkflowConfig
from team.container import ContainerManager


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _make_cfg(tmp_path: Path) -> TeamConfig:
    workspace = tmp_path / "runs" / "t"
    workspace.mkdir(parents=True, exist_ok=True)
    return TeamConfig(
        name="t",
        goal="g",
        workspace=workspace,
        workflow=WorkflowConfig(type="round_robin", max_rounds=1),
        defaults=Defaults(),
        members=[
            MemberConfig(name="alice", role="Researcher", model="llama3:8b", persona="x"),
        ],
    )


def _make_team_yaml(tmp_path: Path) -> Path:
    workspace = tmp_path / "runs" / "test-team"
    workspace.mkdir(parents=True, exist_ok=True)
    p = tmp_path / "team.yaml"
    p.write_text(f"""\
name: test-team
goal: test goal
workspace: {workspace}
workflow:
  type: round_robin
  max_rounds: 1
members:
  - name: alice
    role: Researcher
    model: llama3:8b
    persona: You are Alice.
""")
    return p


# --------------------------------------------------------------------------- #
# ContainerManager unit tests
# --------------------------------------------------------------------------- #


def test_container_manager_client_none_when_docker_unavailable(tmp_path):
    """ContainerManager.__init__ sets client=None instead of crashing."""
    from docker.errors import DockerException

    cfg = _make_cfg(tmp_path)
    with patch("team.container.docker.from_env", side_effect=DockerException("no daemon")):
        cm = ContainerManager(cfg)
    assert cm.client is None


def test_status_returns_absent_when_docker_unavailable(tmp_path):
    """ContainerManager.status() returns 'absent' for all members when Docker is down."""
    from docker.errors import DockerException

    cfg = _make_cfg(tmp_path)
    with patch("team.container.docker.from_env", side_effect=DockerException("no daemon")):
        cm = ContainerManager(cfg)

    rows = cm.status()
    assert len(rows) == 1
    assert rows[0]["member"] == "alice"
    assert rows[0]["status"] == "absent"


def test_stop_all_no_op_when_docker_unavailable(tmp_path):
    """ContainerManager.stop_all() silently no-ops when Docker is down."""
    from docker.errors import DockerException

    cfg = _make_cfg(tmp_path)
    with patch("team.container.docker.from_env", side_effect=DockerException("no daemon")):
        cm = ContainerManager(cfg)

    # Should not raise
    cm.stop_all(remove_volumes=True)


def test_require_docker_raises_on_start_when_unavailable(tmp_path):
    """start_member() raises a clear RuntimeError when Docker is down."""
    from docker.errors import DockerException

    cfg = _make_cfg(tmp_path)
    with patch("team.container.docker.from_env", side_effect=DockerException("no daemon")):
        cm = ContainerManager(cfg)

    with pytest.raises(RuntimeError, match="Docker daemon is not running"):
        cm.start_member(cfg.members[0])


# --------------------------------------------------------------------------- #
# CLI integration: team status / team down without Docker
# --------------------------------------------------------------------------- #


def test_cli_status_no_docker(tmp_path):
    """team status exits cleanly and shows 'absent' when Docker is not running."""
    from docker.errors import DockerException

    p = _make_team_yaml(tmp_path)
    runner = CliRunner()
    with patch("team.container.docker.from_env", side_effect=DockerException("no daemon")):
        result = runner.invoke(cli, ["status", str(p)])

    assert result.exit_code == 0
    assert "absent" in result.output


def test_cli_down_no_docker(tmp_path):
    """team down exits cleanly when Docker is not running."""
    from docker.errors import DockerException

    p = _make_team_yaml(tmp_path)
    runner = CliRunner()
    with patch("team.container.docker.from_env", side_effect=DockerException("no daemon")):
        result = runner.invoke(cli, ["down", str(p)])

    assert result.exit_code == 0


# --------------------------------------------------------------------------- #
# Visualize command has been removed
# --------------------------------------------------------------------------- #


def test_visualize_command_removed():
    """The 'team visualize' command no longer exists."""
    runner = CliRunner()
    result = runner.invoke(cli, ["visualize", "--help"])
    assert result.exit_code != 0
    assert "No such command" in result.output
