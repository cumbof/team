"""Tests for the bridge peer registry, list_peers, broadcast_task, and
cancel_remote_task tools, plus BridgeConfig parsing."""

from __future__ import annotations

import textwrap
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from team.bridge import BridgeResult, BridgeTask
from team.bridge_client import BridgeClientError
from team.config import BridgeConfig, TeamConfigError, load_team
from team.tools import execute_tool


# --------------------------------------------------------------------------- #
# BridgeConfig parsing — peers and task_ttl_seconds
# --------------------------------------------------------------------------- #


def _make_team_yaml(bridge_section: str, tmp_path: Path) -> Path:
    """Write a minimal valid team YAML with the given bridge block."""
    content = textwrap.dedent(f"""\
        name: test-team
        goal: do something
        workspace: {tmp_path / 'ws'}
        members:
          - name: agent
            model: tiny
            persona: You are a helpful agent.
            role: worker
        bridge:
{textwrap.indent(bridge_section, '          ')}
    """)
    p = tmp_path / "team.yaml"
    p.write_text(content, encoding="utf-8")
    return p


class TestBridgeConfigParsing:
    def test_default_bridge_config(self, tmp_path):
        p = _make_team_yaml("listen_port: 7000", tmp_path)
        cfg = load_team(p)
        assert cfg.bridge.peers == {}
        assert cfg.bridge.task_ttl_seconds == 3600.0

    def test_peers_parsed(self, tmp_path):
        section = "peers:\n  lab-b: http://lab-b:7001\n  lab-c: http://lab-c:7002\n"
        p = _make_team_yaml(section, tmp_path)
        cfg = load_team(p)
        assert cfg.bridge.peers == {
            "lab-b": "http://lab-b:7001",
            "lab-c": "http://lab-c:7002",
        }

    def test_task_ttl_seconds_parsed(self, tmp_path):
        p = _make_team_yaml("task_ttl_seconds: 1800", tmp_path)
        cfg = load_team(p)
        assert cfg.bridge.task_ttl_seconds == 1800.0

    def test_empty_peers_stays_empty(self, tmp_path):
        p = _make_team_yaml("listen_port: 7000\npeers: {}\n", tmp_path)
        cfg = load_team(p)
        assert cfg.bridge.peers == {}


# --------------------------------------------------------------------------- #
# list_peers tool
# --------------------------------------------------------------------------- #


class TestListPeersTool:
    def test_no_peers_configured(self, tmp_path):
        result = execute_tool("list_peers", "", workspace_path=tmp_path, peers={})
        assert "No peers configured" in result

    def test_no_peers_kwarg_defaults_to_empty(self, tmp_path):
        result = execute_tool("list_peers", "", workspace_path=tmp_path)
        assert "No peers configured" in result

    def test_reachable_peers_shown(self, tmp_path):
        peers = {"lab-b": "http://lab-b:7001", "lab-c": "http://lab-c:7002"}

        with patch("team.bridge_client.BridgeClient") as MockClient:
            instance = MockClient.return_value
            instance.health.return_value = {"status": "ok", "pending": 0, "running": 1}

            result = execute_tool("list_peers", "", workspace_path=tmp_path, peers=peers)

        assert "lab-b" in result
        assert "lab-c" in result
        assert "ok" in result

    def test_unreachable_peer_flagged(self, tmp_path):
        peers = {"dead-lab": "http://dead:7001"}

        with patch("team.bridge_client.BridgeClient") as MockClient:
            instance = MockClient.return_value
            instance.health.side_effect = BridgeClientError("connection refused")

            result = execute_tool("list_peers", "", workspace_path=tmp_path, peers=peers)

        assert "dead-lab" in result
        assert "unreachable" in result

    def test_mixed_reachable_and_unreachable(self, tmp_path):
        peers = {"good": "http://good:7001", "bad": "http://bad:7002"}

        def _health_side_effect(*args, **kwargs):
            return {"status": "ok", "pending": 0, "running": 0}

        with patch("team.bridge_client.BridgeClient") as MockClient:
            def _make_client(url, **kw):
                inst = MagicMock()
                if "bad" in url:
                    inst.health.side_effect = BridgeClientError("refused")
                else:
                    inst.health.return_value = {"status": "ok", "pending": 0, "running": 0}
                return inst

            MockClient.side_effect = _make_client

            result = execute_tool("list_peers", "", workspace_path=tmp_path, peers=peers)

        assert "good" in result
        assert "bad" in result


# --------------------------------------------------------------------------- #
# broadcast_task tool
# --------------------------------------------------------------------------- #


class TestBroadcastTaskTool:
    def test_missing_goal_returns_error(self, tmp_path):
        result = execute_tool(
            "broadcast_task",
            "peers: lab-b, lab-c",
            workspace_path=tmp_path,
        )
        assert result.startswith("ERROR")
        assert "goal" in result

    def test_missing_peers_returns_error(self, tmp_path):
        result = execute_tool(
            "broadcast_task",
            "goal: do something",
            workspace_path=tmp_path,
        )
        assert result.startswith("ERROR")
        assert "peers" in result

    def test_unknown_peer_name_returns_error(self, tmp_path):
        result = execute_tool(
            "broadcast_task",
            "goal: do it\npeers: no-such-peer",
            workspace_path=tmp_path,
            peers={"lab-b": "http://lab-b:7001"},
        )
        assert result.startswith("ERROR")
        assert "no-such-peer" in result

    def test_direct_url_accepted(self, tmp_path):
        mock_result = BridgeResult(task_id="b1", status="complete", summary="done via url")

        with patch("team.bridge_client.BridgeClient") as MockClient:
            instance = MockClient.return_value
            instance.submit_task.return_value = "b1"
            instance.wait_for_result.return_value = mock_result

            result = execute_tool(
                "broadcast_task",
                "goal: analyse data\npeers: http://lab-b:7001",
                workspace_path=tmp_path,
            )

        assert "done via url" in result

    def test_all_peers_receive_same_goal(self, tmp_path):
        peers = {"lab-b": "http://lab-b:7001", "lab-c": "http://lab-c:7002"}
        submitted_goals: list[str] = []

        def _make_client(url, **kw):
            inst = MagicMock()

            def _submit(task):
                submitted_goals.append(task.goal)
                return "tid"

            inst.submit_task.side_effect = _submit
            inst.wait_for_result.return_value = BridgeResult(
                task_id="tid", status="complete", summary=f"done at {url}"
            )
            return inst

        with patch("team.bridge_client.BridgeClient") as MockClient:
            MockClient.side_effect = _make_client

            execute_tool(
                "broadcast_task",
                "goal: the shared goal\npeers: lab-b, lab-c",
                workspace_path=tmp_path,
                peers=peers,
            )

        assert len(submitted_goals) == 2
        assert all(g == "the shared goal" for g in submitted_goals)

    def test_results_namespaced_by_peer(self, tmp_path):
        peers = {"lab-b": "http://lab-b:7001", "lab-c": "http://lab-c:7002"}

        def _make_client(url, **kw):
            inst = MagicMock()
            inst.submit_task.return_value = "tid"
            peer = "lab-b" if "lab-b" in url else "lab-c"
            inst.wait_for_result.return_value = BridgeResult(
                task_id="tid",
                status="complete",
                summary="ok",
                files={"result/out.txt": f"output from {peer}"},
            )
            return inst

        with patch("team.bridge_client.BridgeClient") as MockClient:
            MockClient.side_effect = _make_client

            execute_tool(
                "broadcast_task",
                "goal: produce files\npeers: lab-b, lab-c",
                workspace_path=tmp_path,
                peers=peers,
            )

        written_paths = list(tmp_path.rglob("out.txt"))
        assert len(written_paths) == 2
        parent_names = {p.parent.parent.name for p in written_paths}
        assert "lab-b" in parent_names
        assert "lab-c" in parent_names

    def test_failed_peer_does_not_abort_others(self, tmp_path):
        peers = {"good": "http://good:7001", "bad": "http://bad:7002"}

        def _make_client(url, **kw):
            inst = MagicMock()
            if "bad" in url:
                inst.submit_task.side_effect = BridgeClientError("refused")
            else:
                inst.submit_task.return_value = "tid"
                inst.wait_for_result.return_value = BridgeResult(
                    task_id="tid", status="complete", summary="good done"
                )
            return inst

        with patch("team.bridge_client.BridgeClient") as MockClient:
            MockClient.side_effect = _make_client

            result = execute_tool(
                "broadcast_task",
                "goal: do it\npeers: good, bad",
                workspace_path=tmp_path,
                peers=peers,
            )

        assert "good done" in result
        assert "ERROR" in result  # bad peer failure flagged

    def test_files_sent_to_all_peers(self, tmp_path):
        (tmp_path / "data.csv").write_text("a,b\n1,2", encoding="utf-8")
        peers = {"lab-b": "http://lab-b:7001"}
        received_files: list[dict] = []

        def _make_client(url, **kw):
            inst = MagicMock()

            def _submit(task):
                received_files.append(dict(task.files))
                return "tid"

            inst.submit_task.side_effect = _submit
            inst.wait_for_result.return_value = BridgeResult(
                task_id="tid", status="complete", summary="ok"
            )
            return inst

        with patch("team.bridge_client.BridgeClient") as MockClient:
            MockClient.side_effect = _make_client

            execute_tool(
                "broadcast_task",
                "goal: process\npeers: lab-b\nfiles: data.csv",
                workspace_path=tmp_path,
                peers=peers,
            )

        assert len(received_files) == 1
        assert "data.csv" in received_files[0]


# --------------------------------------------------------------------------- #
# cancel_remote_task tool
# --------------------------------------------------------------------------- #


class TestCancelRemoteTaskTool:
    def test_missing_url_and_peer_returns_error(self, tmp_path):
        result = execute_tool(
            "cancel_remote_task",
            "task_id: abc-123",
            workspace_path=tmp_path,
        )
        assert result.startswith("ERROR")

    def test_missing_task_id_returns_error(self, tmp_path):
        result = execute_tool(
            "cancel_remote_task",
            "url: http://lab-b:7001",
            workspace_path=tmp_path,
        )
        assert result.startswith("ERROR")
        assert "task_id" in result

    def test_successful_cancel_via_url(self, tmp_path):
        with patch("team.bridge_client.BridgeClient") as MockClient:
            instance = MockClient.return_value
            instance.cancel_task.return_value = True

            result = execute_tool(
                "cancel_remote_task",
                "url: http://lab-b:7001\ntask_id: abc-123",
                workspace_path=tmp_path,
            )

        assert "Cancelled" in result
        assert "abc-123" in result

    def test_successful_cancel_via_peer_name(self, tmp_path):
        peers = {"lab-b": "http://lab-b:7001"}

        with patch("team.bridge_client.BridgeClient") as MockClient:
            instance = MockClient.return_value
            instance.cancel_task.return_value = True

            result = execute_tool(
                "cancel_remote_task",
                "peer: lab-b\ntask_id: xyz-456",
                workspace_path=tmp_path,
                peers=peers,
            )

        assert "Cancelled" in result
        MockClient.assert_called_once_with("http://lab-b:7001", secret=None)

    def test_unknown_peer_returns_error(self, tmp_path):
        result = execute_tool(
            "cancel_remote_task",
            "peer: ghost\ntask_id: abc",
            workspace_path=tmp_path,
            peers={"lab-b": "http://lab-b:7001"},
        )
        assert result.startswith("ERROR")
        assert "ghost" in result

    def test_network_error_returns_error_string(self, tmp_path):
        with patch("team.bridge_client.BridgeClient") as MockClient:
            instance = MockClient.return_value
            instance.cancel_task.side_effect = BridgeClientError("connection refused")

            result = execute_tool(
                "cancel_remote_task",
                "url: http://lab-b:7001\ntask_id: abc-123",
                workspace_path=tmp_path,
            )

        assert result.startswith("ERROR")
        assert "connection refused" in result
