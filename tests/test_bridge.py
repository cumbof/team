"""Tests for the cross-team collaboration bridge.

Covers:
- BridgeTask / BridgeResult serialisation (round-trip).
- TaskStore state transitions (add → running → complete / error / cancelled).
- TaskStore.task_summaries() and TTL eviction.
- BridgeServer + BridgeClient full integration over a real (loopback) HTTP
  connection using an in-process stub runner — no Docker required.
- New endpoints: GET /tasks, GET /capabilities, DELETE /tasks/<id>.
- delegate_task built-in tool with a mocked BridgeClient.
- Error / timeout paths.
"""

from __future__ import annotations

import time
from pathlib import Path
import socket
from unittest.mock import MagicMock, patch

import pytest

from team.bridge import BridgeResult, BridgeTask, TaskStore
from team.bridge_client import BridgeClient, BridgeClientError
from team.bridge_server import BridgeServer
from team.tools import execute_tool


def _tcp_loopback_available() -> bool:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.listen(1)
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.settimeout(1)
        try:
            client.connect(("127.0.0.1", port))
            return True
        except OSError:
            return False
        finally:
            client.close()
    except OSError:
        return False
    finally:
        probe.close()


_requires_loopback_tcp = pytest.mark.skipif(
    not _tcp_loopback_available(),
    reason="TCP loopback connections are not available in this environment",
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _stub_runner(
    sub_workspace: Path,
    goal: str,
    context: str,
    input_files: dict[str, str],
) -> tuple[str, dict[str, str]]:
    """Synchronous stub: echoes the goal and any input files straight back."""
    produced = {k: v + " [processed]" for k, v in input_files.items()}
    produced["result/summary.txt"] = f"Accomplished: {goal}"
    return f"stub completed: {goal}", produced


# --------------------------------------------------------------------------- #
# BridgeTask serialisation
# --------------------------------------------------------------------------- #


class TestBridgeTask:
    def test_round_trip(self):
        t = BridgeTask(
            goal="analyse data",
            context="10 samples",
            files={"data.csv": "a,b\n1,2"},
            sender="lab-a",
        )
        d = t.to_dict()
        t2 = BridgeTask.from_dict(d)
        assert t2.goal == t.goal
        assert t2.context == t.context
        assert t2.files == t.files
        assert t2.sender == t.sender
        assert t2.task_id == t.task_id

    def test_auto_task_id(self):
        t1 = BridgeTask(goal="a")
        t2 = BridgeTask(goal="b")
        assert t1.task_id != t2.task_id

    def test_from_dict_ignores_extra_keys(self):
        d = BridgeTask(goal="x").to_dict()
        d["unknown_future_field"] = "ignored"
        # Should not raise even with an extra key
        t = BridgeTask.from_dict(d)
        assert t.goal == "x"


# --------------------------------------------------------------------------- #
# BridgeResult serialisation
# --------------------------------------------------------------------------- #


class TestBridgeResult:
    def test_round_trip(self):
        r = BridgeResult(
            task_id="abc-123",
            status="complete",
            summary="done",
            files={"out.txt": "hello"},
            completed_at=time.time(),
        )
        d = r.to_dict()
        r2 = BridgeResult.from_dict(d)
        assert r2.task_id == r.task_id
        assert r2.status == "complete"
        assert r2.files == r.files

    def test_default_status_is_pending(self):
        r = BridgeResult(task_id="x")
        assert r.status == "pending"
        assert r.error is None
        assert r.completed_at is None


# --------------------------------------------------------------------------- #
# TaskStore state machine
# --------------------------------------------------------------------------- #


class TestTaskStore:
    def test_add_task_creates_pending_result(self):
        store = TaskStore()
        task = BridgeTask(goal="test")
        store.add_task(task)
        result = store.get_result(task.task_id)
        assert result is not None
        assert result.status == "pending"

    def test_mark_running(self):
        store = TaskStore()
        task = BridgeTask(goal="test")
        store.add_task(task)
        store.mark_running(task.task_id)
        assert store.get_result(task.task_id).status == "running"

    def test_mark_complete(self):
        store = TaskStore()
        task = BridgeTask(goal="test")
        store.add_task(task)
        store.mark_complete(task.task_id, "done", {"out.txt": "data"})
        r = store.get_result(task.task_id)
        assert r.status == "complete"
        assert r.summary == "done"
        assert r.files == {"out.txt": "data"}
        assert r.completed_at is not None

    def test_mark_error(self):
        store = TaskStore()
        task = BridgeTask(goal="test")
        store.add_task(task)
        store.mark_error(task.task_id, "boom")
        r = store.get_result(task.task_id)
        assert r.status == "error"
        assert r.error == "boom"

    def test_get_nonexistent_task(self):
        store = TaskStore()
        assert store.get_task("nonexistent") is None
        assert store.get_result("nonexistent") is None

    def test_list_task_ids(self):
        store = TaskStore(ttl_seconds=0)
        t1 = BridgeTask(goal="a")
        t2 = BridgeTask(goal="b")
        store.add_task(t1)
        store.add_task(t2)
        ids = store.list_task_ids()
        assert t1.task_id in ids
        assert t2.task_id in ids

    def test_mark_cancelled_pending(self):
        store = TaskStore(ttl_seconds=0)
        task = BridgeTask(goal="cancel me")
        store.add_task(task)
        ok = store.mark_cancelled(task.task_id)
        assert ok is True
        r = store.get_result(task.task_id)
        assert r.status == "cancelled"
        assert r.completed_at is not None

    def test_mark_cancelled_running(self):
        store = TaskStore(ttl_seconds=0)
        task = BridgeTask(goal="cancel while running")
        store.add_task(task)
        store.mark_running(task.task_id)
        ok = store.mark_cancelled(task.task_id)
        assert ok is True
        assert store.get_result(task.task_id).status == "cancelled"

    def test_mark_cancelled_already_complete(self):
        store = TaskStore(ttl_seconds=0)
        task = BridgeTask(goal="x")
        store.add_task(task)
        store.mark_complete(task.task_id, "done", {})
        ok = store.mark_cancelled(task.task_id)
        assert ok is False
        assert store.get_result(task.task_id).status == "complete"

    def test_mark_cancelled_nonexistent(self):
        store = TaskStore(ttl_seconds=0)
        assert store.mark_cancelled("no-such-id") is False

    def test_task_summaries_all(self):
        store = TaskStore(ttl_seconds=0)
        t1 = BridgeTask(goal="task one", sender="lab-a")
        t2 = BridgeTask(goal="task two", sender="lab-b")
        store.add_task(t1)
        store.add_task(t2)
        store.mark_complete(t2.task_id, "done", {})
        summaries = store.task_summaries()
        assert len(summaries) == 2
        ids = {s["task_id"] for s in summaries}
        assert t1.task_id in ids
        assert t2.task_id in ids

    def test_task_summaries_filtered(self):
        store = TaskStore(ttl_seconds=0)
        t1 = BridgeTask(goal="pending task")
        t2 = BridgeTask(goal="complete task")
        store.add_task(t1)
        store.add_task(t2)
        store.mark_complete(t2.task_id, "done", {})
        pending = store.task_summaries(status_filter="pending")
        assert len(pending) == 1
        assert pending[0]["task_id"] == t1.task_id

    def test_task_summaries_goal_truncated(self):
        store = TaskStore(ttl_seconds=0)
        long_goal = "x" * 200
        task = BridgeTask(goal=long_goal)
        store.add_task(task)
        s = store.task_summaries()[0]
        assert len(s["goal"]) <= 120

    def test_ttl_eviction(self):
        store = TaskStore(ttl_seconds=0.05)  # 50 ms TTL
        task = BridgeTask(goal="evict me")
        store.add_task(task)
        store.mark_complete(task.task_id, "done", {})
        # The eviction thread runs at TTL/4 intervals; wait long enough.
        time.sleep(0.3)
        assert store.get_result(task.task_id) is None

    def test_ttl_zero_disables_eviction(self):
        store = TaskStore(ttl_seconds=0)
        task = BridgeTask(goal="keep forever")
        store.add_task(task)
        store.mark_complete(task.task_id, "done", {})
        time.sleep(0.05)
        # No eviction — task must still be present.
        assert store.get_result(task.task_id) is not None


# --------------------------------------------------------------------------- #
# BridgeServer + BridgeClient integration (real HTTP, loopback, stub runner)
# --------------------------------------------------------------------------- #


@pytest.fixture
def bridge(tmp_path):
    """Start a BridgeServer with the stub runner on an OS-assigned free port."""
    server = BridgeServer(
        runner=_stub_runner,
        port=0,  # let the OS pick a free port; server.port has the actual value
        workspace_root=tmp_path / "ws",
        task_ttl_seconds=0,  # disable eviction so tests can inspect results freely
        capabilities={"name": "test-team", "models": ["tiny"], "personas": ["agent"], "skills": []},
    )
    server.start()
    yield server, server.port
    server.stop()


@_requires_loopback_tcp
class TestBridgeIntegration:
    def test_health_endpoint(self, bridge):
        server, port = bridge
        client = BridgeClient(f"http://127.0.0.1:{port}")
        h = client.health()
        assert h["status"] == "ok"

    def test_submit_and_wait(self, bridge):
        server, port = bridge
        client = BridgeClient(f"http://127.0.0.1:{port}")
        task = BridgeTask(
            goal="run analysis",
            files={"input.csv": "x,y\n1,2"},
            sender="test-lab",
        )
        task_id = client.submit_task(task)
        assert task_id  # non-empty string

        result = client.wait_for_result(task_id, timeout=10, poll_interval=0.1)
        assert result.status == "complete"
        assert "run analysis" in result.summary
        assert "result/summary.txt" in result.files
        # Input file should be returned processed
        assert "input.csv" in result.files
        assert "[processed]" in result.files["input.csv"]

    def test_submit_no_input_files(self, bridge):
        server, port = bridge
        client = BridgeClient(f"http://127.0.0.1:{port}")
        task = BridgeTask(goal="simple goal")
        task_id = client.submit_task(task)
        result = client.wait_for_result(task_id, timeout=10, poll_interval=0.1)
        assert result.status == "complete"
        assert "result/summary.txt" in result.files

    def test_poll_result_unknown_task(self, bridge):
        server, port = bridge
        client = BridgeClient(f"http://127.0.0.1:{port}")
        with pytest.raises(BridgeClientError, match="404"):
            client.poll_result("nonexistent-task-id")

    def test_wait_timeout(self, bridge):
        server, port = bridge
        client = BridgeClient(f"http://127.0.0.1:{port}")

        # Use a runner that sleeps longer than the timeout.
        def _slow_runner(ws, goal, ctx, files):
            time.sleep(10)
            return "done", {}

        # Inject a slow task directly into the store so it stays "running"
        task = BridgeTask(goal="slow")
        server.store.add_task(task)
        server.store.mark_running(task.task_id)  # frozen in running state

        with pytest.raises(BridgeClientError, match="timed out"):
            client.wait_for_result(task.task_id, timeout=0.3, poll_interval=0.1)

    def test_error_runner(self, tmp_path):
        """A runner that raises should mark the task as error."""
        def _failing_runner(ws, goal, ctx, files):
            raise RuntimeError("deliberate failure")

        server = BridgeServer(
            runner=_failing_runner,
            port=0,
            workspace_root=tmp_path / "ws",
        )
        server.start()
        try:
            client = BridgeClient(f"http://127.0.0.1:{server.port}")
            task = BridgeTask(goal="will fail")
            task_id = client.submit_task(task)
            result = client.wait_for_result(task_id, timeout=10, poll_interval=0.1)
            assert result.status == "error"
            assert "deliberate failure" in (result.error or "")
        finally:
            server.stop()


# --------------------------------------------------------------------------- #
# BridgeClient error handling
# --------------------------------------------------------------------------- #


class TestBridgeClientErrors:
    def test_unreachable_server_raises(self):
        client = BridgeClient("http://127.0.0.1:19999")
        with pytest.raises(BridgeClientError):
            client.health()

    def test_submit_unreachable_raises(self):
        client = BridgeClient("http://127.0.0.1:19999")
        with pytest.raises(BridgeClientError):
            client.submit_task(BridgeTask(goal="x"))


# --------------------------------------------------------------------------- #
# delegate_task built-in tool
# --------------------------------------------------------------------------- #


class TestDelegateTaskTool:
    def test_missing_url_returns_error(self, tmp_path):
        result = execute_tool("delegate_task", "goal: do something", workspace_path=tmp_path)
        assert result.startswith("ERROR")

    def test_missing_goal_returns_error(self, tmp_path):
        result = execute_tool("delegate_task", "url: http://example.com:7000", workspace_path=tmp_path)
        assert result.startswith("ERROR")

    def test_successful_delegation(self, tmp_path):
        mock_result = BridgeResult(
            task_id="t1",
            status="complete",
            summary="Lab B finished the analysis successfully.",
            files={"output/report.md": "# Report\nAll done."},
        )
        with patch("team.bridge_client.BridgeClient") as MockClient:
            instance = MockClient.return_value
            instance.submit_task.return_value = "t1"
            instance.wait_for_result.return_value = mock_result

            body = (
                "url: http://lab-b:7001\n"
                "goal: Run the regression analysis\n"
                "context: data is ready\n"
            )
            result = execute_tool("delegate_task", body, workspace_path=tmp_path)

        assert "Lab B finished" in result
        assert (tmp_path / "output" / "report.md").is_file()
        assert (tmp_path / "output" / "report.md").read_text() == "# Report\nAll done."

    def test_remote_error_returned_as_error_string(self, tmp_path):
        mock_result = BridgeResult(
            task_id="t2",
            status="error",
            error="remote crash",
        )
        with patch("team.bridge_client.BridgeClient") as MockClient:
            instance = MockClient.return_value
            instance.submit_task.return_value = "t2"
            instance.wait_for_result.return_value = mock_result

            result = execute_tool(
                "delegate_task",
                "url: http://lab-b:7001\ngoal: something",
                workspace_path=tmp_path,
            )
        assert "ERROR" in result
        assert "remote crash" in result

    def test_network_error_returned_as_error_string(self, tmp_path):
        with patch("team.bridge_client.BridgeClient") as MockClient:
            instance = MockClient.return_value
            instance.submit_task.side_effect = BridgeClientError("connection refused")

            result = execute_tool(
                "delegate_task",
                "url: http://lab-b:7001\ngoal: something",
                workspace_path=tmp_path,
            )
        assert "ERROR" in result
        assert "connection refused" in result

    def test_files_are_read_and_sent(self, tmp_path):
        """Local files listed in 'files:' should be read and embedded in the task."""
        (tmp_path / "data.csv").write_text("a,b\n1,2", encoding="utf-8")

        submitted_tasks: list[BridgeTask] = []
        mock_result = BridgeResult(task_id="t3", status="complete", summary="ok")

        with patch("team.bridge_client.BridgeClient") as MockClient:
            def _submit(task):
                submitted_tasks.append(task)
                return "t3"

            instance = MockClient.return_value
            instance.submit_task.side_effect = _submit
            instance.wait_for_result.return_value = mock_result

            execute_tool(
                "delegate_task",
                "url: http://lab-b:7001\ngoal: process\nfiles: data.csv",
                workspace_path=tmp_path,
            )

        assert len(submitted_tasks) == 1
        assert "data.csv" in submitted_tasks[0].files
        assert "a,b" in submitted_tasks[0].files["data.csv"]

    def test_path_traversal_in_files_is_skipped(self, tmp_path):
        """Files outside the workspace should be silently skipped."""
        mock_result = BridgeResult(task_id="t4", status="complete", summary="ok")
        submitted_tasks: list[BridgeTask] = []

        with patch("team.bridge_client.BridgeClient") as MockClient:
            def _submit(task):
                submitted_tasks.append(task)
                return "t4"

            instance = MockClient.return_value
            instance.submit_task.side_effect = _submit
            instance.wait_for_result.return_value = mock_result

            execute_tool(
                "delegate_task",
                "url: http://lab-b:7001\ngoal: x\nfiles: ../../etc/passwd",
                workspace_path=tmp_path,
            )

        # The traversal path should not appear in submitted files
        assert "../../etc/passwd" not in submitted_tasks[0].files


# --------------------------------------------------------------------------- #
# New HTTP endpoints: GET /tasks, GET /capabilities, DELETE /tasks/<id>
# --------------------------------------------------------------------------- #


@_requires_loopback_tcp
class TestBridgeNewEndpoints:
    def test_list_tasks_empty(self, bridge):
        server, port = bridge
        client = BridgeClient(f"http://127.0.0.1:{port}")
        tasks = client.list_tasks()
        assert isinstance(tasks, list)
        assert tasks == []

    def test_list_tasks_after_submit(self, bridge):
        server, port = bridge
        client = BridgeClient(f"http://127.0.0.1:{port}")
        task_id = client.submit_task(BridgeTask(goal="listed goal"))
        client.wait_for_result(task_id, timeout=10, poll_interval=0.1)
        tasks = client.list_tasks()
        assert any(t["task_id"] == task_id for t in tasks)

    def test_list_tasks_status_filter(self, bridge):
        server, port = bridge
        client = BridgeClient(f"http://127.0.0.1:{port}")
        task_id = client.submit_task(BridgeTask(goal="for filter test"))
        client.wait_for_result(task_id, timeout=10, poll_interval=0.1)
        complete = client.list_tasks(status="complete")
        assert any(t["task_id"] == task_id for t in complete)
        pending = client.list_tasks(status="pending")
        assert not any(t["task_id"] == task_id for t in pending)

    def test_list_tasks_summary_fields(self, bridge):
        server, port = bridge
        client = BridgeClient(f"http://127.0.0.1:{port}")
        task_id = client.submit_task(BridgeTask(goal="fields check", sender="lab-x"))
        client.wait_for_result(task_id, timeout=10, poll_interval=0.1)
        tasks = client.list_tasks()
        entry = next(t for t in tasks if t["task_id"] == task_id)
        assert "status" in entry
        assert "goal" in entry
        assert "sender" in entry
        assert "created_at" in entry

    def test_get_capabilities(self, bridge):
        server, port = bridge
        client = BridgeClient(f"http://127.0.0.1:{port}")
        caps = client.get_capabilities()
        assert caps["name"] == "test-team"
        assert "models" in caps
        assert "personas" in caps
        assert "skills" in caps
        assert "version" in caps

    def test_cancel_pending_task(self, bridge):
        server, port = bridge
        client = BridgeClient(f"http://127.0.0.1:{port}")
        # Inject a task directly into the store so it stays pending (semaphore occupied).
        from team.bridge import BridgeTask as _BT
        task = _BT(goal="to be cancelled")
        server.store.add_task(task)
        ok = client.cancel_task(task.task_id)
        assert ok is True
        result = client.poll_result(task.task_id)
        assert result.status == "cancelled"

    def test_cancel_nonexistent_task_raises(self, bridge):
        server, port = bridge
        client = BridgeClient(f"http://127.0.0.1:{port}")
        with pytest.raises(BridgeClientError, match="404"):
            client.cancel_task("no-such-task")

    def test_cancel_completed_task_raises(self, bridge):
        server, port = bridge
        client = BridgeClient(f"http://127.0.0.1:{port}")
        task_id = client.submit_task(BridgeTask(goal="complete first"))
        client.wait_for_result(task_id, timeout=10, poll_interval=0.1)
        with pytest.raises(BridgeClientError, match="409"):
            client.cancel_task(task_id)

    def test_cancelled_task_not_executed(self, tmp_path):
        """A task cancelled before a semaphore slot frees should never run."""
        executed: list[str] = []

        def _slow_runner(ws, goal, ctx, files):
            executed.append(goal)
            return "done", {}

        # max_concurrent_tasks=1: second task must wait for the slot.
        server = BridgeServer(
            runner=_slow_runner,
            port=0,
            workspace_root=tmp_path / "ws",
            max_concurrent_tasks=1,
            task_ttl_seconds=0,
        )
        server.start()
        try:
            # Occupy the only slot with a task that will be cancelled first.
            blocker = BridgeTask(goal="blocker")
            server.store.add_task(blocker)
            server.store.mark_running(blocker.task_id)

            # Register a second task and immediately cancel it.
            waiter = BridgeTask(goal="should not run")
            server.store.add_task(waiter)
            server.store.mark_cancelled(waiter.task_id)

            # Simulate what _schedule would do (but semaphore is blocked).
            # Check that _run_task bails if the task is already cancelled.
            server._run_task(waiter)  # noqa: SLF001
            assert waiter.goal not in executed
        finally:
            server.stop()


# --------------------------------------------------------------------------- #
# delegate_task with peer name resolution
# --------------------------------------------------------------------------- #


class TestDelegateTaskPeerResolution:
    def test_peer_name_resolved_from_registry(self, tmp_path):
        mock_result = BridgeResult(task_id="p1", status="complete", summary="ok via peer")
        peers = {"lab-b": "http://lab-b.internal:7001"}

        with patch("team.bridge_client.BridgeClient") as MockClient:
            instance = MockClient.return_value
            instance.submit_task.return_value = "p1"
            instance.wait_for_result.return_value = mock_result

            result = execute_tool(
                "delegate_task",
                "peer: lab-b\ngoal: do the thing",
                workspace_path=tmp_path,
                peers=peers,
            )

        assert "ok via peer" in result
        MockClient.assert_called_once_with("http://lab-b.internal:7001", secret=None)

    def test_unknown_peer_returns_error(self, tmp_path):
        result = execute_tool(
            "delegate_task",
            "peer: unknown-lab\ngoal: do something",
            workspace_path=tmp_path,
            peers={"lab-b": "http://lab-b:7001"},
        )
        assert result.startswith("ERROR")
        assert "unknown-lab" in result

    def test_missing_both_url_and_peer_returns_error(self, tmp_path):
        result = execute_tool("delegate_task", "goal: do something", workspace_path=tmp_path)
        assert result.startswith("ERROR")

