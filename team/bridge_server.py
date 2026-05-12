"""Bridge HTTP server — lets a team accept delegated tasks from remote clusters.

Start with::

    from team.bridge_server import BridgeServer
    from team.config import load_team

    cfg = load_team("my-team.yaml")
    server = BridgeServer(cfg, port=7001)
    server.start()   # background thread
    server.join()    # block until stopped (e.g. Ctrl-C)

Or use the CLI::

    team serve my-team.yaml --port 7001

HTTP API
--------
``POST /tasks``
    Body: JSON-serialised :class:`~team.bridge.BridgeTask`.
    Response 202: ``{"task_id": "<id>"}``

``GET /tasks``
    Response 200: ``{"tasks": [{task_id, status, sender, goal, created_at}, …]}``
    Optional query parameter ``?status=pending|running|complete|error|cancelled``
    to filter by task status.

``GET /tasks/<task_id>``
    Response 200: JSON-serialised :class:`~team.bridge.BridgeResult`.
    ``status`` is one of ``"pending"``, ``"running"``, ``"complete"``,
    ``"error"``, ``"cancelled"``.

``DELETE /tasks/<task_id>``
    Cancel a queued or running task.
    Response 200: ``{"task_id": "<id>", "status": "cancelled"}``
    Response 404: task not found.
    Response 409: task already in terminal state.

``GET /capabilities``
    Response 200: ``{"name": str, "models": [...], "personas": [...],
    "skills": [...], "version": str}``

``GET /health``
    Response 200: ``{"status": "ok", "pending": N, "running": N}``

Isolation
---------
Each task gets its own sub-workspace under
``<team.workspace>/bridge/<task_id>/`` so concurrent tasks (when
``max_concurrent_tasks > 1``) never collide.  The task files are
written there before the workflow starts; the produced artifacts are
read back from ``<sub-workspace>/shared/`` and returned in the result.
"""

from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, urlparse

from team.bridge import BridgeResult, BridgeTask, TaskStore, verify_signature

log = logging.getLogger(__name__)

# Type alias: a factory that accepts a sub-workspace path and the delegated
# goal string, then runs the team workflow and returns (summary, files_dict).
# Keeping this as a callable makes the server unit-testable without Docker.
WorkflowRunner = Callable[[Path, str, str, dict[str, str]], tuple[str, dict[str, str]]]


# --------------------------------------------------------------------------- #
# Default workflow runner — uses the real Orchestrator
# --------------------------------------------------------------------------- #

def _make_default_runner(cfg_path: Path) -> WorkflowRunner:
    """Return a WorkflowRunner that drives a real team Orchestrator.

    A fresh :class:`~team.config.TeamConfig` is loaded from *cfg_path* for
    every task so that concurrent tasks get independent configurations (each
    with its own sub-workspace).

    The remote team's goal is replaced with the delegated task's goal so
    members see the correct objective.  Any files sent by the client are
    written into ``<sub_workspace>/shared/`` before the workflow starts.
    """

    def _run(
        sub_workspace: Path,
        goal: str,
        context: str,
        input_files: dict[str, str],
    ) -> tuple[str, dict[str, str]]:
        from team.config import load_team
        from team.container import ContainerManager
        from team.member import Member
        from team.orchestrator import Orchestrator
        from team.workspace import SharedWorkspace

        # Load a fresh config and override workspace + goal.
        cfg = load_team(cfg_path)
        cfg.workspace = sub_workspace.resolve()
        cfg.goal = goal if not context else f"{goal}\n\nContext from requesting team:\n{context}"

        # Write any files the client sent.
        ws = SharedWorkspace(cfg.workspace)
        for rel, content in input_files.items():
            ws.write(rel, content)

        orch = Orchestrator(cfg)
        orch.up()
        try:
            orch.run()
        finally:
            orch.down()

        # Collect all files from the shared workspace.
        produced: dict[str, str] = {}
        for rel_path in ws.list_files():
            try:
                produced[rel_path] = (ws.shared / rel_path).read_text(
                    encoding="utf-8", errors="replace"
                )
            except OSError:
                pass

        # Build a summary from the final non-orchestrator transcript turn.
        summary_turns = [
            t for t in orch.transcript.turns
            if t.speaker not in ("orchestrator", "human")
        ]
        summary = summary_turns[-1].content[:1000] if summary_turns else "(no output)"
        return summary, produced

    return _run


def _capabilities_from_cfg(cfg_path: Path) -> dict:
    """Build a capabilities dict from the team config at *cfg_path*.

    Returns a best-effort snapshot; failures produce an empty dict rather
    than crashing the server startup.
    """
    try:
        from team.config import load_team
        cfg = load_team(cfg_path)
        skills: list[str] = []
        for m in cfg.members:
            for s in m.skills or []:
                if s not in skills:
                    skills.append(s)
        return {
            "name": cfg.name,
            "models": list({m.model for m in cfg.members}),
            "personas": [m.name for m in cfg.members],
            "skills": skills,
        }
    except Exception:  # noqa: BLE001
        return {}


# --------------------------------------------------------------------------- #
# HTTP request handler
# --------------------------------------------------------------------------- #

class _Handler(BaseHTTPRequestHandler):
    """Minimal HTTP handler — routes requests to the BridgeServer instance."""

    # The server attribute is typed as HTTPServer but we always use BridgeServer.
    server: "BridgeServer"  # type: ignore[assignment]

    def log_message(self, fmt: str, *args: object) -> None:
        log.debug("bridge http: " + fmt, *args)

    def _send_json(self, code: int, data: dict) -> None:
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", 0))
        if not length:
            return b""
        return self.rfile.read(length)

    def _parse_json(self, body: bytes) -> dict | None:
        if not body:
            return None
        try:
            return json.loads(body)
        except (json.JSONDecodeError, ValueError):
            return None

    def _check_auth(self, body: bytes) -> bool:
        """Return True if the request is authenticated (or auth is disabled)."""
        secret = self.server.bridge_server._secret  # noqa: SLF001
        if not secret:
            return True
        timestamp = self.headers.get("X-Bridge-Timestamp", "")
        signature = self.headers.get("X-Bridge-Signature", "")
        if not timestamp or not signature:
            return False
        return verify_signature(secret, timestamp, body, signature)

    def do_GET(self) -> None:  # noqa: N802
        if not self._check_auth(b""):
            self._send_json(401, {"error": "unauthorized"})
            return
        if self.path == "/health":
            self._handle_health()
        elif self.path == "/capabilities":
            self._handle_capabilities()
        elif self.path == "/tasks" or self.path.startswith("/tasks?"):
            self._handle_list_tasks()
        elif self.path.startswith("/tasks/"):
            task_id = self.path[len("/tasks/"):]
            self._handle_get_task(task_id)
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        body = self._read_body()
        if not self._check_auth(body):
            self._send_json(401, {"error": "unauthorized"})
            return
        if self.path == "/tasks":
            self._handle_post_task(body)
        else:
            self._send_json(404, {"error": "not found"})

    def do_DELETE(self) -> None:  # noqa: N802
        if not self._check_auth(b""):
            self._send_json(401, {"error": "unauthorized"})
            return
        if self.path.startswith("/tasks/"):
            task_id = self.path[len("/tasks/"):]
            self._handle_cancel_task(task_id)
        else:
            self._send_json(404, {"error": "not found"})

    def _handle_health(self) -> None:
        store = self.server.store
        ids = store.list_task_ids()
        pending = sum(
            1 for tid in ids
            if (r := store.get_result(tid)) and r.status == "pending"
        )
        running = sum(
            1 for tid in ids
            if (r := store.get_result(tid)) and r.status == "running"
        )
        self._send_json(200, {"status": "ok", "pending": pending, "running": running})

    def _handle_capabilities(self) -> None:
        from team._version import __version__
        caps = dict(self.server.bridge_server._capabilities)  # noqa: SLF001
        caps["version"] = __version__
        self._send_json(200, caps)

    def _handle_list_tasks(self) -> None:
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        status_filter = (qs.get("status") or [None])[0]
        summaries = self.server.store.task_summaries(status_filter)
        self._send_json(200, {"tasks": summaries})

    def _handle_get_task(self, task_id: str) -> None:
        result = self.server.store.get_result(task_id)
        if result is None:
            self._send_json(404, {"error": f"task {task_id!r} not found"})
            return
        self._send_json(200, result.to_dict())

    def _handle_cancel_task(self, task_id: str) -> None:
        ok = self.server.store.mark_cancelled(task_id)
        if ok:
            self._send_json(200, {"task_id": task_id, "status": "cancelled"})
            log.info("bridge: task %s cancelled via HTTP", task_id)
            return
        result = self.server.store.get_result(task_id)
        if result is None:
            self._send_json(404, {"error": f"task {task_id!r} not found"})
        else:
            self._send_json(
                409,
                {"error": f"task is already in terminal state: {result.status!r}"},
            )

    def _handle_post_task(self, body: bytes) -> None:
        data = self._parse_json(body)
        if not data:
            self._send_json(400, {"error": "invalid or empty JSON body"})
            return
        try:
            task = BridgeTask.from_dict(data)
        except (TypeError, KeyError) as exc:
            self._send_json(400, {"error": f"malformed task: {exc}"})
            return

        self.server.store.add_task(task)
        self.server.bridge_server._schedule(task)  # noqa: SLF001
        log.info("bridge: accepted task %s from %s", task.task_id, task.sender)
        self._send_json(202, {"task_id": task.task_id})


# --------------------------------------------------------------------------- #
# BridgeServer
# --------------------------------------------------------------------------- #


class BridgeServer:
    """HTTP bridge server that accepts delegated tasks and runs them.

    Parameters
    ----------
    runner:
        Callable that receives ``(sub_workspace, goal, context, input_files)``
        and returns ``(summary, produced_files)``.  Pass a custom callable
        to unit-test without Docker; leave as ``None`` to use the default
        runner built from *cfg_path*.
    cfg_path:
        Path to the team YAML file.  Required when *runner* is ``None``.
    port:
        TCP port to listen on (default: 7000).
    max_concurrent_tasks:
        Maximum number of tasks running simultaneously.  Tasks beyond this
        limit are queued and started as slots free up.
    workspace_root:
        Root directory for per-task sub-workspaces.  Defaults to a
        ``bridge/`` sub-directory next to the team YAML file.
    """

    def __init__(
        self,
        runner: WorkflowRunner | None = None,
        *,
        cfg_path: Path | None = None,
        port: int = 7000,
        max_concurrent_tasks: int = 1,
        workspace_root: Path | None = None,
        secret: str | None = None,
        capabilities: dict | None = None,
        task_ttl_seconds: float = 3600.0,
    ) -> None:
        if runner is None:
            if cfg_path is None:
                raise ValueError("provide either runner or cfg_path")
            runner = _make_default_runner(cfg_path)
            if workspace_root is None:
                workspace_root = cfg_path.parent / "bridge_workspaces"
            if capabilities is None:
                capabilities = _capabilities_from_cfg(cfg_path)

        self._runner = runner
        self._port = port
        self._secret = secret
        self._capabilities: dict = capabilities or {}
        self._semaphore = threading.Semaphore(max_concurrent_tasks)
        self._workspace_root: Path = Path(workspace_root or "./bridge_workspaces").resolve()

        self.store = TaskStore(ttl_seconds=task_ttl_seconds)
        self._http: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """Start the HTTP server in a background daemon thread.

        Blocks until the server loop is confirmed ready to accept connections
        (up to 5 seconds) before returning.
        """
        self._http = HTTPServer(("", self._port), _Handler)
        self._port = self._http.server_address[1]  # update with actual bound port
        self._http.store = self.store           # type: ignore[attr-defined]
        self._http.bridge_server = self         # type: ignore[attr-defined]

        ready = threading.Event()

        def _serve() -> None:
            ready.set()
            self._http.serve_forever()  # type: ignore[union-attr]

        self._thread = threading.Thread(
            target=_serve,
            daemon=True,
            name="bridge-http",
        )
        self._thread.start()
        ready.wait(timeout=5)
        log.info("bridge server listening on port %d", self._port)

    def stop(self) -> None:
        if self._http:
            self._http.shutdown()
            self._http = None
        log.info("bridge server stopped")

    def join(self) -> None:
        """Block until the server thread exits (call after stop())."""
        if self._thread:
            self._thread.join()

    @property
    def port(self) -> int:
        return self._port

    # ------------------------------------------------------------------ #
    # Task scheduling
    # ------------------------------------------------------------------ #

    def _schedule(self, task: BridgeTask) -> None:
        """Dispatch a task to a worker thread (respecting the semaphore)."""
        t = threading.Thread(
            target=self._run_task,
            args=(task,),
            daemon=True,
            name=f"bridge-task-{task.task_id[:8]}",
        )
        t.start()

    def _run_task(self, task: BridgeTask) -> None:
        """Worker: acquire a slot, run the workflow, release the slot."""
        self._semaphore.acquire()
        try:
            # The task may have been cancelled while waiting for a semaphore slot.
            current = self.store.get_result(task.task_id)
            if current and current.status == "cancelled":
                log.info("bridge: task %s was cancelled before execution", task.task_id)
                return

            self.store.mark_running(task.task_id)
            sub_ws = self._workspace_root / task.task_id
            sub_ws.mkdir(parents=True, exist_ok=True)
            log.info("bridge: running task %s (goal: %.60s…)", task.task_id, task.goal)
            try:
                summary, files = self._runner(
                    sub_ws, task.goal, task.context, task.files
                )
                self.store.mark_complete(task.task_id, summary, files)
                log.info("bridge: task %s complete (%d file(s))", task.task_id, len(files))
            except Exception as exc:  # noqa: BLE001
                log.exception("bridge: task %s failed: %s", task.task_id, exc)
                self.store.mark_error(task.task_id, str(exc))
        finally:
            self._semaphore.release()
