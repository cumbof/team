"""Cross-team collaboration bridge — protocol layer.

Two ``team`` clusters running on separate machines can collaborate on a
shared goal through the bridge protocol.  One cluster acts as the *client*
(it delegates a sub-task) and the other acts as the *server* (it receives
the task, runs its full team workflow, and returns the results).

Protocol objects
----------------
:class:`BridgeTask`
    Sent by the client when it wants to delegate work.  Carries the goal,
    an optional free-text context blurb, and any workspace files the remote
    team needs to start from.

:class:`BridgeResult`
    Returned by the server.  Contains a free-text summary of what the remote
    team accomplished, the files it produced, and a status field.

:class:`TaskStore`
    Server-side, thread-safe registry of in-flight and completed tasks.
    The HTTP server layer reads and writes through this store.

Wire format
-----------
All objects are serialised as plain JSON so any language/tool can interact
with a bridge server — no shared code required.
"""

from __future__ import annotations

import hashlib
import hmac as _hmac
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Literal


# --------------------------------------------------------------------------- #
# HMAC-SHA256 request signing
# --------------------------------------------------------------------------- #

#: Requests with a timestamp older than this (seconds) are rejected.
TIMESTAMP_TOLERANCE: int = 300


def sign_request(secret: str, timestamp: str, body: bytes) -> str:
    """Return the HMAC-SHA256 hex digest that authenticates a bridge request.

    The signed message is ``"{timestamp}:" + body`` so that both the
    freshness token and the payload are covered by the MAC.

    Parameters
    ----------
    secret:
        Shared secret known to both the client and server.
    timestamp:
        Unix timestamp string (integer seconds).  Must match the
        ``X-Bridge-Timestamp`` header sent with the request.
    body:
        Raw request body bytes.  Pass ``b""`` for GET requests.
    """
    msg = f"{timestamp}:".encode() + body
    return _hmac.new(secret.encode(), msg, hashlib.sha256).hexdigest()


def verify_signature(
    secret: str, timestamp: str, body: bytes, signature: str
) -> bool:
    """Return ``True`` when *signature* is valid and *timestamp* is fresh.

    Uses :func:`hmac.compare_digest` to prevent timing-based side-channel
    attacks.  Rejects requests whose timestamp is older than
    :data:`TIMESTAMP_TOLERANCE` seconds (replay-attack protection).

    Parameters
    ----------
    secret:
        Shared secret configured on the server.
    timestamp:
        Value of the ``X-Bridge-Timestamp`` header.
    body:
        Raw request body bytes (``b""`` for GET requests).
    signature:
        Value of the ``X-Bridge-Signature`` header (hex digest only,
        no prefix).
    """
    try:
        age = abs(time.time() - float(timestamp))
    except ValueError:
        return False
    if age > TIMESTAMP_TOLERANCE:
        return False
    expected = sign_request(secret, timestamp, body)
    return _hmac.compare_digest(expected, signature)


# --------------------------------------------------------------------------- #
# Protocol dataclasses
# --------------------------------------------------------------------------- #

TaskStatus = Literal["pending", "running", "complete", "error", "cancelled"]


@dataclass
class BridgeTask:
    """A unit of work delegated from one team cluster to another.

    Parameters
    ----------
    goal:
        What the remote team should accomplish (free text; becomes the
        ``goal`` for a one-shot Orchestrator run on the server side).
    context:
        Optional background information to include in the remote team's
        kickoff prompt.
    files:
        Workspace files to transfer to the remote team, keyed by their
        relative path.  The server writes them into the remote team's
        shared workspace before running the workflow.
    sender:
        Name of the sending team (informational; logged by the server).
    task_id:
        Stable identifier for this task.  Auto-generated if not provided.
    created_at:
        Unix timestamp.  Auto-set if not provided.
    """

    goal: str
    context: str = ""
    files: dict[str, str] = field(default_factory=dict)
    sender: str = "unknown"
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)

    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BridgeTask":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class BridgeResult:
    """The outcome of a delegated task, returned by the bridge server.

    Parameters
    ----------
    task_id:
        Matches the originating :class:`BridgeTask`.
    status:
        ``"pending"`` — task queued but not started.
        ``"running"`` — team workflow in progress.
        ``"complete"`` — workflow finished; inspect *summary* and *files*.
        ``"error"``    — workflow raised an exception; inspect *error*.
    summary:
        Free-text description of what the remote team accomplished.
    files:
        Files produced by the remote team's shared workspace, keyed by
        relative path.  Written into the local workspace by the client.
    error:
        Set when ``status == "error"``; contains the exception message.
    completed_at:
        Unix timestamp of when the workflow finished (or ``None``).
    """

    task_id: str
    status: TaskStatus = "pending"
    summary: str = ""
    files: dict[str, str] = field(default_factory=dict)
    error: str | None = None
    completed_at: float | None = None

    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BridgeResult":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# --------------------------------------------------------------------------- #
# Server-side task store
# --------------------------------------------------------------------------- #


class TaskStore:
    """Thread-safe registry of tasks and their results.

    The HTTP server layer submits tasks here and workers update them.
    Clients poll through the same store.

    Parameters
    ----------
    ttl_seconds:
        How long to keep completed, errored, or cancelled tasks in memory
        before evicting them.  A background daemon thread runs the eviction
        loop at ``ttl_seconds / 4`` intervals (capped at 5 minutes).
        Pass ``0`` to disable eviction (tasks are kept forever — only
        appropriate for short-lived servers or tests).
    """

    _TERMINAL: frozenset[TaskStatus] = frozenset({"complete", "error", "cancelled"})

    def __init__(self, ttl_seconds: float = 3600.0) -> None:
        self._lock = threading.Lock()
        self._tasks: dict[str, BridgeTask] = {}
        self._results: dict[str, BridgeResult] = {}
        self._ttl = ttl_seconds
        if ttl_seconds > 0:
            self._start_eviction()

    # ------------------------------------------------------------------ #
    # Writing
    # ------------------------------------------------------------------ #

    def add_task(self, task: BridgeTask) -> None:
        """Register a new task (status: pending)."""
        with self._lock:
            self._tasks[task.task_id] = task
            self._results[task.task_id] = BridgeResult(
                task_id=task.task_id, status="pending"
            )

    def mark_running(self, task_id: str) -> None:
        with self._lock:
            if task_id in self._results:
                self._results[task_id].status = "running"

    def mark_complete(
        self,
        task_id: str,
        summary: str,
        files: dict[str, str],
    ) -> None:
        with self._lock:
            if task_id in self._results:
                r = self._results[task_id]
                r.status = "complete"
                r.summary = summary
                r.files = files
                r.completed_at = time.time()

    def mark_error(self, task_id: str, error: str) -> None:
        with self._lock:
            if task_id in self._results:
                r = self._results[task_id]
                r.status = "error"
                r.error = error
                r.completed_at = time.time()

    def mark_cancelled(self, task_id: str) -> bool:
        """Mark a queued or running task as cancelled.

        Returns ``True`` if the task existed and was successfully cancelled.
        Returns ``False`` if the task is not found or is already in a terminal
        state (``complete``, ``error``, or ``cancelled``).
        """
        with self._lock:
            r = self._results.get(task_id)
            if r is None:
                return False
            if r.status in self._TERMINAL:
                return False
            r.status = "cancelled"
            r.completed_at = time.time()
            return True

    # ------------------------------------------------------------------ #
    # Reading
    # ------------------------------------------------------------------ #

    def get_task(self, task_id: str) -> BridgeTask | None:
        with self._lock:
            return self._tasks.get(task_id)

    def get_result(self, task_id: str) -> BridgeResult | None:
        with self._lock:
            return self._results.get(task_id)

    def list_task_ids(self) -> list[str]:
        with self._lock:
            return list(self._tasks.keys())

    def task_summaries(self, status_filter: str | None = None) -> list[dict]:
        """Return lightweight summaries of all known tasks.

        Parameters
        ----------
        status_filter:
            When supplied, only tasks whose current status matches this value
            are returned.  Pass ``None`` to return all tasks.
        """
        with self._lock:
            out: list[dict] = []
            for tid, task in self._tasks.items():
                result = self._results.get(tid)
                status = result.status if result else "unknown"
                if status_filter and status != status_filter:
                    continue
                out.append({
                    "task_id": tid,
                    "status": status,
                    "sender": task.sender,
                    "goal": task.goal[:120],
                    "created_at": task.created_at,
                })
            return out

    # ------------------------------------------------------------------ #
    # TTL eviction
    # ------------------------------------------------------------------ #

    def _start_eviction(self) -> None:
        """Start a daemon thread that purges old terminal tasks."""
        interval = min(self._ttl / 4, 300.0)  # at most every 5 minutes

        def _evict() -> None:
            while True:
                time.sleep(interval)
                cutoff = time.time() - self._ttl
                with self._lock:
                    expired = [
                        tid
                        for tid, r in self._results.items()
                        if r.status in self._TERMINAL
                        and r.completed_at is not None
                        and r.completed_at < cutoff
                    ]
                    for tid in expired:
                        self._tasks.pop(tid, None)
                        self._results.pop(tid, None)

        threading.Thread(target=_evict, daemon=True, name="task-store-eviction").start()
