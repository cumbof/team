"""Bridge HTTP client — submit tasks to a remote team cluster and collect results.

Usage::

    from team.bridge_client import BridgeClient
    from team.bridge import BridgeTask

    client = BridgeClient("http://lab-b.example.com:7001")

    task = BridgeTask(
        goal="Run the survival analysis on the preprocessed BRCA data.",
        context="Preprocessing complete; data.csv contains 1 142 samples.",
        files={"data/preprocessed.csv": "<csv content>"},
        sender="lab-a",
    )

    task_id = client.submit_task(task)
    result = client.wait_for_result(task_id, timeout=600)

    if result.status == "complete":
        for path, content in result.files.items():
            print(f"received: {path} ({len(content)} chars)")
    else:
        print(f"task failed: {result.error}")

All network calls use ``requests`` (already a project dependency) and raise
:class:`BridgeClientError` on HTTP or connection failures.
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING

import requests

from team.bridge import BridgeResult, BridgeTask, sign_request

if TYPE_CHECKING:
    from team.belief_federation import BeliefBundle

log = logging.getLogger(__name__)

_DEFAULT_POLL_INTERVAL = 5.0   # seconds between status polls
_DEFAULT_TIMEOUT = 600         # seconds to wait for a result


class BridgeClientError(RuntimeError):
    """Raised when a bridge HTTP call fails."""


class BridgeClient:
    """HTTP client for the cross-team bridge protocol.

    Parameters
    ----------
    base_url:
        Root URL of the remote bridge server, e.g.
        ``"http://lab-b.example.com:7001"``.  Trailing slashes are stripped.
    request_timeout:
        Per-HTTP-call timeout in seconds (default: 30).
    """

    def __init__(self, base_url: str, *, request_timeout: int = 30, secret: str | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self._timeout = request_timeout
        self._secret = secret

    # ------------------------------------------------------------------ #
    # Auth helpers
    # ------------------------------------------------------------------ #

    def _auth_headers(self, body: bytes = b"") -> dict[str, str]:
        """Return HMAC auth headers, or an empty dict when no secret is set."""
        if not self._secret:
            return {}
        timestamp = str(int(time.time()))
        return {
            "X-Bridge-Timestamp": timestamp,
            "X-Bridge-Signature": sign_request(self._secret, timestamp, body),
        }

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def health(self) -> dict:
        """Return the server's health payload or raise :class:`BridgeClientError`."""
        return self._get("/health")

    def get_capabilities(self) -> dict:
        """Return the server's capability advertisement.

        The response contains ``name``, ``models``, ``personas``, ``skills``,
        and ``version`` fields (whatever the server is configured to return).
        Raises :class:`BridgeClientError` on network or HTTP errors.
        """
        return self._get("/capabilities")

    def list_tasks(self, status: str | None = None) -> list[dict]:
        """Return summary dicts for all tasks known to the server.

        Parameters
        ----------
        status:
            Optional filter.  Pass ``"pending"``, ``"running"``,
            ``"complete"``, ``"error"``, or ``"cancelled"`` to restrict
            results to tasks in that state.  ``None`` returns all tasks.
        """
        path = "/tasks" if not status else f"/tasks?status={status}"
        data = self._get(path)
        return data.get("tasks", [])

    def cancel_task(self, task_id: str) -> bool:
        """Ask the server to cancel *task_id*.

        Returns ``True`` if the server accepted the cancellation.
        Raises :class:`BridgeClientError` on 404 (not found) or 409
        (already in terminal state) as well as network errors.
        """
        url = self.base_url + f"/tasks/{task_id}"
        try:
            resp = requests.delete(
                url, headers=self._auth_headers(), timeout=self._timeout
            )
            resp.raise_for_status()
            return True
        except requests.RequestException as exc:
            raise BridgeClientError(f"DELETE {url} failed: {exc}") from exc

    def submit_task(self, task: BridgeTask) -> str:
        """Submit a task and return the assigned ``task_id``.

        Raises :class:`BridgeClientError` on any HTTP or network error.
        """
        resp = self._post("/tasks", task.to_dict())
        task_id = resp.get("task_id")
        if not task_id:
            raise BridgeClientError(f"server returned no task_id: {resp}")
        log.info("bridge client: submitted task %s to %s", task_id, self.base_url)
        return task_id

    def poll_result(self, task_id: str) -> BridgeResult:
        """Fetch the current result/status for *task_id* (single call, no waiting).

        Raises :class:`BridgeClientError` if the task is not found or the
        server returns an error.
        """
        data = self._get(f"/tasks/{task_id}")
        return BridgeResult.from_dict(data)

    def wait_for_result(
        self,
        task_id: str,
        timeout: float = _DEFAULT_TIMEOUT,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
    ) -> BridgeResult:
        """Poll until the task reaches a terminal state and return the result.

        Terminal states are ``"complete"`` and ``"error"``.

        Parameters
        ----------
        task_id:
            The task to wait for.
        timeout:
            Maximum total seconds to wait before raising :class:`BridgeClientError`.
        poll_interval:
            Seconds to sleep between polls (default: 5).

        Raises :class:`BridgeClientError` if the timeout expires or if a
        network error occurs during polling.
        """
        deadline = time.monotonic() + timeout
        while True:
            result = self.poll_result(task_id)
            log.debug(
                "bridge client: task %s status=%s", task_id, result.status
            )
            if result.status in ("complete", "error"):
                return result
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise BridgeClientError(
                    f"timed out waiting for task {task_id!r} after {timeout}s "
                    f"(last status: {result.status!r})"
                )
            time.sleep(min(poll_interval, max(0, remaining)))

    # ------------------------------------------------------------------ #
    # Belief federation
    # ------------------------------------------------------------------ #

    def pull_beliefs(
        self,
        status: str | None = None,
        limit: int = 50,
    ) -> "BeliefBundle | None":
        """Fetch the remote team's belief board as a :class:`~team.belief_federation.BeliefBundle`.

        Returns ``None`` when the remote server does not expose belief
        endpoints (``404``).  Raises :class:`BridgeClientError` on other
        network or HTTP errors.

        Parameters
        ----------
        status:
            Optional filter, e.g. ``"accepted"`` — only retrieve beliefs with
            that status.  ``None`` returns all beliefs.
        limit:
            Maximum number of beliefs to request (server-side cap).
        """
        params: list[str] = []
        if status:
            params.append(f"status={status}")
        if limit != 50:
            params.append(f"limit={limit}")
        path = "/beliefs" + ("?" + "&".join(params) if params else "")
        try:
            data = self._get(path)
        except BridgeClientError as exc:
            if "404" in str(exc):
                return None
            raise
        from team.belief_federation import BeliefBundle
        return BeliefBundle.from_dict(data)

    def push_beliefs(self, bundle: "BeliefBundle") -> tuple[int, int] | None:
        """Push a :class:`~team.belief_federation.BeliefBundle` to the remote team.

        Returns ``(imported, skipped)`` — counts from the remote team's import
        operation.  Returns ``None`` when the remote server does not support
        belief import (``404``).  Raises :class:`BridgeClientError` on other
        errors.

        Parameters
        ----------
        bundle:
            The belief bundle to push.  Build one with
            :func:`~team.belief_federation.export_beliefs`.
        """
        try:
            resp = self._post("/beliefs/import", bundle.to_dict())
        except BridgeClientError as exc:
            if "404" in str(exc):
                return None
            raise
        return resp.get("imported", 0), resp.get("skipped", 0)

    # ------------------------------------------------------------------ #
    # Low-level HTTP helpers
    # ------------------------------------------------------------------ #

    def _get(self, path: str) -> dict:
        url = self.base_url + path
        try:
            resp = requests.get(url, headers=self._auth_headers(), timeout=self._timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            raise BridgeClientError(f"GET {url} failed: {exc}") from exc

    def _post(self, path: str, payload: dict) -> dict:
        url = self.base_url + path
        body = json.dumps(payload, ensure_ascii=False).encode()
        headers = {"Content-Type": "application/json", **self._auth_headers(body)}
        try:
            resp = requests.post(url, data=body, headers=headers, timeout=self._timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            raise BridgeClientError(f"POST {url} failed: {exc}") from exc
