"""Registry HTTP client — register a team with a central discovery hub and query it.

Usage — registering a running team::

    from team.registry_client import RegistryClient
    from team.registry import RegistryEntry

    client = RegistryClient("http://registry.example.com:8000")

    entry = RegistryEntry(
        name="lab-b",
        url="http://lab-b.example.com:7001",
        description="Computational biology team",
        tags=["biology", "python", "statistics"],
        models=["llama3.1:70b", "qwen2.5-coder:7b"],
    )
    client.register(entry)

    # Keep the registration alive; call every ~60 s or use the built-in
    # heartbeat loop:
    client.start_heartbeat("lab-b", interval=60)

Usage — querying for a team::

    teams = client.query(tags=["biology"], keyword="survival analysis")
    for t in teams:
        print(t.name, t.url, t.tags)
"""

from __future__ import annotations

import json
import logging
import threading
import time

import requests

from team.registry import RegistryEntry, sign_request

log = logging.getLogger(__name__)

_DEFAULT_REQUEST_TIMEOUT = 15  # seconds per HTTP call


class RegistryClientError(RuntimeError):
    """Raised when a registry HTTP call fails."""


class RegistryClient:
    """HTTP client for the team registry protocol.

    Parameters
    ----------
    base_url:
        Root URL of the registry server, e.g.
        ``"http://registry.example.com:8000"``.
    request_timeout:
        Per-HTTP-call timeout in seconds (default: 15).
    secret:
        Optional HMAC shared secret.  Required when the server is started
        with ``--secret``.
    """

    def __init__(
        self,
        base_url: str,
        *,
        request_timeout: int = _DEFAULT_REQUEST_TIMEOUT,
        secret: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._timeout = request_timeout
        self._secret = secret
        self._heartbeat_thread: threading.Thread | None = None
        self._heartbeat_stop = threading.Event()

    # ------------------------------------------------------------------ #
    # Auth helpers
    # ------------------------------------------------------------------ #

    def _auth_headers(self, body: bytes = b"") -> dict[str, str]:
        if not self._secret:
            return {}
        timestamp = str(int(time.time()))
        return {
            "X-Registry-Timestamp": timestamp,
            "X-Registry-Signature": sign_request(self._secret, timestamp, body),
        }

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def health(self) -> dict:
        """Return the server's health payload or raise :class:`RegistryClientError`."""
        return self._get("/health")

    def register(self, entry: RegistryEntry) -> RegistryEntry:
        """Register (or update) *entry* with the server.

        Returns the stored entry as returned by the server (with server-side
        timestamps).  Raises :class:`RegistryClientError` on failure.
        """
        data = self._post("/register", entry.to_dict())
        return RegistryEntry.from_dict(data)

    def deregister(self, name: str) -> None:
        """Deregister the team named *name*.

        Raises :class:`RegistryClientError` if the team is not registered or
        on network errors.
        """
        url = self.base_url + f"/teams/{name}"
        try:
            resp = requests.delete(
                url,
                headers=self._auth_headers(),
                timeout=self._timeout,
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise RegistryClientError(f"DELETE {url} failed: {exc}") from exc

    def heartbeat(self, name: str) -> bool:
        """Send a heartbeat for *name*.

        Returns ``True`` on success, ``False`` if the server says the team is
        no longer registered (caller should re-register).  Raises
        :class:`RegistryClientError` only on hard network / server errors.
        """
        url = self.base_url + f"/heartbeat/{name}"
        body = b""
        headers = {"Content-Length": "0", **self._auth_headers(body)}
        try:
            resp = requests.post(url, data=body, headers=headers, timeout=self._timeout)
            if resp.status_code == 404:
                return False
            resp.raise_for_status()
            return True
        except requests.RequestException as exc:
            raise RegistryClientError(f"POST {url} failed: {exc}") from exc

    def list_all(self) -> list[RegistryEntry]:
        """Return all currently registered teams."""
        data = self._get("/teams")
        return [RegistryEntry.from_dict(d) for d in data.get("teams", [])]

    def query(
        self,
        tags: list[str] | None = None,
        keyword: str | None = None,
    ) -> list[RegistryEntry]:
        """Query the registry for matching teams.

        Parameters
        ----------
        tags:
            Required capability tags; all must be present on the matching
            entry.  Repeat to require multiple tags.
        keyword:
            Substring searched across name, description, and tags.
        """
        params: list[str] = []
        for t in tags or []:
            params.append(f"tag={requests.utils.quote(t)}")
        if keyword:
            params.append(f"keyword={requests.utils.quote(keyword)}")
        path = "/teams" + ("?" + "&".join(params) if params else "")
        data = self._get(path)
        return [RegistryEntry.from_dict(d) for d in data.get("teams", [])]

    def get(self, name: str) -> RegistryEntry | None:
        """Return the entry for *name*, or ``None`` if not registered."""
        try:
            data = self._get(f"/teams/{name}")
            return RegistryEntry.from_dict(data)
        except RegistryClientError as exc:
            if "404" in str(exc) or "not registered" in str(exc):
                return None
            raise

    # ------------------------------------------------------------------ #
    # Background heartbeat
    # ------------------------------------------------------------------ #

    def start_heartbeat(
        self,
        name: str,
        interval: float = 60.0,
        entry: RegistryEntry | None = None,
    ) -> None:
        """Start a background thread that sends heartbeats every *interval* seconds.

        If the server responds with 404 (entry evicted) **and** *entry* is
        provided, the client will automatically re-register before continuing.
        Calling this method a second time while a heartbeat loop is already
        running first stops the existing loop.

        Parameters
        ----------
        name:
            Team name to send heartbeats for.
        interval:
            Seconds between heartbeats (default: 60).
        entry:
            If provided, used to re-register when the server reports the
            entry has expired.
        """
        self.stop_heartbeat()
        self._heartbeat_stop.clear()

        def _loop() -> None:
            while not self._heartbeat_stop.wait(interval):
                try:
                    ok = self.heartbeat(name)
                    if not ok and entry is not None:
                        log.warning(
                            "registry: heartbeat for %r returned 404; re-registering",
                            name,
                        )
                        self.register(entry)
                except RegistryClientError as exc:
                    log.warning("registry: heartbeat failed: %s", exc)

        self._heartbeat_thread = threading.Thread(
            target=_loop, daemon=True, name=f"registry-heartbeat-{name}"
        )
        self._heartbeat_thread.start()
        log.debug("registry: heartbeat loop started for %r (interval=%ss)", name, interval)

    def stop_heartbeat(self) -> None:
        """Stop the background heartbeat thread if one is running."""
        self._heartbeat_stop.set()
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=5)
        self._heartbeat_thread = None

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
            raise RegistryClientError(f"GET {url} failed: {exc}") from exc

    def _post(self, path: str, payload: dict) -> dict:
        url = self.base_url + path
        body = json.dumps(payload, ensure_ascii=False).encode()
        headers = {"Content-Type": "application/json", **self._auth_headers(body)}
        try:
            resp = requests.post(url, data=body, headers=headers, timeout=self._timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            raise RegistryClientError(f"POST {url} failed: {exc}") from exc
