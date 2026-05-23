"""Registry HTTP server — central service-discovery hub for team clusters.

Start a standalone registry with the CLI::

    team registry start --port 8000

Or start one programmatically::

    from team.registry_server import RegistryServer

    server = RegistryServer(port=8000, secret="s3cr3t")
    server.start()   # background thread
    server.join()    # block until stopped

HTTP API
--------
``POST /register``
    Body: JSON-serialised :class:`~team.registry.RegistryEntry`.
    Creates or replaces the entry for ``entry.name``.
    Response 200: the stored entry (with server-assigned timestamps).

``DELETE /teams/<name>``
    Deregister the team with the given name.
    Response 200: ``{"name": str, "status": "deregistered"}``
    Response 404: name not found.

``POST /heartbeat/<name>``
    Refresh the TTL for the named team.  Call this periodically (e.g.
    every 60 s) to prevent eviction.
    Response 200: ``{"name": str, "status": "ok"}``
    Response 404: name not known (re-register).

``GET /teams``
    List all live entries.
    Optional query parameters:

    * ``?tag=<tag>`` — filter by a single tag (repeat for multiple; all
      must be present).
    * ``?keyword=<word>`` — substring search across name, description, tags.

    Response 200: ``{"teams": [<RegistryEntry>, …]}``

``GET /teams/<name>``
    Return the entry for a specific team.
    Response 200: the :class:`~team.registry.RegistryEntry` dict.
    Response 404: not found.

``GET /health``
    Response 200: ``{"status": "ok", "registered": N}``

Authentication
--------------
When the server is started with a *secret*, every write request
(``POST /register``, ``DELETE /teams/<name>``, ``POST /heartbeat/<name>``)
must include ``X-Registry-Timestamp`` and ``X-Registry-Signature`` headers
(same HMAC-SHA256 scheme as the bridge server).  Read requests
(``GET /teams``, ``GET /teams/<name>``, ``GET /health``) are public.
"""

from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

from team.registry import RegistryEntry, TeamRegistry, verify_signature

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# HTTP request handler
# --------------------------------------------------------------------------- #


class _Handler(BaseHTTPRequestHandler):
    """Minimal HTTP handler — routes to the RegistryServer instance."""

    server: "RegistryServer"  # type: ignore[assignment]

    def log_message(self, fmt: str, *args: object) -> None:  # noqa: D102
        log.debug("registry http: " + fmt, *args)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _send_json(self, code: int, data: dict) -> None:
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length) if length else b""

    def _parse_json(self, body: bytes) -> dict | None:
        if not body:
            return None
        try:
            return json.loads(body)
        except (json.JSONDecodeError, ValueError):
            return None

    def _check_auth(self, body: bytes) -> bool:
        secret = self.server.registry_server._secret  # noqa: SLF001
        if not secret:
            return True
        timestamp = self.headers.get("X-Registry-Timestamp", "")
        signature = self.headers.get("X-Registry-Signature", "")
        if not timestamp or not signature:
            return False
        return verify_signature(secret, timestamp, body, signature)

    # ------------------------------------------------------------------ #
    # Routing
    # ------------------------------------------------------------------ #

    def do_GET(self) -> None:  # noqa: N802
        # Read requests are always public.
        if self.path == "/health":
            self._handle_health()
        elif self.path == "/teams" or self.path.startswith("/teams?"):
            self._handle_list_teams()
        elif self.path.startswith("/teams/"):
            name = self.path[len("/teams/"):]
            self._handle_get_team(name)
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        body = self._read_body()
        if not self._check_auth(body):
            self._send_json(401, {"error": "unauthorized"})
            return
        if self.path == "/register":
            self._handle_register(body)
        elif self.path.startswith("/heartbeat/"):
            name = self.path[len("/heartbeat/"):]
            self._handle_heartbeat(name)
        else:
            self._send_json(404, {"error": "not found"})

    def do_DELETE(self) -> None:  # noqa: N802
        if not self._check_auth(b""):
            self._send_json(401, {"error": "unauthorized"})
            return
        if self.path.startswith("/teams/"):
            name = self.path[len("/teams/"):]
            self._handle_deregister(name)
        else:
            self._send_json(404, {"error": "not found"})

    # ------------------------------------------------------------------ #
    # Handlers
    # ------------------------------------------------------------------ #

    def _handle_health(self) -> None:
        count = self.server.registry_server.store.count()
        self._send_json(200, {"status": "ok", "registered": count})

    def _handle_list_teams(self) -> None:
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        tags: list[str] = qs.get("tag", [])
        keywords: list[str] = qs.get("keyword", [])
        keyword = keywords[0] if keywords else None
        entries = self.server.registry_server.store.query(
            tags=tags or None, keyword=keyword
        )
        self._send_json(200, {"teams": [e.to_dict() for e in entries]})

    def _handle_get_team(self, name: str) -> None:
        entry = self.server.registry_server.store.get(name)
        if entry is None:
            self._send_json(404, {"error": f"team {name!r} not registered"})
        else:
            self._send_json(200, entry.to_dict())

    def _handle_register(self, body: bytes) -> None:
        data = self._parse_json(body)
        if not data or "name" not in data or "url" not in data:
            self._send_json(400, {"error": "body must include 'name' and 'url'"})
            return
        try:
            entry = RegistryEntry.from_dict(data)
        except (TypeError, KeyError) as exc:
            self._send_json(400, {"error": f"invalid entry: {exc}"})
            return
        stored = self.server.registry_server.store.register(entry)
        log.info("registry: registered team %r at %s", stored.name, stored.url)
        self._send_json(200, stored.to_dict())

    def _handle_heartbeat(self, name: str) -> None:
        ok = self.server.registry_server.store.heartbeat(name)
        if ok:
            self._send_json(200, {"name": name, "status": "ok"})
            log.debug("registry: heartbeat from %r", name)
        else:
            self._send_json(
                404,
                {"name": name, "status": "not_registered",
                 "message": "team not found; please re-register"},
            )

    def _handle_deregister(self, name: str) -> None:
        ok = self.server.registry_server.store.deregister(name)
        if ok:
            log.info("registry: deregistered team %r", name)
            self._send_json(200, {"name": name, "status": "deregistered"})
        else:
            self._send_json(404, {"error": f"team {name!r} not registered"})


# --------------------------------------------------------------------------- #
# Thin HTTPServer subclass that holds server-level state
# --------------------------------------------------------------------------- #


class _RegistryHTTPServer(HTTPServer):
    """Extends HTTPServer to carry a reference to the outer RegistryServer."""

    def __init__(self, *args: object, registry_server: "RegistryServer", **kwargs: object) -> None:
        self.registry_server = registry_server  # type: ignore[assignment]
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Public RegistryServer class
# --------------------------------------------------------------------------- #


class RegistryServer:
    """Standalone HTTP registry server.

    Parameters
    ----------
    port:
        TCP port to listen on (default: 8000).
    host:
        Bind address (default: ``"0.0.0.0"``).
    secret:
        Optional HMAC shared secret.  When set, write requests must carry
        valid ``X-Registry-Timestamp`` / ``X-Registry-Signature`` headers.
    entry_ttl_seconds:
        Seconds after which an entry that has not sent a heartbeat is
        automatically evicted.  Pass ``0`` to disable eviction.
    """

    def __init__(
        self,
        port: int = 8000,
        host: str = "0.0.0.0",
        secret: str | None = None,
        entry_ttl_seconds: float = 300.0,
    ) -> None:
        self.port = port
        self.host = host
        self._secret = secret
        self.store = TeamRegistry(entry_ttl_seconds=entry_ttl_seconds)
        self._http: _RegistryHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the HTTP server in a background daemon thread."""
        self._http = _RegistryHTTPServer(
            (self.host, self.port),
            _Handler,
            registry_server=self,
        )
        self._thread = threading.Thread(
            target=self._http.serve_forever,
            daemon=True,
            name="registry-http",
        )
        self._thread.start()
        log.info("registry server started on %s:%d", self.host, self.port)

    def stop(self) -> None:
        """Shut down the HTTP server gracefully."""
        if self._http:
            self._http.shutdown()
            self._http = None
        log.info("registry server stopped")

    def join(self) -> None:
        """Block until :meth:`stop` is called."""
        if self._thread:
            self._thread.join()
