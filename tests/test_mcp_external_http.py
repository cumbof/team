"""External streamable-HTTP MCP server integration test — real subprocess, no Docker.

Mirrors test_mcp_external_stdio.py for the http transport: a hermetic FastMCP
fixture is launched as a subprocess on a free localhost port and driven through
the bus over streamable-HTTP.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from team.mcp.bus import MCPToolBus

_ECHO = Path(__file__).parent / "fixtures" / "http_echo_server.py"
_AUTH_ECHO = Path(__file__).parent / "fixtures" / "http_auth_echo_server.py"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_port(port: int, timeout: float = 15.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        with socket.socket() as s:
            s.settimeout(0.5)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.1)
    return False


@pytest.fixture()
def http_server():
    """Launch the fixture server on a free port; yield its /mcp url."""
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, str(_ECHO), str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        if not _wait_port(port):
            proc.terminate()
            pytest.fail(f"http fixture server never listened on port {port}")
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.fixture()
def bus():
    b = MCPToolBus()
    b.start()
    try:
        yield b
    finally:
        b.stop()


def test_http_list_and_call(bus, http_server):
    bus.add_http_server("echo", http_server)
    names = {t.wire_name for t in bus.list_tools()}
    assert names == {"echo__echo", "echo__shout"}
    assert bus.call_tool("echo__echo", {"text": "hi"}) == "hi"
    assert bus.call_tool("echo__shout", {"text": "hi"}) == "HI"


def test_http_with_headers_connects(bus, http_server):
    """Supplying headers exercises the build-and-manage http-client branch of
    add_http_server (the client is created with the headers and closed on stop);
    the connection must still list and call tools normally."""
    bus.add_http_server("echo", http_server, headers={"X-Team-Test": "1"})
    names = {t.wire_name for t in bus.list_tools()}
    assert names == {"echo__echo", "echo__shout"}
    assert bus.call_tool("echo__echo", {"text": "hdr"}) == "hdr"


def test_http_unreachable_server_marked_dead_not_wedged(bus):
    """An http url with nothing listening is bounded, marked dead, and does not
    prevent a later good server from connecting."""
    import team.mcp.bus as busmod

    orig = busmod._CONNECT_TIMEOUT
    busmod._CONNECT_TIMEOUT = 3.0
    try:
        # Nothing is listening on this port.
        bus.add_http_server("dead", f"http://127.0.0.1:{_free_port()}/mcp")
        out = bus.call_tool("dead__anything", {}, timeout=2)
        assert out.startswith("ERROR:")
    finally:
        busmod._CONNECT_TIMEOUT = orig


def test_http_and_stdio_coexist(bus, http_server):
    """The same bus serves an http server and a stdio server simultaneously."""
    stdio_echo = Path(__file__).parent / "fixtures" / "echo_server.py"
    bus.add_http_server("hecho", http_server)
    bus.add_stdio_server("secho", sys.executable, [str(stdio_echo)])
    servers = {t.server for t in bus.list_tools()}
    assert {"hecho", "secho"} <= servers
    assert bus.call_tool("hecho__shout", {"text": "a"}) == "A"
    assert bus.call_tool("secho__echo", {"text": "b"}) == "b"


# --------------------------------------------------------------------------- #
# Remote-style auth: a bearer-gated server that behaves like a hosted remote.
# --------------------------------------------------------------------------- #

_AUTH_TOKEN = "test-bearer-token-123"


@pytest.fixture()
def auth_server():
    """Launch the bearer-gated fixture; yield (url, token). Fresh per test so a
    rejected connect can't leave a wedged single-worker between cases."""
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, str(_AUTH_ECHO), str(port)],
        env={**os.environ, "AUTH_TOKEN": _AUTH_TOKEN},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        if not _wait_port(port):
            proc.terminate()
            pytest.fail(f"auth fixture server never listened on port {port}")
        yield f"http://127.0.0.1:{port}/mcp", _AUTH_TOKEN
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_http_bearer_auth_transmitted(bus, auth_server):
    """A correct bearer header is actually transmitted to the server: the server
    rejects any request without it, so listing tools and calling echo prove the
    header reached the remote. This is the core 'agents can use remote authed
    MCP servers' guarantee."""
    url, token = auth_server
    bus.add_http_server("remote", url, headers={"Authorization": f"Bearer {token}"})
    assert {t.name for t in bus.list_tools() if t.server == "remote"} == {"echo"}
    assert bus.call_tool("remote__echo", {"text": "authed"}) == "authed"


def test_http_bearer_auth_enforced_missing_header(bus, auth_server):
    """Without the bearer header the server 401s; the bus must mark it dead and
    return an ERROR string (auth is really enforced; failure is graceful)."""
    import team.mcp.bus as busmod

    url, _ = auth_server
    orig = busmod._CONNECT_TIMEOUT
    busmod._CONNECT_TIMEOUT = 4.0
    try:
        bus.add_http_server("remote", url)  # no Authorization header
        out = bus.call_tool("remote__echo", {"text": "x"}, timeout=3)
        assert out.startswith("ERROR:")
    finally:
        busmod._CONNECT_TIMEOUT = orig


def test_http_bearer_auth_enforced_wrong_token(bus, auth_server):
    """A wrong token is rejected exactly like a missing one."""
    import team.mcp.bus as busmod

    url, _ = auth_server
    orig = busmod._CONNECT_TIMEOUT
    busmod._CONNECT_TIMEOUT = 4.0
    try:
        bus.add_http_server("remote", url, headers={"Authorization": "Bearer wrong"})
        out = bus.call_tool("remote__echo", {"text": "x"}, timeout=3)
        assert out.startswith("ERROR:")
    finally:
        busmod._CONNECT_TIMEOUT = orig


def _blackhole(port: int, stop: threading.Event) -> None:
    """Accept TCP connections and hold them open, never speaking HTTP/MCP."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    srv.listen(8)
    srv.settimeout(0.5)
    held = []
    while not stop.is_set():
        try:
            conn, _ = srv.accept()
            held.append(conn)
        except OSError:
            pass
    for conn in held:
        try:
            conn.close()
        except OSError:
            pass
    srv.close()


def test_http_healthy_server_survives_sibling_connect_timeout(bus, http_server):
    """A remote whose TCP accepts but never speaks MCP causes a bounded connect
    timeout; that must NOT wedge the bus for a healthy sibling server. The stdio
    transport has this guarantee (test_mcp_external_stdio); assert it for http."""
    import team.mcp.bus as busmod

    bh_port = _free_port()
    stop = threading.Event()
    t = threading.Thread(target=_blackhole, args=(bh_port, stop), daemon=True)
    t.start()
    _wait_port(bh_port)

    orig = busmod._CONNECT_TIMEOUT
    busmod._CONNECT_TIMEOUT = 4.0
    try:
        bus.add_http_server("good", http_server)
        assert bus.call_tool("good__echo", {"text": "before"}) == "before"

        # Black-hole connect times out.
        bus.add_http_server("blackhole", f"http://127.0.0.1:{bh_port}/mcp")
        assert bus.call_tool("blackhole__echo", {"text": "x"}, timeout=2).startswith("ERROR:")

        # The healthy server must still work afterwards.
        assert bus.call_tool("good__echo", {"text": "after"}) == "after"
    finally:
        busmod._CONNECT_TIMEOUT = orig
        stop.set()
        t.join(timeout=3)
