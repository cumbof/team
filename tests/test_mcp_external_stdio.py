"""External stdio MCP server integration test — real subprocess, no Docker."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

from team.mcp.bus import MCPToolBus

_ECHO = Path(__file__).parent / "fixtures" / "echo_server.py"


@pytest.fixture()
def bus():
    b = MCPToolBus()
    b.start()
    try:
        yield b
    finally:
        b.stop()


def test_stdio_list_and_call(bus):
    bus.add_stdio_server("echo", sys.executable, [str(_ECHO)])
    names = {t.wire_name for t in bus.list_tools()}
    assert names == {"echo__echo", "echo__shout"}
    assert bus.call_tool("echo__echo", {"text": "hi"}) == "hi"
    assert bus.call_tool("echo__shout", {"text": "hi"}) == "HI"


def test_stdio_child_reaped_on_stop():
    """After stop(), the stdio child process is terminated (no zombie)."""
    before = _child_pids()
    b = MCPToolBus()
    b.start()
    b.add_stdio_server("echo", sys.executable, [str(_ECHO)])
    assert b.call_tool("echo__echo", {"text": "x"}) == "x"
    spawned = _child_pids() - before
    assert spawned, "expected a child process to have been spawned"
    b.stop()
    time.sleep(0.5)  # give the OS a beat to reap
    remaining = spawned & _child_pids()
    assert not remaining, f"stdio child(ren) not reaped: {remaining}"


def test_hung_server_marked_dead_not_wedged(monkeypatch):
    """A stdio server that never speaks MCP is bounded, marked dead, and does not
    prevent later servers from connecting or the bus from stopping."""
    import team.mcp.bus as busmod

    monkeypatch.setattr(busmod, "_CONNECT_TIMEOUT", 1.0)
    b = MCPToolBus()
    b.start()
    try:
        # `sleep` launches but never responds to the MCP initialize handshake.
        b.add_stdio_server("hung", "sleep", ["30"])
        # A good server added afterwards must still connect (manager not wedged).
        b.add_stdio_server("echo", sys.executable, [str(_ECHO)])
        assert b.call_tool("echo__echo", {"text": "ok"}) == "ok"
        # The hung server's tools resolve to a clear error, not a hang.
        out = b.call_tool("hung__anything", {}, timeout=2)
        assert out.startswith("ERROR:")
    finally:
        b.stop()  # must return promptly despite the hung child


def _child_pids() -> set[int]:
    """Best-effort set of this process's child PIDs via pgrep."""
    import subprocess

    try:
        out = subprocess.run(
            ["pgrep", "-P", str(os.getpid())],
            capture_output=True, text=True, timeout=5,
        )
        return {int(p) for p in out.stdout.split()}
    except Exception:  # noqa: BLE001
        return set()
