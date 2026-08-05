"""Tests for the MCP tool bus (Phase 1)."""

from __future__ import annotations

import concurrent.futures
import time

import pytest

from team.mcp.bus import MCPToolBus, MemberToolset, split_wire_name, wire_name


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _make_server(name: str = "demo"):
    """Build an inline FastMCP server exercising the mapping paths."""
    from mcp.server.fastmcp import FastMCP

    srv = FastMCP(name)

    @srv.tool()
    def add(a: int, b: int) -> str:
        """Add two numbers."""
        return str(a + b)

    @srv.tool()
    def echo(text: str) -> str:
        """Echo the given text."""
        return text

    @srv.tool()
    def boom() -> str:
        """Always raises."""
        raise ValueError("kaboom")

    @srv.tool()
    def slow(seconds: float) -> str:
        """Sleep then return."""
        time.sleep(seconds)
        return "done"

    return srv


@pytest.fixture()
def bus():
    b = MCPToolBus()
    b.start()
    try:
        yield b
    finally:
        b.stop()


# --------------------------------------------------------------------------- #
# Naming helpers
# --------------------------------------------------------------------------- #


def test_wire_name_roundtrip():
    assert wire_name("code", "run_python") == "code__run_python"
    assert split_wire_name("code__run_python") == ("code", "run_python")
    # server names never contain underscores, so the first split is unambiguous
    assert split_wire_name("workspace__write_file") == ("workspace", "write_file")
    assert split_wire_name("bare") == ("", "bare")


# --------------------------------------------------------------------------- #
# Discovery + dispatch
# --------------------------------------------------------------------------- #


def test_list_tools(bus):
    bus.add_inprocess_server("demo", _make_server("demo"))
    names = {t.wire_name for t in bus.list_tools()}
    assert names == {"demo__add", "demo__echo", "demo__boom", "demo__slow"}
    info = next(t for t in bus.list_tools() if t.name == "add")
    assert info.server == "demo"
    assert info.description == "Add two numbers."
    assert info.input_schema["required"] == ["a", "b"]


def test_call_ok(bus):
    bus.add_inprocess_server("demo", _make_server())
    assert bus.call_tool("demo__add", {"a": 2, "b": 3}) == "5"


def test_call_error_maps_to_error_prefix(bus):
    bus.add_inprocess_server("demo", _make_server())
    out = bus.call_tool("demo__boom", {})
    assert out.startswith("ERROR:")
    assert "kaboom" in out


def test_unknown_tool_lists_enabled(bus):
    bus.add_inprocess_server("demo", _make_server())
    out = bus.call_tool("demo__nope", {})
    assert out.startswith("ERROR: unknown tool")


def test_timeout(bus):
    bus.add_inprocess_server("demo", _make_server())
    out = bus.call_tool("demo__slow", {"seconds": 2}, timeout=0.5)
    assert out.startswith("ERROR:")
    assert "timed out" in out


# --------------------------------------------------------------------------- #
# Owner scoping
# --------------------------------------------------------------------------- #


def test_owner_scoped_servers(bus):
    # Two members each get their own "demo" instance; a shared "util" server too.
    bus.add_inprocess_server("demo", _make_server("demo"), owner="alice")
    bus.add_inprocess_server("demo", _make_server("demo"), owner="bob")
    bus.add_inprocess_server("util", _make_server("util"))  # shared

    alice = {t.wire_name for t in bus.list_tools(owner="alice")}
    assert "demo__add" in alice and "util__add" in alice
    # Calls route to the member's own server; a member without demo sees shared only.
    assert bus.call_tool("demo__add", {"a": 1, "b": 1}, owner="alice") == "2"
    assert bus.call_tool("util__echo", {"text": "hi"}, owner="bob") == "hi"


# --------------------------------------------------------------------------- #
# Concurrency
# --------------------------------------------------------------------------- #


def test_concurrent_calls(bus):
    bus.add_inprocess_server("demo", _make_server())

    def _one(i: int) -> str:
        return bus.call_tool("demo__add", {"a": i, "b": i})

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(_one, range(8)))
    assert results == [str(2 * i) for i in range(8)]


# --------------------------------------------------------------------------- #
# Lifecycle
# --------------------------------------------------------------------------- #


def test_stop_without_start_is_noop():
    b = MCPToolBus()
    b.stop()  # must not raise


def test_start_is_idempotent():
    b = MCPToolBus()
    b.start()
    b.start()
    try:
        b.add_inprocess_server("demo", _make_server())
        assert b.call_tool("demo__add", {"a": 4, "b": 4}) == "8"
    finally:
        b.stop()
        b.stop()  # double stop is a no-op


# --------------------------------------------------------------------------- #
# MemberToolset
# --------------------------------------------------------------------------- #


def test_member_toolset_enablement(bus):
    bus.add_inprocess_server("demo", _make_server("demo"), owner="alice")
    ts = MemberToolset(bus, "alice", ["demo/add", "demo/echo"])
    names = {t.wire_name for t in ts.tools()}
    assert names == {"demo__add", "demo__echo"}
    assert ts.is_enabled("demo__add")
    assert not ts.is_enabled("demo__boom")
    # disabled tool is refused even though it exists on the server
    out = ts.call("demo__boom", {}, timeout=5)
    assert out.startswith("ERROR:") and "not enabled" in out


def test_member_toolset_wildcard(bus):
    bus.add_inprocess_server("demo", _make_server("demo"), owner="alice")
    ts = MemberToolset(bus, "alice", ["demo/*"])
    assert len(ts.tools()) == 4
    assert ts.call("demo__add", {"a": 5, "b": 6}, timeout=5) == "11"


def test_member_toolset_truncation(bus):
    from mcp.server.fastmcp import FastMCP

    srv = FastMCP("big")

    @srv.tool()
    def huge() -> str:
        """Return a very long string."""
        return "x" * 20000

    bus.add_inprocess_server("big", srv, owner="alice")
    ts = MemberToolset(bus, "alice", ["big/*"])
    out = ts.call("big__huge", {}, timeout=5)
    assert "truncated at 8192" in out
    assert len(out) < 9000
