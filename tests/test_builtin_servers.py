"""Tests for the 8 built-in MCP servers (Phase 2), driven through the bus."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from team.beliefs import BeliefBoard
from team.memory import AgentMemory
from team.mcp.bus import MCPToolBus
from team.mcp.builtin import BUILTIN_SERVERS, BuiltinContext
from team.mcp.builtin.code import (
    VALID_TOOL_SANDBOXES,
    build_sandboxed_cmd,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture()
def ctx(tmp_path):
    mem = AgentMemory(tmp_path / "mem.db")
    beliefs = BeliefBoard(tmp_path / "beliefs.json", ["m1", "m2"])
    return BuiltinContext(
        workspace_path=tmp_path,
        member_name="m1",
        memory=mem,
        beliefs=beliefs,
        peers={"lab-b": "http://lab-b:7001"},
        registry_url="http://registry:8000",
        tool_timeout=5,
    )


@pytest.fixture()
def bus(ctx):
    b = MCPToolBus()
    b.start()
    for name in ("code", "web", "workspace", "memory", "beliefs", "decisions", "federation", "expert"):
        b.add_inprocess_server(name, BUILTIN_SERVERS[name](ctx), owner="m1")
    try:
        yield b
    finally:
        b.stop()


def _call(bus, wire, **args):
    return bus.call_tool(wire, args, owner="m1", timeout=10)


# --------------------------------------------------------------------------- #
# code
# --------------------------------------------------------------------------- #


def test_run_python_simple(bus):
    assert _call(bus, "code__run_python", code="print('hello world')").strip() == "hello world"


def test_run_python_stderr(bus):
    out = _call(bus, "code__run_python", code="import sys; sys.stderr.write('oops')")
    assert "STDERR:" in out and "oops" in out


def test_run_python_syntax_error(bus):
    assert "SyntaxError" in _call(bus, "code__run_python", code="def (:")


def test_run_bash_simple(bus):
    assert _call(bus, "code__run_bash", command="echo greetings").strip() == "greetings"


def test_run_bash_exit_code(bus):
    # non-zero exit still returns (no stdout) rather than raising
    out = _call(bus, "code__run_bash", command="exit 42")
    assert "no output" in out or out.strip() == ""


# --------------------------------------------------------------------------- #
# workspace
# --------------------------------------------------------------------------- #


def test_write_then_read(bus, tmp_path):
    assert "wrote" in _call(bus, "workspace__write_file", path="notes/a.txt", content="hi")
    assert (tmp_path / "notes/a.txt").read_text() == "hi"
    assert _call(bus, "workspace__read_file", path="notes/a.txt") == "hi"


def test_append_file(bus, tmp_path):
    _call(bus, "workspace__write_file", path="a.txt", content="x")
    _call(bus, "workspace__append_file", path="a.txt", content="y")
    assert (tmp_path / "a.txt").read_text() == "xy"


def test_read_missing(bus):
    assert "not found" in _call(bus, "workspace__read_file", path="nope.txt")


def test_path_traversal_blocked(bus):
    assert "escapes the workspace" in _call(bus, "workspace__read_file", path="../../etc/passwd")


def test_list_files(bus):
    _call(bus, "workspace__write_file", path="x.py", content="1")
    _call(bus, "workspace__write_file", path="y.md", content="2")
    out = _call(bus, "workspace__list_files", pattern="*.py")
    assert "x.py" in out and "y.md" not in out


# --------------------------------------------------------------------------- #
# memory
# --------------------------------------------------------------------------- #


def test_memory_roundtrip(bus):
    _call(bus, "memory__remember", key="k1", value="the sky is blue", tags="facts")
    out = _call(bus, "memory__recall", query="sky")
    assert "k1" in out and "sky is blue" in out
    assert "Deleted" in _call(bus, "memory__forget", key="k1")


def test_memory_list(bus):
    _call(bus, "memory__remember", key="k2", value="v2")
    assert "k2" in _call(bus, "memory__list_memories")


def test_memory_disabled(tmp_path):
    c = BuiltinContext(workspace_path=tmp_path, member_name="m1", memory=None)
    b = MCPToolBus()
    b.start()
    try:
        b.add_inprocess_server("memory", BUILTIN_SERVERS["memory"](c), owner="m1")
        out = b.call_tool("memory__recall", {"query": "x"}, owner="m1")
        assert "not enabled" in out
    finally:
        b.stop()


# --------------------------------------------------------------------------- #
# beliefs
# --------------------------------------------------------------------------- #


def test_belief_assert_and_list(bus):
    out = _call(bus, "beliefs__assert_belief", claim="X is true", confidence=0.8)
    assert "asserted" in out
    listing = _call(bus, "beliefs__list_beliefs")
    assert "X is true" in listing


def test_belief_bad_status(bus):
    assert "unknown status" in _call(bus, "beliefs__list_beliefs", status="bogus")


# --------------------------------------------------------------------------- #
# decisions
# --------------------------------------------------------------------------- #


def test_decisions(bus, tmp_path):
    assert "logged" in _call(bus, "decisions__log_decision", title="Use pandas", rationale="mature")
    out = _call(bus, "decisions__read_decisions")
    assert "Use pandas" in out and "@m1" in out


# --------------------------------------------------------------------------- #
# web (mocked)
# --------------------------------------------------------------------------- #


def test_web_search_mocked(bus):
    fake = MagicMock()
    fake.json.return_value = {"Heading": "Python", "AbstractText": "A language."}
    fake.raise_for_status.return_value = None
    with patch("team.mcp.builtin.web.requests.get", return_value=fake):
        out = _call(bus, "web__web_search", query="python")
    assert "Python" in out and "A language." in out


def test_read_url_network_error(bus):
    with patch("team.mcp.builtin.web.requests.get", side_effect=Exception("boom")):
        out = _call(bus, "web__read_url", url="http://example.com")
    assert out.startswith("ERROR")


# --------------------------------------------------------------------------- #
# federation + expert (error paths, no network)
# --------------------------------------------------------------------------- #


def test_list_peers_configured(bus):
    # The configured peer is unreachable in tests; list_peers reports it as such
    # rather than raising.
    out = _call(bus, "federation__list_peers")
    assert "lab-b" in out


def test_delegate_requires_target(bus):
    assert "provide url" in _call(bus, "federation__delegate_task", goal="do it")


def test_expert_unknown_provider(bus):
    assert "unknown provider" in _call(bus, "expert__delegate_to_expert", provider="bogus", prompt="hi")


def test_expert_missing_key(bus, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    out = _call(bus, "expert__delegate_to_expert", provider="openai", prompt="hi")
    assert "OPENAI_API_KEY" in out


# --------------------------------------------------------------------------- #
# sandbox command building (moved from tools.py to builtin/code.py)
# --------------------------------------------------------------------------- #


def test_valid_sandboxes_set():
    assert VALID_TOOL_SANDBOXES == {"none", "firejail", "bubblewrap"}


def test_none_sandbox_unchanged(tmp_path):
    cmd = ["python", "x.py"]
    assert build_sandboxed_cmd(cmd, tmp_path, "none") == cmd


def test_firejail_prepends(tmp_path):
    with patch("team.mcp.builtin.code._sandbox_available", return_value=True):
        result = build_sandboxed_cmd(["python", "x.py"], tmp_path, "firejail")
    assert result[0] == "firejail"
    assert f"--whitelist={tmp_path}" in result
    assert result[-2:] == ["python", "x.py"]


def test_firejail_fallback_when_missing(tmp_path, caplog):
    with patch("team.mcp.builtin.code._sandbox_available", return_value=False):
        result = build_sandboxed_cmd(["python", "x.py"], tmp_path, "firejail")
    assert result == ["python", "x.py"]


def test_bubblewrap_prepends(tmp_path):
    with patch("team.mcp.builtin.code._sandbox_available", return_value=True):
        result = build_sandboxed_cmd(["python", "x.py"], tmp_path, "bubblewrap")
    assert result[0] == "bwrap"
    assert "--bind" in result


def test_unknown_sandbox_falls_back(tmp_path):
    result = build_sandboxed_cmd(["python", "x.py"], tmp_path, "nonexistent")
    assert result == ["python", "x.py"]


# --------------------------------------------------------------------------- #
# Schema regression — pin the 25 generated schemas
# --------------------------------------------------------------------------- #

#: wire_name -> (sorted required list, {prop: json-type}).  Pinned so that an
#: accidental signature change to a built-in tool is caught in review.
EXPECTED_SCHEMAS = {
    "code__run_python": (["code"], {"code": "string"}),
    "code__run_bash": (["command"], {"command": "string"}),
    "web__web_search": (["query"], {"query": "string"}),
    "web__read_url": (["url"], {"url": "string"}),
    "workspace__read_file": (["path"], {"path": "string"}),
    "workspace__write_file": (["content", "path"], {"path": "string", "content": "string"}),
    "workspace__append_file": (["content", "path"], {"path": "string", "content": "string"}),
    "workspace__list_files": ([], {"pattern": "string"}),
    "memory__remember": (
        ["key", "value"],
        {"key": "string", "value": "string", "tags": "string", "importance": "number"},
    ),
    "memory__recall": (["query"], {"query": "string", "limit": "integer"}),
    "memory__forget": (["key"], {"key": "string"}),
    "memory__list_memories": ([], {"tag": "string", "limit": "integer"}),
    "beliefs__assert_belief": (
        ["claim"],
        {"claim": "string", "confidence": "number", "evidence": "string"},
    ),
    "beliefs__contest_belief": (["id"], {"id": "string", "reason": "string"}),
    "beliefs__accept_belief": (["id"], {"id": "string"}),
    "beliefs__list_beliefs": ([], {"status": "string"}),
    "decisions__log_decision": (
        ["title"],
        {"title": "string", "rationale": "string", "alternatives": "string"},
    ),
    "decisions__read_decisions": ([], {}),
    "federation__delegate_task": (
        ["goal"],
        {
            "goal": "string",
            "url": "string",
            "peer": "string",
            "context": "string",
            "files": "string",
            "task_timeout": "integer",
        },
    ),
    "federation__broadcast_task": (
        ["goal", "peers_list"],
        {
            "goal": "string",
            "peers_list": "string",
            "context": "string",
            "files": "string",
            "task_timeout": "integer",
        },
    ),
    "federation__list_peers": ([], {}),
    "federation__cancel_remote_task": (
        ["task_id"],
        {"task_id": "string", "url": "string", "peer": "string"},
    ),
    "federation__query_registry": (
        [],
        {"url": "string", "tags": "array", "keyword": "string", "limit": "integer"},
    ),
    "federation__sync_beliefs": (
        [],
        {
            "url": "string",
            "peer": "string",
            "direction": "string",
            "status": "string",
            "limit": "integer",
            "local_team": "string",
        },
    ),
    "expert__delegate_to_expert": (
        ["prompt", "provider"],
        {
            "provider": "string",
            "prompt": "string",
            "model": "string",
            "max_tokens": "integer",
            "temperature": "number",
        },
    ),
}


def test_offloaded_tools_do_not_block_the_loop(bus):
    """A slow built-in tool must not stall concurrent tool calls (the tools are
    offloaded to worker threads, keeping the single bus loop responsive)."""
    import threading
    import time

    slow_done = threading.Event()

    def _slow():
        # run_bash sleeps ~1s; offload should keep the loop free meanwhile.
        _call(bus, "code__run_bash", command="sleep 1")
        slow_done.set()

    t = threading.Thread(target=_slow)
    t.start()
    time.sleep(0.1)  # let the slow call get in flight
    # A fast call must return well before the slow one finishes.
    start = time.monotonic()
    out = _call(bus, "workspace__list_files")
    elapsed = time.monotonic() - start
    assert not slow_done.is_set(), "slow tool already finished — test timing too loose"
    assert elapsed < 0.5, f"fast call blocked by slow tool ({elapsed:.2f}s)"
    assert not out.startswith("ERROR")
    t.join()


def test_schema_regression(bus):
    tools = {t.wire_name: t for t in bus.list_tools(owner="m1")}
    assert set(tools) == set(EXPECTED_SCHEMAS), "tool set drifted"
    for wire, (exp_required, exp_props) in EXPECTED_SCHEMAS.items():
        schema = tools[wire].input_schema
        assert sorted(schema.get("required", [])) == sorted(exp_required), wire
        got_props = {k: v.get("type") for k, v in schema.get("properties", {}).items()}
        assert got_props == exp_props, wire
