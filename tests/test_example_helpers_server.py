"""Tests for examples/mcp/team_helpers_server.py, driven through the bus.

Replaces the old per-skill tests (test_task_board, test_search_transcript,
test_progress_snapshot, test_critique_request) now that those helpers ship as a
single MCP server instead of exec()'d skill files.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from team.mcp.bus import MCPToolBus

_SERVER_PATH = Path(__file__).parent.parent / "examples" / "mcp" / "team_helpers_server.py"


def _load_server_module():
    spec = importlib.util.spec_from_file_location("team_helpers_server", _SERVER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def helpers(tmp_path, monkeypatch):
    """A bus with the helpers server mounted, TEAM_WORKSPACE=tmp_path/shared."""
    shared = tmp_path / "shared"
    shared.mkdir()
    monkeypatch.setenv("TEAM_WORKSPACE", str(shared))
    mod = _load_server_module()
    bus = MCPToolBus()
    bus.start()
    bus.add_inprocess_server("helpers", mod.mcp, owner="m1")
    try:
        yield bus, shared, tmp_path
    finally:
        bus.stop()


def _call(bus, tool, **args):
    return bus.call_tool(f"helpers__{tool}", args, owner="m1", timeout=10)


# --------------------------------------------------------------------------- #
# task board
# --------------------------------------------------------------------------- #


def test_task_board_flow(helpers):
    bus, shared, _ = helpers
    assert "Added to Pending" in _call(bus, "task_add", task="Design schema")
    assert "Added to Pending" in _call(bus, "task_add", task="Write tests")
    listing = _call(bus, "task_list")
    assert "Design schema" in listing and "Write tests" in listing
    assert "Marked done" in _call(bus, "task_done", match="schema")
    listing = _call(bus, "task_list")
    assert "[x] Design schema" in listing
    assert (shared / "TASKS.md").is_file()


def test_task_done_ambiguous(helpers):
    bus, _, _ = helpers
    _call(bus, "task_add", task="review module A")
    _call(bus, "task_add", task="review module B")
    assert "Ambiguous" in _call(bus, "task_done", match="review")


# --------------------------------------------------------------------------- #
# search_transcript
# --------------------------------------------------------------------------- #


def test_search_transcript(helpers):
    bus, shared, root = helpers
    # transcript lives one level above the shared dir
    transcript = root / "transcript.jsonl"
    transcript.write_text(
        "\n".join([
            json.dumps({"index": 0, "speaker": "alice", "content": "let's use pandas"}),
            json.dumps({"index": 1, "speaker": "bob", "content": "I prefer polars"}),
            json.dumps({"index": 2, "speaker": "alice", "content": "pandas is fine"}),
        ]),
        encoding="utf-8",
    )
    out = _call(bus, "search_transcript", keyword="pandas")
    assert "2 match" in out and "alice" in out
    filtered = _call(bus, "search_transcript", keyword="pandas", speaker="alice")
    assert "bob" not in filtered
    assert "No matches" in _call(bus, "search_transcript", keyword="rust")


def test_search_transcript_missing(helpers):
    bus, _, _ = helpers
    assert "transcript not found" in _call(bus, "search_transcript", keyword="x")


# --------------------------------------------------------------------------- #
# progress_snapshot
# --------------------------------------------------------------------------- #


def test_progress_snapshot_write_then_read(helpers):
    bus, shared, _ = helpers
    assert "No progress snapshot" in _call(bus, "progress_snapshot")
    out = _call(bus, "progress_snapshot", body="## Done\n- built the thing")
    assert "written to PROGRESS.md" in out
    assert "built the thing" in _call(bus, "progress_snapshot")
    assert (shared / "PROGRESS.md").is_file()


# --------------------------------------------------------------------------- #
# critique queue
# --------------------------------------------------------------------------- #


def test_critique_flow(helpers):
    bus, _, _ = helpers
    assert "No pending" in _call(bus, "list_critiques")
    posted = _call(bus, "request_critique", question="Is the API clean?", from_member="alice", file="api.py")
    assert "posted" in posted
    assert "api.py" in _call(bus, "list_critiques")
    claimed = _call(bus, "pick_critique")
    assert "Claimed" in claimed and "Is the API clean?" in claimed
    # once claimed, no longer pending
    assert "No pending" in _call(bus, "list_critiques")
