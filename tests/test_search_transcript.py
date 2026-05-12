"""Tests for the bundled search_transcript skill."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _import_execute(skills_dir: Path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "search_transcript", skills_dir / "search_transcript.py"
    )
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod.execute


_SKILLS_DIR = Path(__file__).parent.parent / "skills"


def _make_transcript(workspace: Path, turns: list[dict]) -> Path:
    """Write a transcript.jsonl one level above the shared workspace dir."""
    team_ws = workspace.parent
    tf = team_ws / "transcript.jsonl"
    lines = [json.dumps(t) for t in turns]
    tf.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tf


@pytest.fixture
def execute():
    return _import_execute(_SKILLS_DIR)


@pytest.fixture
def ws(tmp_path):
    # Mimic the real layout: shared/ is the workspace_path passed to tools.
    shared = tmp_path / "team_ws" / "shared"
    shared.mkdir(parents=True)
    return shared


_TURNS = [
    {"index": 0, "speaker": "orchestrator", "role": "system",
     "content": "Team alpha convened. Goal: build a CLI."},
    {"index": 1, "speaker": "alice", "role": "Developer",
     "content": "I will design the database schema first."},
    {"index": 2, "speaker": "bob", "role": "Reviewer",
     "content": "The schema looks good. Approved."},
    {"index": 3, "speaker": "alice", "role": "Developer",
     "content": "Starting the API endpoints now."},
]


class TestBasicSearch:
    def test_finds_keyword_in_content(self, execute, ws):
        _make_transcript(ws, _TURNS)
        result = execute("keyword: schema", workspace_path=ws)
        assert "schema" in result.lower()
        assert "@alice" in result

    def test_case_insensitive(self, execute, ws):
        _make_transcript(ws, _TURNS)
        result = execute("keyword: SCHEMA", workspace_path=ws)
        assert "schema" in result.lower()

    def test_no_match_returns_no_matches_message(self, execute, ws):
        _make_transcript(ws, _TURNS)
        result = execute("keyword: nonexistentXYZ123", workspace_path=ws)
        assert "No matches" in result

    def test_plain_body_is_treated_as_keyword(self, execute, ws):
        _make_transcript(ws, _TURNS)
        result = execute("schema", workspace_path=ws)
        assert "schema" in result.lower()


class TestFilters:
    def test_speaker_filter(self, execute, ws):
        _make_transcript(ws, _TURNS)
        result = execute("keyword: schema\nspeaker: alice", workspace_path=ws)
        assert "@alice" in result
        assert "@bob" not in result

    def test_speaker_filter_no_match(self, execute, ws):
        _make_transcript(ws, _TURNS)
        result = execute("keyword: approved\nspeaker: alice", workspace_path=ws)
        assert "No matches" in result

    def test_limit_respected(self, execute, ws):
        many_turns = [
            {"index": i, "speaker": "m", "role": "r", "content": f"needle turn {i}"}
            for i in range(10)
        ]
        _make_transcript(ws, many_turns)
        result = execute("keyword: needle\nlimit: 3", workspace_path=ws)
        assert result.count("[turn") == 3

    def test_default_limit(self, execute, ws):
        many_turns = [
            {"index": i, "speaker": "m", "role": "r", "content": f"needle {i}"}
            for i in range(10)
        ]
        _make_transcript(ws, many_turns)
        result = execute("keyword: needle", workspace_path=ws)
        assert result.count("[turn") == 5  # default limit = 5


class TestEdgeCases:
    def test_missing_transcript_returns_error(self, execute, ws):
        result = execute("keyword: anything", workspace_path=ws)
        assert "ERROR" in result

    def test_no_workspace_returns_error(self, execute):
        result = execute("keyword: anything")
        assert "ERROR" in result

    def test_empty_keyword_returns_error(self, execute, ws):
        _make_transcript(ws, _TURNS)
        result = execute("keyword: ", workspace_path=ws)
        assert "ERROR" in result

    def test_malformed_jsonl_lines_skipped(self, execute, ws):
        team_ws = ws.parent
        tf = team_ws / "transcript.jsonl"
        tf.write_text(
            'NOT_JSON\n'
            + json.dumps({"index": 0, "speaker": "x", "role": "r", "content": "needle"})
            + "\n",
            encoding="utf-8",
        )
        result = execute("keyword: needle", workspace_path=ws)
        assert "needle" in result
