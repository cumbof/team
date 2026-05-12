"""Tests for the team test runner (F13).

Covers all assertion types:
- file_exists
- file_not_exists
- file_contains
- file_not_contains
- json_valid
- json_schema (requires jsonschema)
- transcript_contains (with and without speaker filter)
- transcript_count
- unknown type returns failure
- run_assertions aggregates results
- _parse_tests validates YAML structure
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from team.test_runner import AssertionResult, _load_turns, run_assertions


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _make_workspace(tmp_path: Path) -> Path:
    """Create a minimal shared workspace under tmp_path."""
    shared = tmp_path / "shared"
    shared.mkdir(parents=True)
    return tmp_path


def _write_file(workspace: Path, rel_path: str, content: str) -> None:
    p = workspace / "shared" / rel_path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def _write_transcript(workspace: Path, turns: list[dict]) -> Path:
    p = workspace / "transcript.jsonl"
    lines = [json.dumps(t) for t in turns]
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def _sample_turns():
    return [
        {"speaker": "orchestrator", "role": "orchestrator", "content": "kick-off", "index": 0},
        {"speaker": "dev", "role": "Engineer", "content": "I wrote hello.py in Python.", "index": 1},
        {"speaker": "reviewer", "role": "Reviewer", "content": "Looks good, no issues.", "index": 2},
    ]


# --------------------------------------------------------------------------- #
# file_exists
# --------------------------------------------------------------------------- #


class TestFileExists:
    def test_file_present(self, tmp_path):
        ws = _make_workspace(tmp_path)
        _write_file(ws, "output.txt", "hello")
        results = run_assertions(
            [{"type": "file_exists", "path": "output.txt"}],
            ws, ws / "transcript.jsonl",
        )
        assert results[0].passed is True

    def test_file_absent(self, tmp_path):
        ws = _make_workspace(tmp_path)
        results = run_assertions(
            [{"type": "file_exists", "path": "missing.txt"}],
            ws, ws / "transcript.jsonl",
        )
        assert results[0].passed is False


# --------------------------------------------------------------------------- #
# file_not_exists
# --------------------------------------------------------------------------- #


class TestFileNotExists:
    def test_file_absent(self, tmp_path):
        ws = _make_workspace(tmp_path)
        results = run_assertions(
            [{"type": "file_not_exists", "path": "ghost.txt"}],
            ws, ws / "transcript.jsonl",
        )
        assert results[0].passed is True

    def test_file_present(self, tmp_path):
        ws = _make_workspace(tmp_path)
        _write_file(ws, "ghost.txt", "boo")
        results = run_assertions(
            [{"type": "file_not_exists", "path": "ghost.txt"}],
            ws, ws / "transcript.jsonl",
        )
        assert results[0].passed is False


# --------------------------------------------------------------------------- #
# file_contains
# --------------------------------------------------------------------------- #


class TestFileContains:
    def test_text_present(self, tmp_path):
        ws = _make_workspace(tmp_path)
        _write_file(ws, "script.py", "print('hello world')")
        results = run_assertions(
            [{"type": "file_contains", "path": "script.py", "text": "hello"}],
            ws, ws / "transcript.jsonl",
        )
        assert results[0].passed is True

    def test_text_absent(self, tmp_path):
        ws = _make_workspace(tmp_path)
        _write_file(ws, "script.py", "print('hello world')")
        results = run_assertions(
            [{"type": "file_contains", "path": "script.py", "text": "foobar"}],
            ws, ws / "transcript.jsonl",
        )
        assert results[0].passed is False

    def test_file_missing(self, tmp_path):
        ws = _make_workspace(tmp_path)
        results = run_assertions(
            [{"type": "file_contains", "path": "nope.py", "text": "x"}],
            ws, ws / "transcript.jsonl",
        )
        assert results[0].passed is False


# --------------------------------------------------------------------------- #
# file_not_contains
# --------------------------------------------------------------------------- #


class TestFileNotContains:
    def test_text_absent(self, tmp_path):
        ws = _make_workspace(tmp_path)
        _write_file(ws, "report.txt", "everything is fine")
        results = run_assertions(
            [{"type": "file_not_contains", "path": "report.txt", "text": "error"}],
            ws, ws / "transcript.jsonl",
        )
        assert results[0].passed is True

    def test_text_present(self, tmp_path):
        ws = _make_workspace(tmp_path)
        _write_file(ws, "report.txt", "critical error occurred")
        results = run_assertions(
            [{"type": "file_not_contains", "path": "report.txt", "text": "error"}],
            ws, ws / "transcript.jsonl",
        )
        assert results[0].passed is False


# --------------------------------------------------------------------------- #
# json_valid
# --------------------------------------------------------------------------- #


class TestJsonValid:
    def test_valid_json(self, tmp_path):
        ws = _make_workspace(tmp_path)
        _write_file(ws, "data.json", '{"key": "value"}')
        results = run_assertions(
            [{"type": "json_valid", "path": "data.json"}],
            ws, ws / "transcript.jsonl",
        )
        assert results[0].passed is True

    def test_invalid_json(self, tmp_path):
        ws = _make_workspace(tmp_path)
        _write_file(ws, "data.json", "not json {{{")
        results = run_assertions(
            [{"type": "json_valid", "path": "data.json"}],
            ws, ws / "transcript.jsonl",
        )
        assert results[0].passed is False

    def test_file_missing(self, tmp_path):
        ws = _make_workspace(tmp_path)
        results = run_assertions(
            [{"type": "json_valid", "path": "missing.json"}],
            ws, ws / "transcript.jsonl",
        )
        assert results[0].passed is False


# --------------------------------------------------------------------------- #
# json_schema
# --------------------------------------------------------------------------- #


class TestJsonSchema:
    @pytest.mark.skip(reason="requires jsonschema")
    def _skip_if_no_jsonschema(self):
        pass

    def test_valid_schema(self, tmp_path):
        pytest.importorskip("jsonschema")
        ws = _make_workspace(tmp_path)
        _write_file(ws, "result.json", '{"name": "Alice", "score": 42}')
        results = run_assertions(
            [{
                "type": "json_schema",
                "path": "result.json",
                "schema": {
                    "type": "object",
                    "required": ["name", "score"],
                    "properties": {
                        "name": {"type": "string"},
                        "score": {"type": "integer"},
                    },
                },
            }],
            ws, ws / "transcript.jsonl",
        )
        assert results[0].passed is True

    def test_invalid_schema(self, tmp_path):
        pytest.importorskip("jsonschema")
        ws = _make_workspace(tmp_path)
        _write_file(ws, "result.json", '{"score": "not-an-int"}')
        results = run_assertions(
            [{
                "type": "json_schema",
                "path": "result.json",
                "schema": {"type": "object", "required": ["name"]},
            }],
            ws, ws / "transcript.jsonl",
        )
        assert results[0].passed is False


# --------------------------------------------------------------------------- #
# transcript_contains
# --------------------------------------------------------------------------- #


class TestTranscriptContains:
    def test_text_found_any_speaker(self, tmp_path):
        ws = _make_workspace(tmp_path)
        tp = _write_transcript(ws, _sample_turns())
        results = run_assertions(
            [{"type": "transcript_contains", "text": "Python"}],
            ws, tp,
        )
        assert results[0].passed is True

    def test_text_not_found(self, tmp_path):
        ws = _make_workspace(tmp_path)
        tp = _write_transcript(ws, _sample_turns())
        results = run_assertions(
            [{"type": "transcript_contains", "text": "Rust"}],
            ws, tp,
        )
        assert results[0].passed is False

    def test_text_found_by_speaker(self, tmp_path):
        ws = _make_workspace(tmp_path)
        tp = _write_transcript(ws, _sample_turns())
        results = run_assertions(
            [{"type": "transcript_contains", "speaker": "dev", "text": "Python"}],
            ws, tp,
        )
        assert results[0].passed is True

    def test_text_not_found_by_wrong_speaker(self, tmp_path):
        ws = _make_workspace(tmp_path)
        tp = _write_transcript(ws, _sample_turns())
        # "Python" is only in dev's turn, not reviewer's
        results = run_assertions(
            [{"type": "transcript_contains", "speaker": "reviewer", "text": "Python"}],
            ws, tp,
        )
        assert results[0].passed is False

    def test_no_transcript(self, tmp_path):
        ws = _make_workspace(tmp_path)
        results = run_assertions(
            [{"type": "transcript_contains", "text": "anything"}],
            ws, ws / "transcript.jsonl",
        )
        assert results[0].passed is False


# --------------------------------------------------------------------------- #
# transcript_count
# --------------------------------------------------------------------------- #


class TestTranscriptCount:
    def test_correct_count(self, tmp_path):
        ws = _make_workspace(tmp_path)
        tp = _write_transcript(ws, _sample_turns())
        # orchestrator is excluded; dev + reviewer = 2
        results = run_assertions(
            [{"type": "transcript_count", "count": 2}],
            ws, tp,
        )
        assert results[0].passed is True

    def test_wrong_count(self, tmp_path):
        ws = _make_workspace(tmp_path)
        tp = _write_transcript(ws, _sample_turns())
        results = run_assertions(
            [{"type": "transcript_count", "count": 5}],
            ws, tp,
        )
        assert results[0].passed is False
        assert "2" in results[0].detail


# --------------------------------------------------------------------------- #
# Unknown type
# --------------------------------------------------------------------------- #


class TestUnknownType:
    def test_unknown_type_fails(self, tmp_path):
        ws = _make_workspace(tmp_path)
        results = run_assertions(
            [{"type": "does_not_exist"}],
            ws, ws / "transcript.jsonl",
        )
        assert results[0].passed is False
        assert "unknown" in results[0].detail.lower()


# --------------------------------------------------------------------------- #
# AssertionResult name fallback
# --------------------------------------------------------------------------- #


class TestAssertionResultName:
    def test_uses_name_field(self, tmp_path):
        ws = _make_workspace(tmp_path)
        results = run_assertions(
            [{"type": "file_exists", "name": "My custom name", "path": "x.txt"}],
            ws, ws / "transcript.jsonl",
        )
        assert results[0].name == "My custom name"

    def test_falls_back_to_type(self, tmp_path):
        ws = _make_workspace(tmp_path)
        results = run_assertions(
            [{"type": "file_exists", "path": "x.txt"}],
            ws, ws / "transcript.jsonl",
        )
        assert results[0].name == "file_exists"


# --------------------------------------------------------------------------- #
# _parse_tests in config.py
# --------------------------------------------------------------------------- #


class TestParseTests:
    def test_valid_tests(self, tmp_path):
        from team.config import load_team
        y = tmp_path / "team.yaml"
        y.write_text(f"""\
name: test-team
goal: test goal
workspace: {tmp_path}/runs

members:
  - name: dev
    role: Dev
    model: m
    persona: You code.

tests:
  - name: creates output
    type: file_exists
    path: output.txt
  - type: transcript_count
    count: 1
""")
        cfg = load_team(str(y))
        assert len(cfg.tests) == 2
        assert cfg.tests[0]["type"] == "file_exists"
        assert cfg.tests[1]["type"] == "transcript_count"

    def test_no_tests_section(self, tmp_path):
        from team.config import load_team
        y = tmp_path / "team.yaml"
        y.write_text(f"""\
name: test-team
goal: test goal
workspace: {tmp_path}/runs

members:
  - name: dev
    role: Dev
    model: m
    persona: You code.
""")
        cfg = load_team(str(y))
        assert cfg.tests == []

    def test_missing_type_raises(self, tmp_path):
        from team.config import TeamConfigError, load_team
        y = tmp_path / "team.yaml"
        y.write_text(f"""\
name: test-team
goal: test goal
workspace: {tmp_path}/runs

members:
  - name: dev
    role: Dev
    model: m
    persona: You code.

tests:
  - name: bad assertion
    path: output.txt
""")
        with pytest.raises(TeamConfigError, match="type"):
            load_team(str(y))
