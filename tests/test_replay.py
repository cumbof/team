"""Tests for team.replay — transcript loader and ReplaySession."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from team.bus import Turn
from team.replay import ReplaySession, load_transcript


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _make_turn(
    index: int,
    speaker: str = "alice",
    role: str = "Researcher",
    content: str = "Hello",
    files_written: list[str] | None = None,
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    timestamp: float | None = None,
) -> Turn:
    return Turn(
        index=index,
        speaker=speaker,
        role=role,
        content=content,
        files_written=files_written or [],
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        timestamp=timestamp if timestamp is not None else float(index + 1),
    )


def _write_jsonl(path: Path, turns: list[Turn]) -> None:
    from dataclasses import asdict
    lines = [json.dumps(asdict(t)) for t in turns]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# load_transcript
# --------------------------------------------------------------------------- #


def test_load_missing_file(tmp_path):
    result = load_transcript(tmp_path / "no_such.jsonl")
    assert result == []


def test_load_empty_file(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("", encoding="utf-8")
    assert load_transcript(p) == []


def test_load_blank_lines_skipped(tmp_path):
    p = tmp_path / "t.jsonl"
    turn = _make_turn(0)
    from dataclasses import asdict
    p.write_text("\n" + json.dumps(asdict(turn)) + "\n\n", encoding="utf-8")
    result = load_transcript(p)
    assert len(result) == 1


def test_load_single_turn(tmp_path):
    p = tmp_path / "t.jsonl"
    turn = _make_turn(0, speaker="alice", content="First turn")
    _write_jsonl(p, [turn])
    result = load_transcript(p)
    assert len(result) == 1
    assert result[0].speaker == "alice"
    assert result[0].content == "First turn"


def test_load_multiple_turns(tmp_path):
    p = tmp_path / "t.jsonl"
    turns = [_make_turn(i, speaker=["alice", "bob"][i % 2]) for i in range(6)]
    _write_jsonl(p, turns)
    result = load_transcript(p)
    assert len(result) == 6
    assert result[0].speaker == "alice"
    assert result[1].speaker == "bob"


def test_load_preserves_all_fields(tmp_path):
    p = tmp_path / "t.jsonl"
    turn = _make_turn(
        3, speaker="carol", role="Lead", content="Multi\nline",
        files_written=["out.txt"], prompt_tokens=42, completion_tokens=17,
        timestamp=1_700_000_000.5,
    )
    _write_jsonl(p, [turn])
    result = load_transcript(p)[0]
    assert result.index == 3
    assert result.speaker == "carol"
    assert result.role == "Lead"
    assert result.content == "Multi\nline"
    assert result.files_written == ["out.txt"]
    assert result.prompt_tokens == 42
    assert result.completion_tokens == 17
    assert result.timestamp == pytest.approx(1_700_000_000.5)


def test_load_skips_malformed_lines(tmp_path):
    from dataclasses import asdict
    p = tmp_path / "t.jsonl"
    good = _make_turn(0)
    lines = [
        "not-json-at-all",
        json.dumps(asdict(good)),
        '{"incomplete":',           # truncated JSON
        json.dumps({"x": 1}),       # valid JSON but wrong shape
    ]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = load_transcript(p)
    # Only the one fully valid Turn survives.
    assert len(result) == 1
    assert result[0].index == 0


# --------------------------------------------------------------------------- #
# ReplaySession — construction and state properties
# --------------------------------------------------------------------------- #


def test_session_empty_total():
    s = ReplaySession([])
    assert s.total == 0


def test_session_total():
    turns = [_make_turn(i) for i in range(5)]
    s = ReplaySession(turns)
    assert s.total == 5


def test_session_initial_index():
    s = ReplaySession([_make_turn(0)])
    assert s.index == 0


def test_session_current_first():
    turns = [_make_turn(0, speaker="alice"), _make_turn(1, speaker="bob")]
    s = ReplaySession(turns)
    assert s.current() is not None
    assert s.current().speaker == "alice"


def test_session_current_empty():
    s = ReplaySession([])
    assert s.current() is None


def test_session_at_start_true_initially():
    s = ReplaySession([_make_turn(i) for i in range(3)])
    assert s.at_start is True


def test_session_at_start_false_after_next():
    s = ReplaySession([_make_turn(i) for i in range(3)])
    s.next()
    assert s.at_start is False


def test_session_at_end_single_turn():
    s = ReplaySession([_make_turn(0)])
    assert s.at_end is True


def test_session_at_end_false_initially_multi():
    s = ReplaySession([_make_turn(i) for i in range(4)])
    assert s.at_end is False


def test_session_at_end_true_after_full_traversal():
    turns = [_make_turn(i) for i in range(3)]
    s = ReplaySession(turns)
    s.jump(2)
    assert s.at_end is True


def test_session_at_end_empty():
    s = ReplaySession([])
    assert s.at_end is True


# --------------------------------------------------------------------------- #
# ReplaySession — next / prev
# --------------------------------------------------------------------------- #


def test_next_returns_true():
    s = ReplaySession([_make_turn(i) for i in range(3)])
    assert s.next() is True


def test_next_advances_index():
    s = ReplaySession([_make_turn(i) for i in range(3)])
    s.next()
    assert s.index == 1


def test_next_at_end_returns_false():
    s = ReplaySession([_make_turn(0)])
    assert s.next() is False


def test_next_at_end_position_unchanged():
    s = ReplaySession([_make_turn(i) for i in range(2)])
    s.jump(1)
    s.next()
    assert s.index == 1


def test_prev_returns_true():
    s = ReplaySession([_make_turn(i) for i in range(3)])
    s.jump(2)
    assert s.prev() is True


def test_prev_goes_back():
    s = ReplaySession([_make_turn(i) for i in range(3)])
    s.jump(2)
    s.prev()
    assert s.index == 1


def test_prev_at_start_returns_false():
    s = ReplaySession([_make_turn(i) for i in range(3)])
    assert s.prev() is False


def test_prev_at_start_position_unchanged():
    s = ReplaySession([_make_turn(i) for i in range(3)])
    s.prev()
    assert s.index == 0


# --------------------------------------------------------------------------- #
# ReplaySession — jump
# --------------------------------------------------------------------------- #


def test_jump_valid():
    s = ReplaySession([_make_turn(i) for i in range(5)])
    assert s.jump(3) is True
    assert s.index == 3


def test_jump_to_zero():
    s = ReplaySession([_make_turn(i) for i in range(5)])
    s.jump(4)
    assert s.jump(0) is True
    assert s.index == 0


def test_jump_to_last():
    turns = [_make_turn(i) for i in range(5)]
    s = ReplaySession(turns)
    assert s.jump(4) is True
    assert s.index == 4


def test_jump_out_of_range_high():
    s = ReplaySession([_make_turn(i) for i in range(3)])
    assert s.jump(10) is False
    assert s.index == 0


def test_jump_negative():
    s = ReplaySession([_make_turn(i) for i in range(3)])
    assert s.jump(-1) is False
    assert s.index == 0


# --------------------------------------------------------------------------- #
# ReplaySession — find_next / find_prev
# --------------------------------------------------------------------------- #


def test_find_next_found():
    turns = [
        _make_turn(0, speaker="alice"),
        _make_turn(1, speaker="bob"),
        _make_turn(2, speaker="alice"),
    ]
    s = ReplaySession(turns)
    # Position 0 is alice; find_next searches from index 1 → finds turn 2.
    assert s.find_next("alice") is True
    assert s.index == 2


def test_find_next_skips_current():
    turns = [_make_turn(i, speaker="alice") for i in range(3)]
    s = ReplaySession(turns)
    # All turns are alice; find_next should skip position 0 and land on 1.
    assert s.find_next("alice") is True
    assert s.index == 1


def test_find_next_not_found():
    turns = [
        _make_turn(0, speaker="alice"),
        _make_turn(1, speaker="alice"),
    ]
    s = ReplaySession(turns)
    s.jump(1)
    assert s.find_next("alice") is False
    assert s.index == 1  # unchanged


def test_find_next_from_mid():
    turns = [
        _make_turn(0, speaker="alice"),
        _make_turn(1, speaker="bob"),
        _make_turn(2, speaker="carol"),
        _make_turn(3, speaker="bob"),
    ]
    s = ReplaySession(turns)
    s.jump(1)
    assert s.find_next("bob") is True
    assert s.index == 3


def test_find_prev_found():
    turns = [
        _make_turn(0, speaker="alice"),
        _make_turn(1, speaker="bob"),
        _make_turn(2, speaker="alice"),
    ]
    s = ReplaySession(turns)
    s.jump(2)
    assert s.find_prev("alice") is True
    assert s.index == 0


def test_find_prev_skips_current():
    turns = [_make_turn(i, speaker="alice") for i in range(3)]
    s = ReplaySession(turns)
    s.jump(2)
    assert s.find_prev("alice") is True
    assert s.index == 1  # not 2


def test_find_prev_not_found():
    turns = [
        _make_turn(0, speaker="bob"),
        _make_turn(1, speaker="alice"),
    ]
    s = ReplaySession(turns)
    s.jump(0)
    assert s.find_prev("alice") is False
    assert s.index == 0


def test_find_case_insensitive_next():
    turns = [_make_turn(0, speaker="Alice"), _make_turn(1, speaker="alice")]
    s = ReplaySession(turns)
    assert s.find_next("ALICE") is True
    assert s.index == 1


def test_find_case_insensitive_prev():
    turns = [_make_turn(0, speaker="ALICE"), _make_turn(1, speaker="bob")]
    s = ReplaySession(turns)
    s.jump(1)
    assert s.find_prev("alice") is True
    assert s.index == 0


# --------------------------------------------------------------------------- #
# ReplaySession — stats
# --------------------------------------------------------------------------- #


def test_stats_empty():
    s = ReplaySession([])
    st = s.stats()
    assert st["total_turns"] == 0
    assert st["total_prompt_tokens"] == 0
    assert st["total_completion_tokens"] == 0
    assert st["duration_seconds"] is None
    assert st["files_written"] == 0


def test_stats_total_turns():
    turns = [_make_turn(i) for i in range(7)]
    s = ReplaySession(turns)
    assert s.stats()["total_turns"] == 7


def test_stats_turns_by_speaker():
    turns = [
        _make_turn(0, speaker="alice"),
        _make_turn(1, speaker="bob"),
        _make_turn(2, speaker="alice"),
        _make_turn(3, speaker="alice"),
    ]
    s = ReplaySession(turns)
    st = s.stats()
    assert st["turns_by_speaker"]["alice"] == 3
    assert st["turns_by_speaker"]["bob"] == 1


def test_stats_token_totals():
    turns = [
        _make_turn(0, prompt_tokens=10, completion_tokens=5),
        _make_turn(1, prompt_tokens=20, completion_tokens=8),
    ]
    s = ReplaySession(turns)
    st = s.stats()
    assert st["total_prompt_tokens"] == 30
    assert st["total_completion_tokens"] == 13


def test_stats_tokens_by_speaker():
    turns = [
        _make_turn(0, speaker="alice", prompt_tokens=10, completion_tokens=4),
        _make_turn(1, speaker="bob",   prompt_tokens=5,  completion_tokens=2),
        _make_turn(2, speaker="alice", prompt_tokens=6,  completion_tokens=3),
    ]
    s = ReplaySession(turns)
    st = s.stats()
    assert st["tokens_by_speaker"]["alice"]["prompt"] == 16
    assert st["tokens_by_speaker"]["alice"]["completion"] == 7
    assert st["tokens_by_speaker"]["bob"]["prompt"] == 5


def test_stats_duration():
    turns = [
        _make_turn(0, timestamp=1000.0),
        _make_turn(1, timestamp=1050.0),
        _make_turn(2, timestamp=1120.0),
    ]
    s = ReplaySession(turns)
    st = s.stats()
    assert st["duration_seconds"] == pytest.approx(120.0)


def test_stats_duration_single_turn():
    s = ReplaySession([_make_turn(0, timestamp=999.0)])
    assert s.stats()["duration_seconds"] is None


def test_stats_files_written():
    turns = [
        _make_turn(0, files_written=["a.txt", "b.txt"]),
        _make_turn(1, files_written=["c.txt"]),
        _make_turn(2, files_written=[]),
    ]
    s = ReplaySession(turns)
    assert s.stats()["files_written"] == 3


# --------------------------------------------------------------------------- #
# CLI — non-interactive mode via CliRunner
# --------------------------------------------------------------------------- #


def test_replay_cli_no_transcript(tmp_path):
    """replay exits gracefully when no transcript.jsonl exists."""
    from click.testing import CliRunner
    from team.cli import cli

    yaml_content = (
        "name: myteam\n"
        "goal: test\n"
        f"workspace: {tmp_path / 'ws'}\n"
        "members:\n"
        "  - name: alice\n"
        "    role: Researcher\n"
        "    model: llama3:8b\n"
        "    persona: You are alice.\n"
        "workflow:\n"
        "  type: round_robin\n"
        "  max_rounds: 1\n"
    )
    yaml_file = tmp_path / "myteam.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(cli, ["replay", str(yaml_file)])
    assert result.exit_code == 0
    assert "no transcript" in result.output.lower()


def test_replay_cli_non_interactive(tmp_path):
    """replay prints all turns when stdin is not a TTY."""
    from click.testing import CliRunner
    from team.cli import cli

    ws = tmp_path / "ws"
    ws.mkdir()
    yaml_content = (
        "name: myteam\n"
        "goal: test\n"
        f"workspace: {ws}\n"
        "members:\n"
        "  - name: alice\n"
        "    role: Researcher\n"
        "    model: llama3:8b\n"
        "    persona: You are alice.\n"
        "  - name: bob\n"
        "    role: Engineer\n"
        "    model: llama3:8b\n"
        "    persona: You are bob.\n"
        "workflow:\n"
        "  type: round_robin\n"
        "  max_rounds: 1\n"
    )
    yaml_file = tmp_path / "myteam.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")

    # Write a small transcript.
    turns = [
        _make_turn(0, speaker="alice", content="Hello from alice"),
        _make_turn(1, speaker="bob",   content="Hello from bob"),
    ]
    _write_jsonl(ws / "transcript.jsonl", turns)

    # CliRunner provides a non-TTY stdin by default.
    runner = CliRunner()
    result = runner.invoke(cli, ["replay", str(yaml_file)])
    assert result.exit_code == 0
    assert "alice" in result.output
    assert "bob" in result.output
    assert "Hello from alice" in result.output
    assert "Hello from bob" in result.output


def test_replay_cli_start_turn(tmp_path):
    """--from N starts at the specified turn."""
    from click.testing import CliRunner
    from team.cli import cli

    ws = tmp_path / "ws"
    ws.mkdir()
    yaml_content = (
        "name: myteam\n"
        "goal: test\n"
        f"workspace: {ws}\n"
        "members:\n"
        "  - name: alice\n"
        "    role: Researcher\n"
        "    model: llama3:8b\n"
        "    persona: You are alice.\n"
        "workflow:\n"
        "  type: round_robin\n"
        "  max_rounds: 1\n"
    )
    yaml_file = tmp_path / "myteam.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")

    turns = [_make_turn(i, content=f"Turn {i}") for i in range(4)]
    _write_jsonl(ws / "transcript.jsonl", turns)

    runner = CliRunner()
    # --from 2 should still print all turns (non-interactive fallback prints everything)
    result = runner.invoke(cli, ["replay", str(yaml_file), "--from", "2"])
    assert result.exit_code == 0


def test_replay_cli_start_turn_out_of_range(tmp_path):
    """--from N beyond transcript length exits with an error message."""
    from click.testing import CliRunner
    from team.cli import cli

    ws = tmp_path / "ws"
    ws.mkdir()
    yaml_content = (
        "name: myteam\n"
        "goal: test\n"
        f"workspace: {ws}\n"
        "members:\n"
        "  - name: alice\n"
        "    role: Researcher\n"
        "    model: llama3:8b\n"
        "    persona: You are alice.\n"
        "workflow:\n"
        "  type: round_robin\n"
        "  max_rounds: 1\n"
    )
    yaml_file = tmp_path / "myteam.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")

    turns = [_make_turn(i) for i in range(3)]
    _write_jsonl(ws / "transcript.jsonl", turns)

    runner = CliRunner()
    result = runner.invoke(cli, ["replay", str(yaml_file), "--from", "999"])
    assert result.exit_code == 0
    assert "out of range" in result.output.lower() or "range" in result.output.lower()


def test_replay_cli_speaker_not_found(tmp_path):
    """--speaker NAME exits with a message when that speaker isn't in the transcript."""
    from click.testing import CliRunner
    from team.cli import cli

    ws = tmp_path / "ws"
    ws.mkdir()
    yaml_content = (
        "name: myteam\n"
        "goal: test\n"
        f"workspace: {ws}\n"
        "members:\n"
        "  - name: alice\n"
        "    role: Researcher\n"
        "    model: llama3:8b\n"
        "    persona: You are alice.\n"
        "workflow:\n"
        "  type: round_robin\n"
        "  max_rounds: 1\n"
    )
    yaml_file = tmp_path / "myteam.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")

    turns = [_make_turn(i, speaker="alice") for i in range(3)]
    _write_jsonl(ws / "transcript.jsonl", turns)

    runner = CliRunner()
    result = runner.invoke(cli, ["replay", str(yaml_file), "--speaker", "unknown"])
    assert result.exit_code == 0
    assert "no turns" in result.output.lower()
