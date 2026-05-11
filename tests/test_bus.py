from pathlib import Path

from team.bus import Transcript


def test_transcript_persists(tmp_path: Path) -> None:
    p = tmp_path / "t.jsonl"
    tr = Transcript(persist_path=p)
    tr.append("orchestrator", "system", "hello")
    tr.append("alice", "Lead", "world", files_written=["a.md"])
    lines = p.read_text().strip().splitlines()
    assert len(lines) == 2
    assert "alice" in lines[1]


def test_transcript_render_marks_viewer() -> None:
    tr = Transcript()
    tr.append("alice", "Lead", "first")
    tr.append("bob", "Eng", "reply")
    rendered = tr.render(viewer="alice")
    assert "@alice (you)" in rendered
    assert "@bob" in rendered and "@bob (you)" not in rendered


def test_transcript_max_turns() -> None:
    tr = Transcript()
    for i in range(5):
        tr.append("a", "r", f"msg{i}")
    out = tr.render(max_turns=2)
    assert "msg3" in out and "msg4" in out
    assert "msg0" not in out


def test_transcript_resume_loads_existing(tmp_path: Path) -> None:
    p = tmp_path / "t.jsonl"
    # Write a transcript with two turns.
    tr = Transcript(persist_path=p)
    tr.append("orchestrator", "system", "kickoff")
    tr.append("alice", "Lead", "first message", files_written=["out.md"])
    assert len(tr.turns) == 2

    # Resume: a new Transcript instance should reload those turns.
    tr2 = Transcript(persist_path=p, resume=True)
    assert len(tr2.turns) == 2
    assert tr2.turns[0].speaker == "orchestrator"
    assert tr2.turns[1].speaker == "alice"
    assert tr2.turns[1].files_written == ["out.md"]
    # The file must not be truncated.
    assert p.stat().st_size > 0


def test_transcript_resume_missing_file_starts_fresh(tmp_path: Path) -> None:
    p = tmp_path / "nonexistent.jsonl"
    tr = Transcript(persist_path=p, resume=True)
    assert tr.turns == []
    assert p.exists()  # file is created (empty) so future appends work


# --------------------------------------------------------------------------- #
# Transcript.stats()
# --------------------------------------------------------------------------- #


def test_stats_empty_transcript() -> None:
    tr = Transcript()
    s = tr.stats()
    assert s["total_turns"] == 0
    assert s["turns_by_speaker"] == {}
    assert s["total_prompt_tokens"] == 0
    assert s["total_completion_tokens"] == 0
    assert s["duration_seconds"] is None
    assert s["files_written"] == 0


def test_stats_counts_turns_and_tokens() -> None:
    tr = Transcript()
    tr.append("orchestrator", "system", "kickoff", prompt_tokens=10, completion_tokens=5)
    tr.append("alice", "Lead", "hello", files_written=["a.txt"], prompt_tokens=20, completion_tokens=8)
    tr.append("bob", "Eng", "reply", prompt_tokens=15, completion_tokens=6)
    tr.append("alice", "Lead", "done", files_written=["b.txt", "c.txt"], prompt_tokens=12, completion_tokens=4)

    s = tr.stats()
    assert s["total_turns"] == 4
    assert s["turns_by_speaker"]["orchestrator"] == 1
    assert s["turns_by_speaker"]["alice"] == 2
    assert s["turns_by_speaker"]["bob"] == 1

    assert s["total_prompt_tokens"] == 10 + 20 + 15 + 12
    assert s["total_completion_tokens"] == 5 + 8 + 6 + 4

    assert s["tokens_by_speaker"]["alice"]["prompt"] == 20 + 12
    assert s["tokens_by_speaker"]["alice"]["completion"] == 8 + 4

    # files_written counts individual file paths across all turns
    assert s["files_written"] == 3  # a.txt, b.txt, c.txt


def test_stats_duration_two_or_more_turns() -> None:
    tr = Transcript()
    tr.append("a", "r", "first")
    tr.append("b", "r", "last")
    # Manually set timestamps so duration is deterministic
    tr.turns[0].timestamp = 1000.0
    tr.turns[1].timestamp = 1060.0

    s = tr.stats()
    assert s["duration_seconds"] == pytest.approx(60.0)


def test_stats_duration_single_turn_is_none() -> None:
    tr = Transcript()
    tr.append("a", "r", "only turn")
    assert tr.stats()["duration_seconds"] is None


import pytest  # noqa: E402 — kept at bottom to match existing test style
