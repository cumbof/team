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
