"""Tests for run-resume: the orchestrator replays persisted turns without
calling the LLM, then continues live from the first new turn.

We test the key components (Transcript resume-loading, SharedWorkspace.touch,
and the replay-queue logic) without importing docker-dependent modules.
"""

from __future__ import annotations

from pathlib import Path

from team.bus import Transcript
from team.workspace import SharedWorkspace


# --------------------------------------------------------------------------- #
# Transcript resume
# --------------------------------------------------------------------------- #


def test_transcript_resume_loads_member_turns(tmp_path: Path) -> None:
    """Replayed turns are available and the file is not truncated."""
    tr_path = tmp_path / "transcript.jsonl"
    tr = Transcript(persist_path=tr_path)
    tr.append("orchestrator", "system", "kickoff")
    tr.append("alice", "Lead", "turn 1", files_written=["report.md"])
    tr.append("bob", "Eng", "turn 2")

    # Resume: a new instance must reload the existing turns.
    tr2 = Transcript(persist_path=tr_path, resume=True)
    assert len(tr2.turns) == 3
    assert tr2.turns[1].speaker == "alice"
    assert tr2.turns[1].files_written == ["report.md"]
    assert tr2.turns[2].speaker == "bob"
    # Transcript file must not be emptied.
    assert tr_path.stat().st_size > 0


def test_transcript_resume_no_file_starts_fresh(tmp_path: Path) -> None:
    """resume=True on a missing transcript behaves like a fresh start."""
    p = tmp_path / "missing.jsonl"
    tr = Transcript(persist_path=p, resume=True)
    assert tr.turns == []
    assert p.exists()


def test_transcript_resume_false_truncates(tmp_path: Path) -> None:
    """Without resume=True, the transcript is always cleared on init."""
    p = tmp_path / "t.jsonl"
    tr = Transcript(persist_path=p)
    tr.append("a", "R", "old")
    assert p.stat().st_size > 0

    tr2 = Transcript(persist_path=p, resume=False)
    assert tr2.turns == []
    assert p.stat().st_size == 0


def test_transcript_resume_replay_queue_built(tmp_path: Path) -> None:
    """After resume-loading, member turns (non-orchestrator) are identifiable."""
    p = tmp_path / "t.jsonl"
    tr = Transcript(persist_path=p)
    tr.append("orchestrator", "system", "kickoff")
    tr.append("alice", "Lead", "hello")
    tr.append("bob", "Eng", "world")

    tr2 = Transcript(persist_path=p, resume=True)
    member_turns = [t for t in tr2.turns if t.speaker != "orchestrator"]
    assert [t.speaker for t in member_turns] == ["alice", "bob"]


# --------------------------------------------------------------------------- #
# SharedWorkspace.touch
# --------------------------------------------------------------------------- #


def test_workspace_touch_shows_in_recent_changes(tmp_path: Path) -> None:
    ws = SharedWorkspace(tmp_path)
    # touch a path that was never written in this session
    ws.touch("pre_existing/file.md")
    assert "pre_existing/file.md" in ws.recent_changes()


def test_workspace_touch_after_resume_context(tmp_path: Path) -> None:
    """Touching multiple paths (as resume does) all appear in recent_changes."""
    ws = SharedWorkspace(tmp_path)
    paths = ["a/b.txt", "c/d.py", "e.md"]
    for p in paths:
        ws.touch(p)
    recent = ws.recent_changes(limit=10)
    for p in paths:
        assert p in recent

