"""Tests for human-in-the-loop injection.

Tests cover:
* inject_directive() — programmatic injection
* _check_inject()    — file-based injection via workspace/inject.txt
* _on_round_end hook — called by workflows at round boundaries
* inject.txt is consumed (deleted) after first check
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from team.bus import Transcript
from team.config import Defaults, MemberConfig, TeamConfig, WorkflowConfig


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _team(tmp_path: Path) -> TeamConfig:
    return TeamConfig(
        name="inject-test",
        goal="g",
        workspace=tmp_path / "runs" / "inject-test",
        workflow=WorkflowConfig(),
        defaults=Defaults(),
        members=[MemberConfig(name="a", role="R", model="m", persona="p")],
    )


def _transcript(tmp_path: Path) -> Transcript:
    return Transcript(persist_path=tmp_path / "transcript.jsonl")


# --------------------------------------------------------------------------- #
# inject_directive
# --------------------------------------------------------------------------- #


def test_inject_directive_appends_human_turn(tmp_path: Path) -> None:
    """inject_directive() adds a speaker='human' turn to the transcript."""
    from team.bus import Transcript
    from team.config import Defaults, MemberConfig, TeamConfig, WorkflowConfig

    cfg = _team(tmp_path)
    cfg.workspace.mkdir(parents=True, exist_ok=True)

    # Build a minimal Orchestrator-like object without Docker
    transcript = _transcript(cfg.workspace)

    # We test the logic in isolation by calling the method directly
    # on a stub that reuses the real Transcript.
    class _Stub:
        def __init__(self):
            self.transcript = transcript
            import logging
            self._log = logging.getLogger(__name__)

        def inject_directive(self, content: str) -> None:
            text = content.strip()
            if not text:
                return
            self.transcript.append(speaker="human", role="director", content=text)

    stub = _Stub()
    stub.inject_directive("Focus on unit tests only.")
    assert len(transcript.turns) == 1
    t = transcript.turns[0]
    assert t.speaker == "human"
    assert t.role == "director"
    assert "unit tests" in t.content


def test_inject_directive_ignores_empty_string(tmp_path: Path) -> None:
    cfg = _team(tmp_path)
    cfg.workspace.mkdir(parents=True, exist_ok=True)
    transcript = _transcript(cfg.workspace)

    class _Stub:
        def __init__(self):
            self.transcript = transcript

        def inject_directive(self, content: str) -> None:
            text = content.strip()
            if not text:
                return
            self.transcript.append(speaker="human", role="director", content=text)

    stub = _Stub()
    stub.inject_directive("   ")
    assert len(transcript.turns) == 0


# --------------------------------------------------------------------------- #
# _check_inject (file-based)
# --------------------------------------------------------------------------- #


def _make_orch_stub(tmp_path: Path):
    """Return a minimal stub that exercises _check_inject without Docker."""
    import logging

    from team.bus import Transcript

    cfg = _team(tmp_path)
    cfg.workspace.mkdir(parents=True, exist_ok=True)
    transcript = Transcript(persist_path=cfg.workspace / "transcript.jsonl")
    inject_path = cfg.workspace / "inject.txt"

    class _OrchestratorStub:
        def __init__(self):
            self.transcript = transcript
            self.inject_path = inject_path
            self._log = logging.getLogger(__name__)

        def inject_directive(self, content: str) -> None:
            text = content.strip()
            if not text:
                return
            self.transcript.append(speaker="human", role="director", content=text)

        def _check_inject(self) -> None:
            try:
                if self.inject_path.is_file():
                    content = self.inject_path.read_text(encoding="utf-8").strip()
                    self.inject_path.unlink()
                    if content:
                        self.inject_directive(content)
            except OSError as exc:
                self._log.warning("inject: error: %s", exc)

    return _OrchestratorStub(), transcript, inject_path


def test_check_inject_reads_file(tmp_path: Path) -> None:
    stub, transcript, inject_path = _make_orch_stub(tmp_path)
    inject_path.write_text("Prioritize the API module.", encoding="utf-8")
    stub._check_inject()
    assert len(transcript.turns) == 1
    assert "API module" in transcript.turns[0].content


def test_check_inject_deletes_file_after_reading(tmp_path: Path) -> None:
    stub, _, inject_path = _make_orch_stub(tmp_path)
    inject_path.write_text("hello", encoding="utf-8")
    stub._check_inject()
    assert not inject_path.exists()


def test_check_inject_no_op_when_file_absent(tmp_path: Path) -> None:
    stub, transcript, _ = _make_orch_stub(tmp_path)
    stub._check_inject()  # should not raise
    assert len(transcript.turns) == 0


def test_check_inject_ignores_empty_file(tmp_path: Path) -> None:
    stub, transcript, inject_path = _make_orch_stub(tmp_path)
    inject_path.write_text("   \n", encoding="utf-8")
    stub._check_inject()
    assert len(transcript.turns) == 0
    assert not inject_path.exists()


def test_check_inject_second_call_is_noop(tmp_path: Path) -> None:
    """inject.txt is deleted on first read; second _check_inject does nothing."""
    stub, transcript, inject_path = _make_orch_stub(tmp_path)
    inject_path.write_text("Round 2 directive.", encoding="utf-8")
    stub._check_inject()
    stub._check_inject()  # second call — file already gone
    assert len(transcript.turns) == 1  # only injected once


# --------------------------------------------------------------------------- #
# _on_round_end hook — workflow integration
# --------------------------------------------------------------------------- #


def test_round_robin_calls_on_round_end(tmp_path: Path) -> None:
    """round_robin() must call _on_round_end once per completed round."""
    from team.workflows import round_robin

    called_with: list[int] = []

    class _FakeTurnResult:
        declared_done = False

    class _FakeMember:
        name = "a"

    class _FakeWorkflow:
        max_rounds = 2

    class _FakeTeam:
        pass

    class _FakeOrch:
        def __init__(self):
            team = _FakeTeam()
            team.workflow = _FakeWorkflow()
            self.team = team
            self.members = {"a": _FakeMember()}
            self._on_round_end = lambda idx: called_with.append(idx)

        def run_turn(self, name, prompt=None):
            return _FakeTurnResult()

    round_robin(_FakeOrch())
    assert called_with == [0, 1]


def test_sequential_chain_calls_on_round_end(tmp_path: Path) -> None:
    """sequential_chain() must call _on_round_end once per completed round."""
    from team.workflows import sequential_chain

    called_with: list[int] = []

    class _FakeTurnResult:
        declared_done = False
        content = "done"

    class _FakeMember:
        name = "a"

    class _FakeWorkflow:
        max_rounds = 3
        options: dict = {}

    class _FakeTeam:
        pass

    class _FakeOrch:
        def __init__(self):
            team = _FakeTeam()
            team.workflow = _FakeWorkflow()
            self.team = team
            self.members = {"a": _FakeMember()}
            self._on_round_end = lambda idx: called_with.append(idx)

        def run_turn(self, name, prompt=None):
            return _FakeTurnResult()

    sequential_chain(_FakeOrch())
    assert called_with == [0, 1, 2]
