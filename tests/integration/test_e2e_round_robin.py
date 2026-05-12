"""Integration tests: round-robin workflow with a real LLM.

Covers:
- Single member produces a non-empty response.
- Transcript is written to disk after a run.
- Two members each get a turn and both produce output.
- The orchestrator kickoff turn is recorded in the transcript.
"""

from __future__ import annotations

import textwrap

import pytest

from team.config import load_team
from team.orchestrator import Orchestrator


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_team_yaml(ollama_url: str, model: str, workspace: str, members: str,
                    max_rounds: int = 1) -> str:
    return textwrap.dedent(f"""\
        name: inttest-rr
        goal: Respond with exactly one short sentence confirming you are ready.
        workspace: {workspace}
        defaults:
          ollama_url: {ollama_url}
          gpus: none
          context_window: 2048
          temperature: 0.0
          keep_alive: "0"
        workflow:
          type: round_robin
          max_rounds: {max_rounds}
        members:
        {members}
    """)


def _member_block(name: str, model: str, role: str = "Assistant") -> str:
    return textwrap.dedent(f"""\
          - name: {name}
            role: {role}
            model: {model}
            persona: |
              You are a helpful assistant. Reply with one short sentence only.
    """)


def _run(yaml_text: str, team_file_path) -> Orchestrator:
    team_file_path.write_text(yaml_text)
    cfg = load_team(team_file_path)
    orch = Orchestrator(cfg)
    orch.up()
    try:
        orch.run()
    finally:
        orch.down(remove_volumes=False)
    return orch


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestSingleMember:
    def test_member_produces_output(self, ollama_url, tiny_model, tmp_path):
        yaml_text = _make_team_yaml(
            ollama_url=ollama_url,
            model=tiny_model,
            workspace=str(tmp_path / "ws"),
            members=_member_block("alpha", tiny_model),
        )
        orch = _run(yaml_text, tmp_path / "team.yaml")

        member_turns = [t for t in orch.transcript.turns if t.speaker == "alpha"]
        assert len(member_turns) == 1, "Expected exactly one turn for 'alpha'"
        assert member_turns[0].content.strip(), "Member response must not be empty"

    def test_orchestrator_kickoff_recorded(self, ollama_url, tiny_model, tmp_path):
        yaml_text = _make_team_yaml(
            ollama_url=ollama_url,
            model=tiny_model,
            workspace=str(tmp_path / "ws"),
            members=_member_block("alpha", tiny_model),
        )
        orch = _run(yaml_text, tmp_path / "team.yaml")

        orchestrator_turns = [t for t in orch.transcript.turns if t.speaker == "orchestrator"]
        assert orchestrator_turns, "Orchestrator kickoff turn must be recorded"
        assert "inttest-rr" in orchestrator_turns[0].content

    def test_transcript_persisted_to_disk(self, ollama_url, tiny_model, tmp_path):
        workspace = tmp_path / "ws"
        yaml_text = _make_team_yaml(
            ollama_url=ollama_url,
            model=tiny_model,
            workspace=str(workspace),
            members=_member_block("alpha", tiny_model),
        )
        _run(yaml_text, tmp_path / "team.yaml")

        transcript_file = workspace / "transcript.jsonl"
        assert transcript_file.exists(), "transcript.jsonl must be written to the workspace"
        lines = transcript_file.read_text().strip().splitlines()
        assert len(lines) >= 2, "Transcript must have at least two entries (kickoff + member)"


class TestTwoMembers:
    def test_both_members_produce_output(self, ollama_url, tiny_model, tmp_path):
        members = (
            _member_block("alpha", tiny_model, role="First")
            + _member_block("beta", tiny_model, role="Second")
        )
        yaml_text = _make_team_yaml(
            ollama_url=ollama_url,
            model=tiny_model,
            workspace=str(tmp_path / "ws"),
            members=members,
        )
        orch = _run(yaml_text, tmp_path / "team.yaml")

        for name in ("alpha", "beta"):
            turns = [t for t in orch.transcript.turns if t.speaker == name]
            assert len(turns) == 1, f"Expected one turn for '{name}'"
            assert turns[0].content.strip(), f"'{name}' response must not be empty"

    def test_second_member_sees_first_member_in_transcript(self, ollama_url, tiny_model, tmp_path):
        """The transcript fed to beta must include alpha's turn."""
        members = (
            _member_block("alpha", tiny_model, role="First")
            + _member_block("beta", tiny_model, role="Second")
        )
        yaml_text = _make_team_yaml(
            ollama_url=ollama_url,
            model=tiny_model,
            workspace=str(tmp_path / "ws"),
            members=members,
        )
        orch = _run(yaml_text, tmp_path / "team.yaml")

        turns = orch.transcript.turns
        alpha_idx = next(i for i, t in enumerate(turns) if t.speaker == "alpha")
        beta_idx = next(i for i, t in enumerate(turns) if t.speaker == "beta")
        assert alpha_idx < beta_idx, "alpha must speak before beta in the transcript"
