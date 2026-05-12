"""Integration tests: parallel workflow with a real LLM.

Covers:
- Two members run concurrently; both produce output.
- Transcript entries appear in declaration order (deterministic ordering).
- Token counts are accumulated separately per member.
"""

from __future__ import annotations

import textwrap

import pytest

from team.config import load_team
from team.orchestrator import Orchestrator


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_parallel_yaml(ollama_url: str, model: str, workspace: str,
                        max_rounds: int = 1) -> str:
    return textwrap.dedent(f"""\
        name: inttest-parallel
        goal: Each member should confirm they are ready with one sentence.
        workspace: {workspace}
        defaults:
          ollama_url: {ollama_url}
          gpus: none
          context_window: 2048
          temperature: 0.0
          keep_alive: "0"
        workflow:
          type: parallel
          max_rounds: {max_rounds}
        members:
          - name: alpha
            role: First
            model: {model}
            persona: |
              You are a helpful assistant. Reply with one short sentence only.
          - name: beta
            role: Second
            model: {model}
            persona: |
              You are a helpful assistant. Reply with one short sentence only.
    """)


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestParallelWorkflow:
    def test_both_members_produce_output(self, ollama_url, tiny_model, tmp_path):
        yaml_text = _make_parallel_yaml(
            ollama_url=ollama_url,
            model=tiny_model,
            workspace=str(tmp_path / "ws"),
        )
        team_file = tmp_path / "team.yaml"
        team_file.write_text(yaml_text)
        cfg = load_team(team_file)
        orch = Orchestrator(cfg)
        orch.up()
        try:
            orch.run()
        finally:
            orch.down(remove_volumes=False)

        for name in ("alpha", "beta"):
            turns = [t for t in orch.transcript.turns if t.speaker == name]
            assert len(turns) >= 1, f"Expected at least one turn for '{name}'"
            assert turns[0].content.strip(), f"'{name}' must produce non-empty output"

    def test_transcript_order_is_deterministic(self, ollama_url, tiny_model, tmp_path):
        """alpha must appear before beta in the transcript regardless of thread scheduling."""
        yaml_text = _make_parallel_yaml(
            ollama_url=ollama_url,
            model=tiny_model,
            workspace=str(tmp_path / "ws"),
        )
        team_file = tmp_path / "team.yaml"
        team_file.write_text(yaml_text)
        cfg = load_team(team_file)
        orch = Orchestrator(cfg)
        orch.up()
        try:
            orch.run()
        finally:
            orch.down(remove_volumes=False)

        speakers = [t.speaker for t in orch.transcript.turns if t.speaker not in ("orchestrator",)]
        alpha_idx = speakers.index("alpha")
        beta_idx = speakers.index("beta")
        assert alpha_idx < beta_idx, (
            "Parallel workflow must commit turns in declaration order (alpha before beta)"
        )

    def test_token_counts_accumulated_per_member(self, ollama_url, tiny_model, tmp_path):
        yaml_text = _make_parallel_yaml(
            ollama_url=ollama_url,
            model=tiny_model,
            workspace=str(tmp_path / "ws"),
        )
        team_file = tmp_path / "team.yaml"
        team_file.write_text(yaml_text)
        cfg = load_team(team_file)
        orch = Orchestrator(cfg)
        orch.up()
        try:
            orch.run()
        finally:
            orch.down(remove_volumes=False)

        for name in ("alpha", "beta"):
            totals = orch._token_totals.get(name, {})
            # At least some tokens were consumed (prompt or completion).
            total = totals.get("prompt", 0) + totals.get("completion", 0)
            assert total > 0, f"Expected non-zero token count for '{name}', got {totals}"
