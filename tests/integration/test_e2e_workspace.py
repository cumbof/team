"""Integration tests: workspace behaviour with a real LLM.

Covers:
- Shared and per-member workspace directories are created by up().
- Checkpoints directory is created after a run.
- A file pre-placed in the shared workspace is listed in the workspace
  (verified via the SharedWorkspace API, not by LLM output quality).
- A member that emits a ``file:`` block has that file written to disk.
"""

from __future__ import annotations

import textwrap

import pytest

from team.config import load_team
from team.orchestrator import Orchestrator
from team.workspace import SharedWorkspace


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_yaml(ollama_url: str, model: str, workspace: str, persona: str,
               max_rounds: int = 1) -> str:
    return textwrap.dedent(f"""\
        name: inttest-ws
        goal: Follow your instructions precisely.
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
          - name: writer
            role: Writer
            model: {model}
            persona: |
              {persona}
    """)


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestWorkspaceDirs:
    def test_shared_and_member_dirs_created_on_up(self, ollama_url, tiny_model, tmp_path):
        workspace = tmp_path / "ws"
        yaml_text = _make_yaml(
            ollama_url=ollama_url,
            model=tiny_model,
            workspace=str(workspace),
            persona="You are a helpful assistant. Reply with one sentence.",
        )
        team_file = tmp_path / "team.yaml"
        team_file.write_text(yaml_text)
        cfg = load_team(team_file)
        orch = Orchestrator(cfg)
        orch.up()
        try:
            assert (workspace / "shared").is_dir(), "shared/ must exist after up()"
            assert (workspace / "members" / "writer").is_dir(), "members/writer/ must exist after up()"
        finally:
            orch.down(remove_volumes=False)

    def test_checkpoints_dir_created_after_run(self, ollama_url, tiny_model, tmp_path):
        workspace = tmp_path / "ws"
        yaml_text = _make_yaml(
            ollama_url=ollama_url,
            model=tiny_model,
            workspace=str(workspace),
            persona="You are a helpful assistant. Reply with one sentence.",
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

        assert (workspace / "checkpoints").is_dir(), "checkpoints/ must exist after a run"


class TestPrePlacedFile:
    def test_pre_placed_file_is_readable_via_workspace_api(self, ollama_url, tiny_model, tmp_path):
        """Files placed in shared/ before the run are visible to SharedWorkspace."""
        workspace = tmp_path / "ws"
        shared = workspace / "shared"
        shared.mkdir(parents=True)
        (shared / "brief.txt").write_text("The project brief: build a hello-world CLI.\n")

        yaml_text = _make_yaml(
            ollama_url=ollama_url,
            model=tiny_model,
            workspace=str(workspace),
            persona="You are a helpful assistant. Reply with one sentence.",
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

        ws = SharedWorkspace(workspace)
        files = ws.list_files()
        assert any("brief.txt" in f for f in files), (
            "brief.txt should be visible in the shared workspace listing"
        )


class TestFileBlockWrite:
    """Verify that a ``file:`` block in an LLM reply is written to disk.

    This test is best-effort: very small models may not reliably follow the
    protocol, so we check for the file but do not fail if the model chose a
    different reply format.  A warning is emitted when the file is absent.
    """

    def test_file_block_written_when_emitted(self, ollama_url, tiny_model, tmp_path):
        workspace = tmp_path / "ws"
        # Extremely explicit instruction — maximises the chance even a tiny
        # model will emit the required fenced block.
        persona = textwrap.dedent("""\
            You must reply with ONLY the following text, nothing else:

            ```file:hello.txt
            hello world
            ```
        """)
        yaml_text = _make_yaml(
            ollama_url=ollama_url,
            model=tiny_model,
            workspace=str(workspace),
            persona=persona,
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

        hello_file = workspace / "shared" / "hello.txt"
        if not hello_file.exists():
            import warnings
            warnings.warn(
                f"Model '{tiny_model}' did not emit a file: block — "
                "file-write via LLM output was not exercised in this run. "
                "Consider using a larger model for more reliable protocol compliance.",
                stacklevel=1,
            )
        else:
            assert "hello" in hello_file.read_text().lower()
