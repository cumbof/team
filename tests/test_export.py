"""Tests for team.export — report generation from a transcript and workspace.

No Docker required; we just create temporary files and call the export
functions directly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from team.bus import Transcript
from team.config import Defaults, MemberConfig, TeamConfig, WorkflowConfig
from team.export import (
    export_run,
    _compute_token_summary,
    _load_transcript,
    _load_shared_files,
    _render_markdown,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _team(tmp_path: Path) -> TeamConfig:
    members = [
        MemberConfig(name="lead", role="Project Lead", model="llama3.1:8b", persona="p1"),
        MemberConfig(name="eng", role="Engineer", model="qwen2.5-coder:7b", persona="p2"),
    ]
    return TeamConfig(
        name="test-team",
        goal="Build a widget.\n",
        workspace=tmp_path,
        workflow=WorkflowConfig(type="round_robin", max_rounds=3),
        defaults=Defaults(),
        members=members,
    )


def _seed_workspace(tmp_path: Path) -> None:
    """Write a small transcript and a shared file."""
    tr = Transcript(persist_path=tmp_path / "transcript.jsonl")
    tr.append("orchestrator", "system", "kickoff")
    tr.append("lead", "Project Lead", "Let's get started.", files_written=[])
    tr.append("eng", "Engineer", "Here is the code.\n```file:src/widget.py\nprint('hi')\n```", files_written=["src/widget.py"])

    shared = tmp_path / "shared"
    shared.mkdir(parents=True, exist_ok=True)
    (shared / "src").mkdir()
    (shared / "src" / "widget.py").write_text("print('hi')\n")


# --------------------------------------------------------------------------- #
# _load_transcript
# --------------------------------------------------------------------------- #


def test_load_transcript_returns_all_turns(tmp_path: Path) -> None:
    _seed_workspace(tmp_path)
    turns = _load_transcript(tmp_path / "transcript.jsonl")
    assert len(turns) == 3
    assert turns[0]["speaker"] == "orchestrator"
    assert turns[1]["speaker"] == "lead"


def test_load_transcript_missing_file(tmp_path: Path) -> None:
    turns = _load_transcript(tmp_path / "nonexistent.jsonl")
    assert turns == []


# --------------------------------------------------------------------------- #
# _load_shared_files
# --------------------------------------------------------------------------- #


def test_load_shared_files_returns_contents(tmp_path: Path) -> None:
    _seed_workspace(tmp_path)
    files = _load_shared_files(tmp_path / "shared")
    assert "src/widget.py" in files
    assert "print('hi')" in files["src/widget.py"]


def test_load_shared_files_empty_dir(tmp_path: Path) -> None:
    (tmp_path / "shared").mkdir()
    files = _load_shared_files(tmp_path / "shared")
    assert files == {}


def test_load_shared_files_missing_dir(tmp_path: Path) -> None:
    files = _load_shared_files(tmp_path / "no_such_dir")
    assert files == {}


# --------------------------------------------------------------------------- #
# Markdown export
# --------------------------------------------------------------------------- #


def test_export_markdown_contains_team_name(tmp_path: Path) -> None:
    _seed_workspace(tmp_path)
    cfg = _team(tmp_path)
    text = export_run(cfg, fmt="markdown")
    assert "test-team" in text


def test_export_markdown_contains_goal(tmp_path: Path) -> None:
    _seed_workspace(tmp_path)
    cfg = _team(tmp_path)
    text = export_run(cfg, fmt="markdown")
    assert "Build a widget." in text


def test_export_markdown_contains_member_names(tmp_path: Path) -> None:
    _seed_workspace(tmp_path)
    cfg = _team(tmp_path)
    text = export_run(cfg, fmt="markdown")
    assert "@lead" in text
    assert "@eng" in text


def test_export_markdown_contains_turn_content(tmp_path: Path) -> None:
    _seed_workspace(tmp_path)
    cfg = _team(tmp_path)
    text = export_run(cfg, fmt="markdown")
    assert "Let's get started." in text
    assert "Here is the code." in text


def test_export_markdown_embeds_artifact(tmp_path: Path) -> None:
    _seed_workspace(tmp_path)
    cfg = _team(tmp_path)
    text = export_run(cfg, fmt="markdown")
    assert "src/widget.py" in text
    assert "print('hi')" in text


def test_export_markdown_no_transcript(tmp_path: Path) -> None:
    """Exporting without a transcript still returns a valid document."""
    cfg = _team(tmp_path)
    (tmp_path / "shared").mkdir(parents=True, exist_ok=True)
    # Touch the transcript so export_run doesn't return an empty turn list error
    (tmp_path / "transcript.jsonl").write_text("")
    text = export_run(cfg, fmt="markdown")
    assert "test-team" in text


# --------------------------------------------------------------------------- #
# HTML export
# --------------------------------------------------------------------------- #


def test_export_html_is_valid_html(tmp_path: Path) -> None:
    _seed_workspace(tmp_path)
    cfg = _team(tmp_path)
    html = export_run(cfg, fmt="html")
    assert html.strip().startswith("<!DOCTYPE html>")
    assert "<body>" in html
    assert "</html>" in html


def test_export_html_contains_team_name(tmp_path: Path) -> None:
    _seed_workspace(tmp_path)
    cfg = _team(tmp_path)
    html = export_run(cfg, fmt="html")
    assert "test-team" in html


def test_export_html_contains_turn_content(tmp_path: Path) -> None:
    _seed_workspace(tmp_path)
    cfg = _team(tmp_path)
    html = export_run(cfg, fmt="html")
    assert "Let&#39;s get started." in html or "Let's get started." in html


def test_export_unknown_format_raises(tmp_path: Path) -> None:
    cfg = _team(tmp_path)
    (tmp_path / "transcript.jsonl").write_text("")
    (tmp_path / "shared").mkdir(exist_ok=True)
    with pytest.raises(ValueError, match="unknown format"):
        export_run(cfg, fmt="pdf")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Token summary
# --------------------------------------------------------------------------- #


def _seed_with_tokens(tmp_path: Path) -> None:
    """Transcript with token counts so _compute_token_summary has data."""
    tr = Transcript(persist_path=tmp_path / "transcript.jsonl")
    tr.append("orchestrator", "system", "kickoff")
    tr.append(
        "lead", "Project Lead", "First response.",
        files_written=[],
        prompt_tokens=100, completion_tokens=50,
    )
    tr.append(
        "eng", "Engineer", "Second response.",
        files_written=[],
        prompt_tokens=80, completion_tokens=40,
    )
    (tmp_path / "shared").mkdir(parents=True, exist_ok=True)


def test_compute_token_summary_aggregates(tmp_path: Path) -> None:
    _seed_with_tokens(tmp_path)
    cfg = _team(tmp_path)
    turns = _load_transcript(tmp_path / "transcript.jsonl")
    summary = _compute_token_summary(cfg, turns)
    assert summary["total_prompt"] == 180
    assert summary["total_completion"] == 90
    assert summary["total_tokens"] == 270
    assert "lead" in summary["by_member"]
    assert summary["by_member"]["lead"]["prompt"] == 100
    assert summary["by_member"]["lead"]["completion"] == 50


def test_compute_token_summary_empty(tmp_path: Path) -> None:
    cfg = _team(tmp_path)
    summary = _compute_token_summary(cfg, [])
    assert summary["total_tokens"] == 0
    assert summary["by_member"] == {}


def test_export_markdown_contains_token_table(tmp_path: Path) -> None:
    _seed_with_tokens(tmp_path)
    cfg = _team(tmp_path)
    text = export_run(cfg, fmt="markdown")
    # Token table header row
    assert "Prompt" in text
    assert "Completion" in text


def test_export_markdown_metadata_footer(tmp_path: Path) -> None:
    _seed_with_tokens(tmp_path)
    cfg = _team(tmp_path)
    text = export_run(cfg, fmt="markdown")
    # Footer with version + generated timestamp
    assert "team-core" in text


# --------------------------------------------------------------------------- #
# JSON export
# --------------------------------------------------------------------------- #


def test_export_json_is_valid_json(tmp_path: Path) -> None:
    _seed_workspace(tmp_path)
    cfg = _team(tmp_path)
    raw = export_run(cfg, fmt="json")
    data = json.loads(raw)
    assert isinstance(data, dict)


def test_export_json_structure(tmp_path: Path) -> None:
    _seed_workspace(tmp_path)
    cfg = _team(tmp_path)
    data = json.loads(export_run(cfg, fmt="json"))
    assert data["format_version"] == 1
    assert "team" in data
    assert "turns" in data
    assert "token_usage" in data
    assert "artifacts" in data


def test_export_json_team_fields(tmp_path: Path) -> None:
    _seed_workspace(tmp_path)
    cfg = _team(tmp_path)
    data = json.loads(export_run(cfg, fmt="json"))
    assert data["team"]["name"] == "test-team"
    assert data["team"]["goal"].strip() == "Build a widget."
    assert len(data["team"]["members"]) == 2


def test_export_json_turns_excludes_orchestrator(tmp_path: Path) -> None:
    _seed_workspace(tmp_path)
    cfg = _team(tmp_path)
    data = json.loads(export_run(cfg, fmt="json"))
    speakers = [t["speaker"] for t in data["turns"]]
    assert "orchestrator" not in speakers
    assert "lead" in speakers
    assert "eng" in speakers


def test_export_json_token_usage(tmp_path: Path) -> None:
    _seed_with_tokens(tmp_path)
    cfg = _team(tmp_path)
    data = json.loads(export_run(cfg, fmt="json"))
    # totals live in stats; per-member detail lives in token_usage
    assert data["stats"]["total_tokens"] == 270
    assert "lead" in data["token_usage"]
    assert data["token_usage"]["lead"]["prompt_tokens"] == 100


def test_export_json_artifacts_present(tmp_path: Path) -> None:
    _seed_workspace(tmp_path)
    cfg = _team(tmp_path)
    data = json.loads(export_run(cfg, fmt="json"))
    assert "src/widget.py" in data["artifacts"]


# --------------------------------------------------------------------------- #
# --no-artifacts flag
# --------------------------------------------------------------------------- #


def test_export_no_artifacts_markdown(tmp_path: Path) -> None:
    _seed_workspace(tmp_path)
    cfg = _team(tmp_path)
    full = export_run(cfg, fmt="markdown")
    no_art = export_run(cfg, fmt="markdown", no_artifacts=True)
    # Full export has the artifacts section; no_artifacts omits it
    assert "## Produced artifacts" in full
    assert "## Produced artifacts" not in no_art


def test_export_no_artifacts_json(tmp_path: Path) -> None:
    _seed_workspace(tmp_path)
    cfg = _team(tmp_path)
    data = json.loads(export_run(cfg, fmt="json", no_artifacts=True))
    assert data["artifacts"] == {}


# --------------------------------------------------------------------------- #
# HTML — dark mode + token section
# --------------------------------------------------------------------------- #


def test_export_html_dark_mode_css(tmp_path: Path) -> None:
    _seed_workspace(tmp_path)
    cfg = _team(tmp_path)
    html = export_run(cfg, fmt="html")
    assert "prefers-color-scheme: dark" in html


def test_export_html_contains_token_section(tmp_path: Path) -> None:
    _seed_with_tokens(tmp_path)
    cfg = _team(tmp_path)
    html = export_run(cfg, fmt="html")
    assert "Token usage" in html or "token_summary" in html or "Prompt" in html


def test_export_html_footer_version(tmp_path: Path) -> None:
    _seed_workspace(tmp_path)
    cfg = _team(tmp_path)
    html = export_run(cfg, fmt="html")
    assert "team-core" in html

