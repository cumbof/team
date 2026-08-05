"""Tests for context-management strategies in Member._apply_context_strategy.

Covers all four strategies that can be set via ``context_strategy`` in the
team YAML:

* ``none``          — full transcript, no trimming.
* ``sliding_window`` — last N turns (N = context_budget).
* ``truncate``       — drop oldest turns until within context_budget tokens.
* ``summarize``      — summarize old turns via LLM, keep recent verbatim.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from team.bus import Transcript, Turn
from team.config import Defaults, MemberConfig, TeamConfig, WorkflowConfig
from team.member import Member


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _make_team(
    tmp_path: Path,
    *,
    strategy: str = "none",
    budget: int = 0,
) -> TeamConfig:
    member_cfg = MemberConfig(
        name="alice",
        role="Researcher",
        model="llama3.1:8b",
        persona="You are a researcher.",
        context_strategy=strategy,
        context_budget=budget,
    )
    return TeamConfig(
        name="test-team",
        goal="Do science",
        workspace=tmp_path,
        workflow=WorkflowConfig(type="round_robin", max_rounds=1),
        defaults=Defaults(),
        members=[member_cfg],
    )


def _make_member(team: TeamConfig) -> Member:
    cfg = team.members[0]
    runtime = MagicMock()
    runtime.base_url = "http://localhost:11434"
    runtime.member = cfg
    with (
        patch("team.member.OllamaClient"),
        patch("team.member.OpenAICompatClient"),
        patch("team.member.render_system_prompt", return_value="system"),
    ):
        return Member(team, cfg, runtime)


def _make_transcript(*contents: str) -> Transcript:
    """Build an in-memory Transcript from a list of content strings."""
    t = Transcript()
    for i, content in enumerate(contents):
        t.turns.append(
            Turn(index=i, speaker="alice", role="Researcher", content=content)
        )
    return t


# --------------------------------------------------------------------------- #
# bus.Transcript.render — first_n parameter
# --------------------------------------------------------------------------- #


def test_render_first_n_returns_oldest_turns():
    t = _make_transcript("turn-0", "turn-1", "turn-2", "turn-3")
    result = t.render(first_n=2)
    assert "turn-0" in result
    assert "turn-1" in result
    assert "turn-2" not in result
    assert "turn-3" not in result


def test_render_first_n_no_op_when_fewer_turns():
    t = _make_transcript("only-turn")
    result = t.render(first_n=5)
    assert "only-turn" in result


def test_render_max_turns_takes_precedence_over_first_n():
    """When both are supplied, max_turns wins (most-recent slice)."""
    t = _make_transcript("old", "recent")
    result = t.render(max_turns=1, first_n=1)
    assert "recent" in result
    assert "old" not in result


# --------------------------------------------------------------------------- #
# strategy: none
# --------------------------------------------------------------------------- #


def test_strategy_none_returns_full_transcript(tmp_path):
    team = _make_team(tmp_path, strategy="none", budget=0)
    member = _make_member(team)
    transcript = _make_transcript("alpha", "beta", "gamma")
    result = member._apply_context_strategy(transcript)
    assert "alpha" in result
    assert "beta" in result
    assert "gamma" in result


def test_strategy_none_empty_transcript(tmp_path):
    team = _make_team(tmp_path, strategy="none", budget=0)
    member = _make_member(team)
    result = member._apply_context_strategy(_make_transcript())
    assert result == ""


# --------------------------------------------------------------------------- #
# strategy: sliding_window
# --------------------------------------------------------------------------- #


def test_sliding_window_keeps_last_n_turns(tmp_path):
    team = _make_team(tmp_path, strategy="sliding_window", budget=2)
    member = _make_member(team)
    transcript = _make_transcript("old-1", "old-2", "recent-1", "recent-2")
    result = member._apply_context_strategy(transcript)
    assert "recent-1" in result
    assert "recent-2" in result
    assert "old-1" not in result
    assert "old-2" not in result


def test_sliding_window_returns_all_when_fewer_than_budget(tmp_path):
    team = _make_team(tmp_path, strategy="sliding_window", budget=10)
    member = _make_member(team)
    transcript = _make_transcript("a", "b")
    result = member._apply_context_strategy(transcript)
    assert "a" in result
    assert "b" in result


def test_sliding_window_budget_zero_falls_back_to_full(tmp_path):
    """budget=0 disables the strategy — full transcript is returned."""
    team = _make_team(tmp_path, strategy="sliding_window", budget=0)
    member = _make_member(team)
    transcript = _make_transcript("x", "y", "z")
    result = member._apply_context_strategy(transcript)
    assert "x" in result


# --------------------------------------------------------------------------- #
# strategy: truncate
# --------------------------------------------------------------------------- #


def test_truncate_returns_full_when_fits(tmp_path):
    team = _make_team(tmp_path, strategy="truncate", budget=100_000)
    member = _make_member(team)
    transcript = _make_transcript("short")
    result = member._apply_context_strategy(transcript)
    assert "short" in result
    assert "omitted" not in result


def test_truncate_drops_old_turns_and_adds_note(tmp_path):
    # Each turn ~100 chars → 8 turns ≈ 800 chars / 4 ≈ 200 tokens.
    # Set budget=50 to force dropping.
    team = _make_team(tmp_path, strategy="truncate", budget=50)
    member = _make_member(team)
    turns = [f"turn content number {i} " + ("x" * 80) for i in range(8)]
    transcript = _make_transcript(*turns)
    result = member._apply_context_strategy(transcript)
    assert "omitted" in result.lower()
    # The most-recent turn should survive.
    assert "turn content number 7" in result


def test_truncate_budget_zero_falls_back_to_full(tmp_path):
    team = _make_team(tmp_path, strategy="truncate", budget=0)
    member = _make_member(team)
    transcript = _make_transcript("a", "b", "c")
    result = member._apply_context_strategy(transcript)
    assert "a" in result


# --------------------------------------------------------------------------- #
# strategy: summarize
# --------------------------------------------------------------------------- #


def test_summarize_returns_full_when_fits(tmp_path):
    """No LLM call when the transcript already fits in the budget."""
    team = _make_team(tmp_path, strategy="summarize", budget=100_000)
    member = _make_member(team)
    transcript = _make_transcript("short turn")
    # Patch client.chat to detect unexpected calls.
    member.client.chat = MagicMock(side_effect=AssertionError("LLM should not be called"))
    result = member._apply_context_strategy(transcript)
    assert "short turn" in result


def test_summarize_calls_llm_and_prepends_summary(tmp_path):
    """When over budget, old turns are summarized and recent turns kept."""
    team = _make_team(tmp_path, strategy="summarize", budget=50)
    member = _make_member(team)
    turns = [f"turn content number {i} " + ("x" * 80) for i in range(8)]
    transcript = _make_transcript(*turns)

    member.client.chat = MagicMock(return_value="• Key point from old turns")
    result = member._apply_context_strategy(transcript)

    member.client.chat.assert_called_once()
    assert "Key point from old turns" in result
    # The most-recent turn must appear verbatim.
    assert "turn content number 7" in result
    # The section headings must be present.
    assert "Summary of" in result
    assert "Recent conversation" in result


def test_summarize_llm_failure_falls_back_gracefully(tmp_path):
    """If the LLM call for summarization fails, a fallback note is used."""
    team = _make_team(tmp_path, strategy="summarize", budget=50)
    member = _make_member(team)
    turns = [f"turn content number {i} " + ("x" * 80) for i in range(8)]
    transcript = _make_transcript(*turns)

    member.client.chat = MagicMock(side_effect=RuntimeError("network error"))
    result = member._apply_context_strategy(transcript)

    # Fallback notice should mention the failure.
    assert "Summarization failed" in result or "summarization failed" in result.lower()
    # Recent turns should still be present.
    assert "turn content number 7" in result


def test_summarize_budget_zero_falls_back_to_full(tmp_path):
    team = _make_team(tmp_path, strategy="summarize", budget=0)
    member = _make_member(team)
    transcript = _make_transcript("a", "b", "c")
    member.client.chat = MagicMock(side_effect=AssertionError("no LLM call expected"))
    result = member._apply_context_strategy(transcript)
    assert "a" in result


def test_summarize_single_turn_over_budget(tmp_path):
    """Edge case: even a single turn exceeds budget — still no crash."""
    team = _make_team(tmp_path, strategy="summarize", budget=1)
    member = _make_member(team)
    transcript = _make_transcript("x" * 100)
    member.client.chat = MagicMock(return_value="• summary")
    # Should not raise.
    result = member._apply_context_strategy(transcript)
    assert result  # some non-empty text


def test_summarize_summary_section_precedes_recent_section(tmp_path):
    """The digest must appear before the recent-turns block."""
    team = _make_team(tmp_path, strategy="summarize", budget=50)
    member = _make_member(team)
    turns = [f"turn content number {i} " + ("x" * 80) for i in range(8)]
    transcript = _make_transcript(*turns)

    member.client.chat = MagicMock(return_value="• Digest bullet")
    result = member._apply_context_strategy(transcript)

    idx_summary = result.index("Summary of")
    idx_recent = result.index("Recent conversation")
    assert idx_summary < idx_recent
