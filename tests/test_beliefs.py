"""Tests for team.beliefs — shared team belief board."""

from __future__ import annotations

import json

import pytest

from team.beliefs import Belief, BeliefBoard


MEMBERS = ["alice", "bob", "carol"]


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def board(tmp_path):
    """A fresh BeliefBoard for a 3-member team with 0.5 threshold."""
    return BeliefBoard(
        path=tmp_path / "beliefs.json",
        member_names=MEMBERS,
        consensus_threshold=0.5,
    )


# --------------------------------------------------------------------------- #
# Initialisation
# --------------------------------------------------------------------------- #


def test_empty_board_count(board):
    assert board.count() == 0


def test_empty_list_beliefs(board):
    assert board.list_beliefs() == []


# --------------------------------------------------------------------------- #
# assert_belief
# --------------------------------------------------------------------------- #


def test_assert_belief_creates_entry(board):
    b = board.assert_belief("X is true", author="alice", confidence=0.8)
    assert b.id
    assert b.claim == "X is true"
    assert b.author == "alice"
    assert b.confidence == pytest.approx(0.8)
    assert "alice" in b.votes_for
    assert board.count() == 1


def test_assert_belief_status_pending_single_voter(board):
    """One vote out of 3 = 33% < 50% threshold → pending."""
    b = board.assert_belief("X", author="alice")
    assert b.status == "pending"


def test_assert_belief_auto_accepted_full_vote(tmp_path):
    """Single-member team: author's vote alone reaches 100% threshold."""
    board = BeliefBoard(tmp_path / "b.json", member_names=["solo"])
    b = board.assert_belief("X", author="solo")
    assert b.status == "accepted"


def test_assert_belief_persists(tmp_path):
    path = tmp_path / "b.json"
    board = BeliefBoard(path, member_names=MEMBERS)
    board.assert_belief("Y", author="bob")
    # Reload from disk.
    board2 = BeliefBoard(path, member_names=MEMBERS)
    assert board2.count() == 1
    assert board2.list_beliefs()[0].claim == "Y"


# --------------------------------------------------------------------------- #
# accept_belief
# --------------------------------------------------------------------------- #


def test_accept_increases_votes_for(board):
    b = board.assert_belief("X", author="alice")  # 1/3 votes
    b = board.accept_belief(b.id, voter="bob")      # 2/3 votes
    assert "bob" in b.votes_for
    assert len(b.votes_for) == 2


def test_accept_transitions_to_accepted(board):
    """2/3 votes ≥ 0.5 → accepted."""
    b = board.assert_belief("X", author="alice")
    b = board.accept_belief(b.id, voter="bob")
    assert b.status == "accepted"


def test_accept_removes_against_vote(board):
    b = board.assert_belief("X", author="alice")
    b = board.contest_belief(b.id, voter="bob")
    b = board.accept_belief(b.id, voter="bob")
    assert "bob" not in b.votes_against
    assert "bob" in b.votes_for


def test_accept_nonexistent_belief(board):
    with pytest.raises(KeyError):
        board.accept_belief("nope", voter="alice")


# --------------------------------------------------------------------------- #
# contest_belief
# --------------------------------------------------------------------------- #


def test_contest_moves_to_contested(board):
    b = board.assert_belief("X", author="alice")
    b = board.contest_belief(b.id, voter="bob", reason="insufficient data")
    assert b.status == "contested"
    assert "bob" in b.votes_against
    assert b.reasons["bob"] == "insufficient data"


def test_contest_removes_accept_vote(board):
    b = board.assert_belief("X", author="alice")
    b = board.accept_belief(b.id, voter="bob")
    b = board.contest_belief(b.id, voter="bob")
    assert "bob" not in b.votes_for
    assert "bob" in b.votes_against


def test_contest_nonexistent(board):
    with pytest.raises(KeyError):
        board.contest_belief("ghost", voter="carol")


# --------------------------------------------------------------------------- #
# reject_belief
# --------------------------------------------------------------------------- #


def test_reject_belief(board):
    b = board.assert_belief("X", author="alice")
    b = board.reject_belief(b.id)
    assert b.status == "rejected"


# --------------------------------------------------------------------------- #
# Consensus threshold
# --------------------------------------------------------------------------- #


def test_consensus_threshold_high(tmp_path):
    """With threshold=1.0, all 3 members must accept."""
    board = BeliefBoard(tmp_path / "b.json", member_names=MEMBERS, consensus_threshold=1.0)
    b = board.assert_belief("X", author="alice")
    board.accept_belief(b.id, voter="bob")
    b = board.accept_belief(b.id, voter="carol")
    assert b.status == "accepted"


def test_consensus_threshold_not_reached(tmp_path):
    """With threshold=0.9, 2/3 votes is not enough."""
    board = BeliefBoard(tmp_path / "b.json", member_names=MEMBERS, consensus_threshold=0.9)
    b = board.assert_belief("X", author="alice")
    b = board.accept_belief(b.id, voter="bob")
    assert b.status == "pending"


# --------------------------------------------------------------------------- #
# list_beliefs
# --------------------------------------------------------------------------- #


def test_list_beliefs_filter_by_status(board):
    b1 = board.assert_belief("X", author="alice")
    board.accept_belief(b1.id, voter="bob")  # now accepted (2/3)
    board.assert_belief("Y", author="carol")  # pending

    accepted = board.list_beliefs(status="accepted")
    pending = board.list_beliefs(status="pending")
    assert len(accepted) == 1
    assert len(pending) == 1


def test_list_beliefs_sorted_by_recency(board):
    b1 = board.assert_belief("First", author="alice")
    b2 = board.assert_belief("Second", author="bob")
    items = board.list_beliefs()
    assert items[0].id == b2.id  # most recent first


# --------------------------------------------------------------------------- #
# get_belief
# --------------------------------------------------------------------------- #


def test_get_belief_found(board):
    b = board.assert_belief("X", author="alice")
    found = board.get_belief(b.id)
    assert found is not None
    assert found.id == b.id


def test_get_belief_not_found(board):
    assert board.get_belief("ghost") is None


# --------------------------------------------------------------------------- #
# summary_for_prompt
# --------------------------------------------------------------------------- #


def test_summary_for_prompt_empty(board):
    assert board.summary_for_prompt() == ""


def test_summary_for_prompt_contains_claim(board):
    board.assert_belief("RNA is cool", author="alice")
    summary = board.summary_for_prompt()
    assert "RNA is cool" in summary
    assert "## Shared team belief board" in summary


def test_summary_for_prompt_limit(board):
    for i in range(5):
        board.assert_belief(f"claim {i}", author="alice")
    summary = board.summary_for_prompt(limit=3)
    assert summary.count("- [") == 3


# --------------------------------------------------------------------------- #
# Persistence round-trip
# --------------------------------------------------------------------------- #


def test_full_round_trip(tmp_path):
    path = tmp_path / "b.json"
    board = BeliefBoard(path, member_names=MEMBERS)
    b = board.assert_belief("A", author="alice", confidence=0.7, evidence="paper X")
    board.accept_belief(b.id, voter="bob")
    board.contest_belief(b.id, voter="carol", reason="too small n")

    # Reload
    board2 = BeliefBoard(path, member_names=MEMBERS)
    items = board2.list_beliefs()
    assert len(items) == 1
    loaded = items[0]
    assert loaded.claim == "A"
    assert loaded.confidence == pytest.approx(0.7)
    assert loaded.evidence == "paper X"
    assert loaded.status == "contested"
    assert "alice" in loaded.votes_for
    assert "bob" in loaded.votes_for
    assert "carol" in loaded.votes_against
    assert loaded.reasons["carol"] == "too small n"


# --------------------------------------------------------------------------- #
# Corrupt file recovery
# --------------------------------------------------------------------------- #


def test_corrupt_file_loads_empty(tmp_path):
    path = tmp_path / "b.json"
    path.write_text("{not valid json", encoding="utf-8")
    board = BeliefBoard(path, member_names=MEMBERS)
    assert board.count() == 0
