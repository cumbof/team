"""Tests for team.memory — per-agent persistent cross-session memory store."""

from __future__ import annotations

import time

import pytest

from team.memory import AgentMemory


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def mem(tmp_path):
    """A fresh AgentMemory backed by a temp SQLite db."""
    return AgentMemory(tmp_path / "test_member.db")


# --------------------------------------------------------------------------- #
# Initialisation
# --------------------------------------------------------------------------- #


def test_init_creates_db_file(tmp_path):
    db = tmp_path / "sub" / "mem.db"
    AgentMemory(db)
    assert db.exists()


def test_empty_store_count(mem):
    assert mem.count() == 0


# --------------------------------------------------------------------------- #
# remember
# --------------------------------------------------------------------------- #


def test_remember_returns_confirmation(mem):
    result = mem.remember("k1", "v1")
    assert "k1" in result


def test_remember_stores_value(mem):
    mem.remember("key", "hello world")
    entries = mem.list_memories()
    assert len(entries) == 1
    assert entries[0]["key"] == "key"
    assert entries[0]["value"] == "hello world"


def test_remember_with_tags(mem):
    mem.remember("k", "v", tags=["science", "bio"])
    entries = mem.list_memories()
    assert "science" in entries[0]["tags"]
    assert "bio" in entries[0]["tags"]


def test_remember_with_importance(mem):
    mem.remember("k", "v", importance=0.8)
    entries = mem.list_memories()
    assert entries[0]["importance"] == pytest.approx(0.8)


def test_remember_updates_existing(mem):
    mem.remember("k", "original")
    result = mem.remember("k", "updated")
    assert "Updated" in result
    assert mem.count() == 1
    entries = mem.list_memories()
    assert entries[0]["value"] == "updated"


def test_remember_multiple_entries(mem):
    for i in range(5):
        mem.remember(f"key_{i}", f"value_{i}")
    assert mem.count() == 5


# --------------------------------------------------------------------------- #
# recall
# --------------------------------------------------------------------------- #


def test_recall_by_key_substring(mem):
    mem.remember("protein_folding", "AlphaFold beats RoseTTAFold")
    mem.remember("rna_sequencing", "RNA-seq protocol v2")
    results = mem.recall("protein")
    assert len(results) == 1
    assert results[0]["key"] == "protein_folding"


def test_recall_by_value_substring(mem):
    mem.remember("method_a", "uses gradient descent")
    mem.remember("method_b", "uses genetic algorithm")
    results = mem.recall("gradient")
    assert len(results) == 1
    assert results[0]["key"] == "method_a"


def test_recall_no_results(mem):
    mem.remember("key", "value")
    results = mem.recall("xyz_nonexistent")
    assert results == []


def test_recall_limit(mem):
    for i in range(10):
        mem.remember(f"key_{i}", "common value")
    results = mem.recall("common", limit=3)
    assert len(results) == 3


def test_recall_sorted_by_importance(mem):
    mem.remember("low", "common term", importance=0.1)
    mem.remember("high", "common term", importance=0.9)
    results = mem.recall("common", limit=10)
    assert results[0]["key"] == "high"


# --------------------------------------------------------------------------- #
# forget
# --------------------------------------------------------------------------- #


def test_forget_existing(mem):
    mem.remember("k", "v")
    deleted = mem.forget("k")
    assert deleted is True
    assert mem.count() == 0


def test_forget_nonexistent(mem):
    deleted = mem.forget("ghost_key")
    assert deleted is False


# --------------------------------------------------------------------------- #
# list_memories
# --------------------------------------------------------------------------- #


def test_list_memories_all(mem):
    mem.remember("a", "v1")
    mem.remember("b", "v2")
    entries = mem.list_memories()
    assert len(entries) == 2


def test_list_memories_tag_filter(mem):
    mem.remember("x", "vx", tags=["physics"])
    mem.remember("y", "vy", tags=["chemistry"])
    mem.remember("z", "vz", tags=["physics", "chemistry"])
    phys = mem.list_memories(tag="physics")
    assert len(phys) == 2
    assert all("physics" in e["tags"] for e in phys)


def test_list_memories_tag_filter_empty(mem):
    mem.remember("k", "v", tags=["bio"])
    result = mem.list_memories(tag="physics")
    assert result == []


def test_list_memories_limit(mem):
    for i in range(10):
        mem.remember(f"k{i}", "v")
    entries = mem.list_memories(limit=4)
    assert len(entries) == 4


# --------------------------------------------------------------------------- #
# recent
# --------------------------------------------------------------------------- #


def test_recent_returns_n(mem):
    for i in range(6):
        mem.remember(f"k{i}", "v")
        time.sleep(0.01)  # ensure distinct timestamps
    recent = mem.recent(n=3)
    assert len(recent) == 3
    # Most recently updated key is k5.
    assert recent[0]["key"] == "k5"


def test_recent_empty(mem):
    assert mem.recent() == []


# --------------------------------------------------------------------------- #
# summary_for_prompt
# --------------------------------------------------------------------------- #


def test_summary_for_prompt_empty(mem):
    assert mem.summary_for_prompt() == ""


def test_summary_for_prompt_has_header(mem):
    mem.remember("alpha", "result A", tags=["results"])
    summary = mem.summary_for_prompt(n=5)
    assert "## Your persistent memories" in summary
    assert "alpha" in summary
    assert "result A" in summary


def test_summary_for_prompt_respects_n(mem):
    for i in range(10):
        mem.remember(f"k{i}", "v")
    summary = mem.summary_for_prompt(n=3)
    # Should list 3 items — count bullet points.
    bullet_count = summary.count("\n- ")
    assert bullet_count == 3


# --------------------------------------------------------------------------- #
# Multi-member isolation
# --------------------------------------------------------------------------- #


def test_separate_dbs_are_isolated(tmp_path):
    mem_alice = AgentMemory(tmp_path / "alice.db")
    mem_bob = AgentMemory(tmp_path / "bob.db")
    mem_alice.remember("secret", "alice's data")
    assert mem_bob.recall("alice") == []
    assert mem_alice.count() == 1
    assert mem_bob.count() == 0
