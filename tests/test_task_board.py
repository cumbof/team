"""Tests for the bundled task_board skill."""

from __future__ import annotations

from pathlib import Path

import pytest


def _import_tools(skills_dir: Path):
    """Load the task_board.py skill and return its tool functions."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "task_board", skills_dir / "task_board.py"
    )
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod.TOOLS


_SKILLS_DIR = Path(__file__).parent.parent / "skills"


@pytest.fixture
def tools():
    return _import_tools(_SKILLS_DIR)


@pytest.fixture
def ws(tmp_path):
    return tmp_path


class TestTaskAdd:
    def test_adds_task_to_board(self, tools, ws):
        result = tools["task_add"]("Design the schema", workspace_path=ws)
        assert "Design the schema" in result
        assert (ws / "TASKS.md").is_file()

    def test_creates_board_file_if_absent(self, tools, ws):
        tools["task_add"]("First task", workspace_path=ws)
        content = (ws / "TASKS.md").read_text()
        assert "First task" in content
        assert "[ ]" in content

    def test_duplicate_task_not_added(self, tools, ws):
        tools["task_add"]("Same task", workspace_path=ws)
        result = tools["task_add"]("Same task", workspace_path=ws)
        assert "already exists" in result

    def test_empty_body_returns_error(self, tools, ws):
        result = tools["task_add"]("", workspace_path=ws)
        assert result.startswith("ERROR")

    def test_no_workspace_returns_error(self, tools):
        result = tools["task_add"]("task")
        assert result.startswith("ERROR")


class TestTaskDone:
    def test_marks_task_done(self, tools, ws):
        tools["task_add"]("Write tests", workspace_path=ws)
        result = tools["task_done"]("Write tests", workspace_path=ws)
        assert "Marked done" in result
        content = (ws / "TASKS.md").read_text()
        assert "[x] Write tests" in content
        assert "[ ] Write tests" not in content

    def test_partial_match_works(self, tools, ws):
        tools["task_add"]("Implement the API endpoints", workspace_path=ws)
        result = tools["task_done"]("API endpoints", workspace_path=ws)
        assert "Marked done" in result

    def test_no_match_returns_error(self, tools, ws):
        tools["task_add"]("Real task", workspace_path=ws)
        result = tools["task_done"]("nonexistent task", workspace_path=ws)
        assert "No pending" in result

    def test_ambiguous_match_returns_error(self, tools, ws):
        tools["task_add"]("Test module A", workspace_path=ws)
        tools["task_add"]("Test module B", workspace_path=ws)
        result = tools["task_done"]("Test module", workspace_path=ws)
        assert "Ambiguous" in result

    def test_empty_body_returns_error(self, tools, ws):
        result = tools["task_done"]("", workspace_path=ws)
        assert result.startswith("ERROR")


class TestTaskList:
    def test_lists_pending_and_done(self, tools, ws):
        tools["task_add"]("Task A", workspace_path=ws)
        tools["task_add"]("Task B", workspace_path=ws)
        tools["task_done"]("Task B", workspace_path=ws)
        result = tools["task_list"]("", workspace_path=ws)
        assert "Task A" in result
        assert "Task B" in result
        assert "Pending (1)" in result
        assert "Done (1)" in result

    def test_empty_board(self, tools, ws):
        result = tools["task_list"]("", workspace_path=ws)
        assert "empty" in result.lower() or "Pending (0)" in result

    def test_no_workspace_returns_error(self, tools):
        result = tools["task_list"]("")
        assert result.startswith("ERROR")


class TestPersistence:
    def test_board_persists_across_calls(self, tools, ws):
        tools["task_add"]("Persistent task", workspace_path=ws)
        # Re-import to simulate a fresh skill call reading from disk.
        tools2 = _import_tools(_SKILLS_DIR)
        result = tools2["task_list"]("", workspace_path=ws)
        assert "Persistent task" in result
