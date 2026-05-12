"""Tests for the bundled progress_snapshot skill."""

from __future__ import annotations

from pathlib import Path

import pytest


def _import_execute(skills_dir: Path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "progress_snapshot", skills_dir / "progress_snapshot.py"
    )
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod.execute


_SKILLS_DIR = Path(__file__).parent.parent / "skills"


@pytest.fixture
def execute():
    return _import_execute(_SKILLS_DIR)


@pytest.fixture
def ws(tmp_path):
    return tmp_path


class TestWriteSnapshot:
    def test_creates_progress_file(self, execute, ws):
        execute("## Done\n- Task A\n", workspace_path=ws)
        assert (ws / "PROGRESS.md").is_file()

    def test_content_appears_in_file(self, execute, ws):
        execute("## Done\n- Implemented schema\n", workspace_path=ws)
        content = (ws / "PROGRESS.md").read_text()
        assert "Implemented schema" in content

    def test_file_includes_timestamp(self, execute, ws):
        execute("Some progress.", workspace_path=ws)
        content = (ws / "PROGRESS.md").read_text()
        assert "Last updated" in content

    def test_overwrites_previous_snapshot(self, execute, ws):
        execute("## Done\n- Old task\n", workspace_path=ws)
        execute("## Done\n- New task\n", workspace_path=ws)
        content = (ws / "PROGRESS.md").read_text()
        assert "New task" in content
        assert "Old task" not in content

    def test_return_value_mentions_file(self, execute, ws):
        result = execute("Progress here.", workspace_path=ws)
        assert "PROGRESS.md" in result

    def test_no_workspace_returns_error(self, execute):
        result = execute("Some progress.")
        assert result.startswith("ERROR")


class TestReadSnapshot:
    def test_empty_body_reads_existing_snapshot(self, execute, ws):
        execute("## Done\n- X\n", workspace_path=ws)
        result = execute("", workspace_path=ws)
        assert "Done" in result
        assert "X" in result

    def test_empty_body_no_file_returns_message(self, execute, ws):
        result = execute("", workspace_path=ws)
        assert "No progress snapshot" in result

    def test_whitespace_body_treated_as_empty(self, execute, ws):
        result = execute("   \n  ", workspace_path=ws)
        assert "No progress snapshot" in result
