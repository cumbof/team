"""Tests for the bundled critique_request skill."""

from __future__ import annotations

from pathlib import Path

import pytest


def _import_tools(skills_dir: Path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "critique_request", skills_dir / "critique_request.py"
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


class TestRequestCritique:
    def test_posts_request_and_creates_queue_file(self, tools, ws):
        body = "from: alice\nfile: src/api.py\nquestion: Is error handling correct?"
        result = tools["request_critique"](body, workspace_path=ws)
        assert "CR-" in result
        assert (ws / "critique_queue.md").is_file()

    def test_question_appears_in_result(self, tools, ws):
        body = "from: bob\nfile: general\nquestion: Does the design scale?"
        result = tools["request_critique"](body, workspace_path=ws)
        assert "Does the design scale" in result

    def test_missing_question_returns_error(self, tools, ws):
        result = tools["request_critique"]("from: alice\nfile: x.py", workspace_path=ws)
        assert result.startswith("ERROR")

    def test_no_workspace_returns_error(self, tools):
        result = tools["request_critique"]("from: x\nquestion: ?")
        assert result.startswith("ERROR")

    def test_multiple_requests_accumulate(self, tools, ws):
        tools["request_critique"]("from: a\nfile: f\nquestion: Q1", workspace_path=ws)
        tools["request_critique"]("from: b\nfile: g\nquestion: Q2", workspace_path=ws)
        result = tools["list_critiques"]("", workspace_path=ws)
        assert "Q1" in result
        assert "Q2" in result


class TestPickCritique:
    def test_picks_oldest_request(self, tools, ws):
        tools["request_critique"]("from: a\nfile: f\nquestion: First", workspace_path=ws)
        tools["request_critique"]("from: b\nfile: g\nquestion: Second", workspace_path=ws)
        result = tools["pick_critique"]("", workspace_path=ws)
        assert "First" in result

    def test_marks_picked_as_claimed(self, tools, ws):
        tools["request_critique"]("from: a\nfile: f\nquestion: Q", workspace_path=ws)
        tools["pick_critique"]("", workspace_path=ws)
        # The picked request should no longer appear in list_critiques.
        result = tools["list_critiques"]("", workspace_path=ws)
        assert "No pending" in result

    def test_empty_queue_returns_no_pending(self, tools, ws):
        result = tools["pick_critique"]("", workspace_path=ws)
        assert "No pending" in result

    def test_no_workspace_returns_error(self, tools):
        result = tools["pick_critique"]("")
        assert result.startswith("ERROR")


class TestListCritiques:
    def test_lists_pending_only(self, tools, ws):
        tools["request_critique"]("from: a\nfile: f\nquestion: Open", workspace_path=ws)
        tools["request_critique"]("from: b\nfile: g\nquestion: Also open", workspace_path=ws)
        tools["pick_critique"]("", workspace_path=ws)  # claims first
        result = tools["list_critiques"]("", workspace_path=ws)
        assert "Also open" in result
        assert "Open" not in result or "Also open" in result  # first one claimed

    def test_empty_queue_returns_no_pending(self, tools, ws):
        result = tools["list_critiques"]("", workspace_path=ws)
        assert "No pending" in result

    def test_no_workspace_returns_error(self, tools):
        result = tools["list_critiques"]("")
        assert result.startswith("ERROR")
