"""Tests for team/tools.py (F4 — agent tool use)."""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from team.tools import (
    TOOL_DESCRIPTIONS,
    TOOLS,
    execute_tool,
    parse_tool_blocks,
)


# --------------------------------------------------------------------------- #
# parse_tool_blocks
# --------------------------------------------------------------------------- #


def test_parse_no_blocks():
    assert parse_tool_blocks("hello world") == []


def test_parse_single_block():
    text = "```tool:web_search\nquery: climate\n```"
    blocks = parse_tool_blocks(text)
    assert blocks == [("web_search", "query: climate")]


def test_parse_multiple_blocks():
    text = textwrap.dedent("""\
        Thinking...

        ```tool:web_search
        query: foo
        ```

        ```tool:run_bash
        echo hello
        ```
    """)
    blocks = parse_tool_blocks(text)
    assert len(blocks) == 2
    assert blocks[0] == ("web_search", "query: foo")
    assert blocks[1] == ("run_bash", "echo hello")


def test_parse_block_body_stripped():
    text = "```tool:run_python\n\nprint('hi')\n\n```"
    blocks = parse_tool_blocks(text)
    assert blocks[0][1] == "print('hi')"


# --------------------------------------------------------------------------- #
# TOOLS registry
# --------------------------------------------------------------------------- #


def test_all_tools_have_descriptions():
    for name in TOOLS:
        assert name in TOOL_DESCRIPTIONS


def test_execute_tool_unknown_raises():
    with pytest.raises(KeyError):
        execute_tool("nonexistent_tool", "body")


# --------------------------------------------------------------------------- #
# run_python
# --------------------------------------------------------------------------- #


def test_run_python_simple(tmp_path):
    result = execute_tool("run_python", "print('hello world')", workspace_path=tmp_path)
    assert "hello world" in result


def test_run_python_stderr(tmp_path):
    code = "import sys; sys.stderr.write('err line\\n')"
    result = execute_tool("run_python", code, workspace_path=tmp_path)
    assert "err line" in result


def test_run_python_syntax_error(tmp_path):
    result = execute_tool("run_python", "def (:", workspace_path=tmp_path)
    # Should return an error string, not raise.
    assert result.startswith("ERROR") or "SyntaxError" in result or "STDERR" in result


def test_run_python_timeout(tmp_path):
    result = execute_tool("run_python", "import time; time.sleep(99)", workspace_path=tmp_path, timeout=1)
    assert "timed out" in result.lower() or "ERROR" in result


# --------------------------------------------------------------------------- #
# run_bash
# --------------------------------------------------------------------------- #


def test_run_bash_simple(tmp_path):
    result = execute_tool("run_bash", "echo greetings", workspace_path=tmp_path)
    assert "greetings" in result


def test_run_bash_exit_code(tmp_path):
    result = execute_tool("run_bash", "exit 42", workspace_path=tmp_path)
    # Non-zero exit should return STDERR or be captured without raising.
    assert isinstance(result, str)


def test_run_bash_timeout(tmp_path):
    result = execute_tool("run_bash", "sleep 99", workspace_path=tmp_path, timeout=1)
    assert "timed out" in result.lower() or "ERROR" in result


# --------------------------------------------------------------------------- #
# read_file
# --------------------------------------------------------------------------- #


def test_read_file_success(tmp_path):
    f = tmp_path / "hello.txt"
    f.write_text("file contents here", encoding="utf-8")
    result = execute_tool("read_file", "path: hello.txt", workspace_path=tmp_path)
    assert "file contents here" in result


def test_read_file_missing(tmp_path):
    result = execute_tool("read_file", "path: does_not_exist.txt", workspace_path=tmp_path)
    assert result.startswith("ERROR")


def test_read_file_path_traversal(tmp_path):
    result = execute_tool("read_file", "path: ../../etc/passwd", workspace_path=tmp_path)
    assert result.startswith("ERROR")


def test_read_file_no_workspace():
    result = execute_tool("read_file", "path: foo.txt", workspace_path=None)
    assert "no workspace" in result.lower() or result.startswith("ERROR")


# --------------------------------------------------------------------------- #
# web_search (mocked)
# --------------------------------------------------------------------------- #


def test_web_search_success():
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "Heading": "Python",
        "AbstractText": "A programming language.",
        "Answer": "",
        "RelatedTopics": [],
    }
    mock_response.raise_for_status = MagicMock()
    with patch("team.tools.requests.get", return_value=mock_response):
        result = execute_tool("web_search", "query: python programming")
    assert "Python" in result
    assert "programming language" in result


def test_web_search_empty_result():
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "Heading": "",
        "AbstractText": "",
        "Answer": "",
        "RelatedTopics": [],
    }
    mock_response.raise_for_status = MagicMock()
    with patch("team.tools.requests.get", return_value=mock_response):
        result = execute_tool("web_search", "query: xyzzy nonsense")
    assert "No instant answer" in result or isinstance(result, str)


def test_web_search_network_error():
    with patch("team.tools.requests.get", side_effect=ConnectionError("no network")):
        result = execute_tool("web_search", "query: test")
    assert result.startswith("ERROR")


# --------------------------------------------------------------------------- #
# read_url (mocked)
# --------------------------------------------------------------------------- #


def test_read_url_html():
    mock_response = MagicMock()
    mock_response.text = "<html><body><p>Hello world</p><script>js()</script></body></html>"
    mock_response.headers = {"Content-Type": "text/html"}
    mock_response.raise_for_status = MagicMock()
    with patch("team.tools.requests.get", return_value=mock_response):
        result = execute_tool("read_url", "url: http://example.com")
    assert "Hello world" in result
    # Scripts should be stripped.
    assert "js()" not in result


def test_read_url_network_error():
    with patch("team.tools.requests.get", side_effect=ConnectionError("no network")):
        result = execute_tool("read_url", "url: http://example.com")
    assert result.startswith("ERROR")


def test_read_url_missing_url():
    result = execute_tool("read_url", "")
    assert result.startswith("ERROR")


# --------------------------------------------------------------------------- #
# write_file
# --------------------------------------------------------------------------- #


def test_write_file_creates_new(tmp_path):
    body = "path: hello.txt\n---\nHello, world!\n"
    result = execute_tool("write_file", body, workspace_path=tmp_path)
    assert "wrote" in result
    assert (tmp_path / "hello.txt").read_text() == "Hello, world!\n"


def test_write_file_overwrites_existing(tmp_path):
    (tmp_path / "existing.txt").write_text("old content", encoding="utf-8")
    body = "path: existing.txt\n---\nnew content\n"
    result = execute_tool("write_file", body, workspace_path=tmp_path)
    assert "wrote" in result
    assert (tmp_path / "existing.txt").read_text() == "new content\n"


def test_write_file_creates_subdirectory(tmp_path):
    body = "path: sub/dir/note.md\n---\n# Note\n"
    result = execute_tool("write_file", body, workspace_path=tmp_path)
    assert "wrote" in result
    assert (tmp_path / "sub" / "dir" / "note.md").is_file()


def test_write_file_missing_path(tmp_path):
    body = "\n---\ncontent only"
    result = execute_tool("write_file", body, workspace_path=tmp_path)
    assert result.startswith("ERROR")


def test_write_file_missing_separator(tmp_path):
    body = "path: test.txt\nsome content with no separator"
    result = execute_tool("write_file", body, workspace_path=tmp_path)
    assert result.startswith("ERROR")


def test_write_file_no_workspace():
    body = "path: file.txt\n---\ncontent"
    result = execute_tool("write_file", body, workspace_path=None)
    assert "no workspace" in result.lower() or result.startswith("ERROR")


def test_write_file_path_traversal(tmp_path):
    body = "path: ../../etc/passwd\n---\nevil"
    result = execute_tool("write_file", body, workspace_path=tmp_path)
    assert result.startswith("ERROR")


# --------------------------------------------------------------------------- #
# append_file
# --------------------------------------------------------------------------- #


def test_append_file_creates_new(tmp_path):
    body = "path: log.txt\n---\nfirst line\n"
    result = execute_tool("append_file", body, workspace_path=tmp_path)
    assert "appended" in result
    assert (tmp_path / "log.txt").read_text() == "first line\n"


def test_append_file_appends_to_existing(tmp_path):
    (tmp_path / "log.txt").write_text("line one\n", encoding="utf-8")
    body = "path: log.txt\n---\nline two\n"
    result = execute_tool("append_file", body, workspace_path=tmp_path)
    assert "appended" in result
    assert (tmp_path / "log.txt").read_text() == "line one\nline two\n"


def test_append_file_no_workspace():
    body = "path: file.txt\n---\ncontent"
    result = execute_tool("append_file", body, workspace_path=None)
    assert "no workspace" in result.lower() or result.startswith("ERROR")


def test_append_file_path_traversal(tmp_path):
    body = "path: ../../etc/passwd\n---\nevil"
    result = execute_tool("append_file", body, workspace_path=tmp_path)
    assert result.startswith("ERROR")


def test_append_file_missing_separator(tmp_path):
    body = "path: test.txt"
    result = execute_tool("append_file", body, workspace_path=tmp_path)
    assert result.startswith("ERROR")


# --------------------------------------------------------------------------- #
# list_files
# --------------------------------------------------------------------------- #


def test_list_files_empty_workspace(tmp_path):
    result = execute_tool("list_files", "", workspace_path=tmp_path)
    assert "empty" in result.lower() or "no files" in result.lower()


def test_list_files_all(tmp_path):
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "b.py").write_text("b", encoding="utf-8")
    result = execute_tool("list_files", "", workspace_path=tmp_path)
    assert "a.txt" in result
    assert "b.py" in result


def test_list_files_with_glob_pattern(tmp_path):
    (tmp_path / "main.py").write_text("x", encoding="utf-8")
    (tmp_path / "README.md").write_text("y", encoding="utf-8")
    result = execute_tool("list_files", "pattern: *.py", workspace_path=tmp_path)
    assert "main.py" in result
    assert "README.md" not in result


def test_list_files_no_match(tmp_path):
    (tmp_path / "data.csv").write_text("1,2,3", encoding="utf-8")
    result = execute_tool("list_files", "pattern: *.py", workspace_path=tmp_path)
    assert "no files" in result.lower() or "match" in result.lower()


def test_list_files_no_workspace():
    result = execute_tool("list_files", "", workspace_path=None)
    assert result.startswith("ERROR")


# --------------------------------------------------------------------------- #
# memory tools (remember, recall, forget, list_memories)
# --------------------------------------------------------------------------- #


from team.memory import AgentMemory


@pytest.fixture
def mem(tmp_path):
    return AgentMemory(tmp_path / "m.db")


def test_remember_tool_basic(tmp_path, mem):
    body = "key: exp_baseline\ntags: results\nimportance: 0.8\n---\nAlphaFold beats rosettafold"
    result = execute_tool("remember", body, memory=mem)
    assert "exp_baseline" in result
    assert mem.count() == 1


def test_remember_tool_no_memory():
    body = "key: k\n---\nv"
    result = execute_tool("remember", body, memory=None)
    assert result.startswith("ERROR")


def test_remember_tool_missing_key():
    result = execute_tool("remember", "\n---\nsome value", memory=MagicMock())
    assert result.startswith("ERROR")


def test_remember_tool_missing_separator(mem):
    result = execute_tool("remember", "key: k\nvalue: v", memory=mem)
    assert result.startswith("ERROR")


def test_recall_tool_found(tmp_path, mem):
    mem.remember("protein_folding", "AlphaFold analysis")
    result = execute_tool("recall", "query: protein", memory=mem)
    assert "protein_folding" in result
    assert "AlphaFold" in result


def test_recall_tool_not_found(mem):
    result = execute_tool("recall", "query: xyz_nothing", memory=mem)
    assert "No memories" in result or "not found" in result.lower()


def test_recall_tool_no_memory():
    result = execute_tool("recall", "query: anything", memory=None)
    assert result.startswith("ERROR")


def test_forget_tool_existing(mem):
    mem.remember("k", "v")
    result = execute_tool("forget", "key: k", memory=mem)
    assert "Deleted" in result
    assert mem.count() == 0


def test_forget_tool_nonexistent(mem):
    result = execute_tool("forget", "key: ghost", memory=mem)
    assert "No memory" in result


def test_list_memories_tool_empty(mem):
    result = execute_tool("list_memories", "", memory=mem)
    assert "No memories" in result or "stored yet" in result.lower()


def test_list_memories_tool_all(mem):
    mem.remember("a", "va")
    mem.remember("b", "vb")
    result = execute_tool("list_memories", "", memory=mem)
    assert "a" in result and "b" in result


def test_list_memories_tool_by_tag(mem):
    mem.remember("key_science", "va", tags=["science"])
    mem.remember("key_bio", "vb", tags=["bio"])
    result = execute_tool("list_memories", "tag: science", memory=mem)
    assert "key_science" in result
    assert "key_bio" not in result


# --------------------------------------------------------------------------- #
# belief-board tools
# --------------------------------------------------------------------------- #


from team.beliefs import BeliefBoard

MEMBERS_3 = ["alice", "bob", "carol"]


@pytest.fixture
def bboard(tmp_path):
    return BeliefBoard(tmp_path / "b.json", member_names=MEMBERS_3, consensus_threshold=0.5)


def test_assert_belief_tool_basic(bboard):
    body = "confidence: 0.8\n---\nRNA polymerase is key"
    result = execute_tool("assert_belief", body, beliefs=bboard, member_name="alice")
    assert "asserted" in result.lower() or "belief" in result.lower()
    assert bboard.count() == 1


def test_assert_belief_tool_no_beliefs():
    body = "confidence: 0.5\n---\nclaim"
    result = execute_tool("assert_belief", body, beliefs=None, member_name="alice")
    assert result.startswith("ERROR")


def test_assert_belief_tool_missing_separator(bboard):
    result = execute_tool("assert_belief", "confidence: 0.5\nclaim text no sep", beliefs=bboard, member_name="alice")
    assert result.startswith("ERROR")


def test_accept_belief_tool(bboard):
    b = bboard.assert_belief("X is true", author="alice")
    result = execute_tool("accept_belief", f"id: {b.id}", beliefs=bboard, member_name="bob")
    assert "accept" in result.lower()
    b2 = bboard.get_belief(b.id)
    assert "bob" in b2.votes_for


def test_accept_belief_tool_unknown_id(bboard):
    result = execute_tool("accept_belief", "id: xxxxxxxx", beliefs=bboard, member_name="alice")
    assert result.startswith("ERROR")


def test_contest_belief_tool(bboard):
    b = bboard.assert_belief("X", author="alice")
    body = f"id: {b.id}\nreason: insufficient data"
    result = execute_tool("contest_belief", body, beliefs=bboard, member_name="bob")
    assert "contested" in result.lower()
    b2 = bboard.get_belief(b.id)
    assert b2.status == "contested"


def test_list_beliefs_tool_empty(bboard):
    result = execute_tool("list_beliefs", "", beliefs=bboard)
    assert "empty" in result.lower() or "No belief" in result


def test_list_beliefs_tool_all(bboard):
    bboard.assert_belief("A", author="alice")
    bboard.assert_belief("B", author="bob")
    result = execute_tool("list_beliefs", "", beliefs=bboard)
    assert "A" in result and "B" in result


def test_list_beliefs_tool_status_filter(bboard):
    b = bboard.assert_belief("X", author="alice")
    bboard.accept_belief(b.id, voter="bob")  # now accepted (2/3 >= 0.5)
    bboard.assert_belief("Y", author="carol")  # pending

    result = execute_tool("list_beliefs", "status: accepted", beliefs=bboard)
    assert "X" in result
    assert "Y" not in result


def test_list_beliefs_tool_invalid_status(bboard):
    result = execute_tool("list_beliefs", "status: unknown_status", beliefs=bboard)
    assert result.startswith("ERROR")


# --------------------------------------------------------------------------- #
# execute_tool passes memory + beliefs kwargs
# --------------------------------------------------------------------------- #


def test_execute_tool_forwards_memory_and_beliefs(mem, bboard):
    """Verify execute_tool properly forwards memory/beliefs/member_name."""
    # Use a tool that requires memory to be non-None to return a non-error result.
    mem.remember("k", "v")
    result = execute_tool("recall", "query: k", memory=mem, beliefs=bboard, member_name="alice")
    assert not result.startswith("ERROR")


# --------------------------------------------------------------------------- #
# log_decision / read_decisions
# --------------------------------------------------------------------------- #


def test_log_decision_creates_file(tmp_path):
    result = execute_tool(
        "log_decision",
        "title: Use pandas\nrationale: Mature ecosystem\nalternatives: polars, dask",
        workspace_path=tmp_path,
        member_name="alice",
    )
    assert "logged" in result.lower() or "pandas" in result.lower()
    decisions_file = tmp_path / "decisions.md"
    assert decisions_file.is_file()
    content = decisions_file.read_text()
    assert "Use pandas" in content
    assert "@alice" in content


def test_log_decision_title_required(tmp_path):
    result = execute_tool("log_decision", "", workspace_path=tmp_path)
    assert result.startswith("ERROR")


def test_log_decision_no_workspace():
    result = execute_tool("log_decision", "title: foo", workspace_path=None)
    assert result.startswith("ERROR")


def test_log_decision_appends_multiple_entries(tmp_path):
    execute_tool("log_decision", "title: First", workspace_path=tmp_path, member_name="a")
    execute_tool("log_decision", "title: Second", workspace_path=tmp_path, member_name="b")
    content = (tmp_path / "decisions.md").read_text()
    assert "First" in content
    assert "Second" in content


def test_log_decision_records_rationale_and_alternatives(tmp_path):
    execute_tool(
        "log_decision",
        "title: T\nrationale: Because reasons\nalternatives: option_a, option_b",
        workspace_path=tmp_path,
    )
    content = (tmp_path / "decisions.md").read_text()
    assert "Because reasons" in content
    assert "option_a" in content


def test_read_decisions_empty(tmp_path):
    result = execute_tool("read_decisions", "", workspace_path=tmp_path)
    assert "no decisions" in result.lower() or "does not exist" in result.lower()


def test_read_decisions_returns_content(tmp_path):
    execute_tool("log_decision", "title: MyDecision", workspace_path=tmp_path)
    result = execute_tool("read_decisions", "", workspace_path=tmp_path)
    assert "MyDecision" in result


def test_read_decisions_no_workspace():
    result = execute_tool("read_decisions", "", workspace_path=None)
    assert result.startswith("ERROR")


def test_log_and_read_decisions_round_trip(tmp_path):
    execute_tool(
        "log_decision",
        "title: Alpha\nrationale: It works\nalternatives: beta",
        workspace_path=tmp_path,
        member_name="tester",
    )
    result = execute_tool("read_decisions", "", workspace_path=tmp_path)
    assert "Alpha" in result
    assert "It works" in result
    assert "@tester" in result


def test_log_decision_in_tool_registry():
    assert "log_decision" in TOOLS
    assert "read_decisions" in TOOLS
    assert "log_decision" in TOOL_DESCRIPTIONS
    assert "read_decisions" in TOOL_DESCRIPTIONS


# --------------------------------------------------------------------------- #
# delegate_to_expert
# --------------------------------------------------------------------------- #


import os


def test_delegate_to_expert_in_registry():
    assert "delegate_to_expert" in TOOLS
    assert "delegate_to_expert" in TOOL_DESCRIPTIONS


def test_delegate_to_expert_missing_provider():
    result = execute_tool("delegate_to_expert", "prompt: hello")
    assert result.startswith("ERROR")
    assert "provider" in result.lower()


def test_delegate_to_expert_unknown_provider():
    result = execute_tool("delegate_to_expert", "provider: fakeai\nprompt: hello")
    assert result.startswith("ERROR")
    assert "fakeai" in result


def test_delegate_to_expert_missing_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = execute_tool("delegate_to_expert", "provider: openai\nprompt: hello")
    assert result.startswith("ERROR")
    assert "OPENAI_API_KEY" in result


def test_delegate_to_expert_missing_prompt(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    result = execute_tool("delegate_to_expert", "provider: openai")
    assert result.startswith("ERROR")
    assert "prompt" in result.lower()


def test_delegate_to_expert_openai_success(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Paris is the capital of France."}}]
    }
    mock_response.raise_for_status = MagicMock()
    with patch("team.tools.requests.post", return_value=mock_response):
        result = execute_tool(
            "delegate_to_expert",
            "provider: openai\nmodel: gpt-4o\nprompt: What is the capital of France?",
        )
    assert "Paris" in result
    assert "openai" in result.lower()


def test_delegate_to_expert_anthropic_success(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "content": [{"type": "text", "text": "The answer is 42."}]
    }
    mock_response.raise_for_status = MagicMock()
    with patch("team.tools.requests.post", return_value=mock_response):
        result = execute_tool(
            "delegate_to_expert",
            "provider: anthropic\nprompt: What is the answer?",
        )
    assert "42" in result
    assert "anthropic" in result.lower()


def test_delegate_to_expert_google_success(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "candidates": [
            {"content": {"parts": [{"text": "Quantum entanglement explained."}]}}
        ]
    }
    mock_response.raise_for_status = MagicMock()
    with patch("team.tools.requests.post", return_value=mock_response):
        result = execute_tool(
            "delegate_to_expert",
            "provider: google\nprompt: Explain quantum entanglement.",
        )
    assert "Quantum" in result
    assert "google" in result.lower()


def test_delegate_to_expert_multiline_prompt(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Sure, here are three analogies."}}]
    }
    mock_response.raise_for_status = MagicMock()

    body = (
        "provider: openai\n"
        "model: gpt-4o\n"
        "max_tokens: 1024\n"
        "temperature: 0.5\n"
        "---\n"
        "Explain quantum entanglement.\n"
        "Give at least three analogies."
    )
    with patch("team.tools.requests.post", return_value=mock_response) as mock_post:
        result = execute_tool("delegate_to_expert", body)

    assert "analogies" in result.lower()
    # Verify the full multi-line prompt was sent to the API.
    call_kwargs = mock_post.call_args
    payload = call_kwargs[1]["json"] if call_kwargs[1] else call_kwargs[0][1]
    sent_prompt = payload["messages"][0]["content"]
    assert "three analogies" in sent_prompt


def test_delegate_to_expert_network_error(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    with patch("team.tools.requests.post", side_effect=ConnectionError("no route")):
        result = execute_tool("delegate_to_expert", "provider: openai\nprompt: hello")
    assert result.startswith("ERROR")


def test_delegate_to_expert_http_error(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "bad-key")
    mock_response = MagicMock()
    mock_response.status_code = 401
    http_err = __import__("requests").exceptions.HTTPError(response=mock_response)
    mock_response.raise_for_status.side_effect = http_err
    with patch("team.tools.requests.post", return_value=mock_response):
        result = execute_tool("delegate_to_expert", "provider: anthropic\nprompt: hello")
    assert result.startswith("ERROR")
    assert "401" in result


def test_delegate_to_expert_invalid_max_tokens(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    result = execute_tool(
        "delegate_to_expert",
        "provider: openai\nmax_tokens: not-a-number\nprompt: hello",
    )
    assert result.startswith("ERROR")
    assert "max_tokens" in result


def test_delegate_to_expert_invalid_temperature(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    result = execute_tool(
        "delegate_to_expert",
        "provider: openai\ntemperature: hot\nprompt: hello",
    )
    assert result.startswith("ERROR")
    assert "temperature" in result


def test_delegate_to_expert_openai_empty_choices(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    mock_response = MagicMock()
    mock_response.json.return_value = {"choices": []}
    mock_response.raise_for_status = MagicMock()
    with patch("team.tools.requests.post", return_value=mock_response):
        result = execute_tool("delegate_to_expert", "provider: openai\nprompt: hello")
    assert result.startswith("ERROR")


def test_delegate_to_expert_google_no_candidates(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    mock_response = MagicMock()
    mock_response.json.return_value = {"candidates": []}
    mock_response.raise_for_status = MagicMock()
    with patch("team.tools.requests.post", return_value=mock_response):
        result = execute_tool("delegate_to_expert", "provider: google\nprompt: hello")
    assert result.startswith("ERROR")
