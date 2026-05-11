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

