"""Tests for team/mcp/textmode.py — text-mode tool block parsing + rendering."""

from __future__ import annotations

import textwrap

from team.mcp.bus import ToolInfo
from team.mcp.textmode import (
    arg_summary,
    parse_tool_blocks,
    parse_tool_body,
    render_tool_protocol,
)


def _ti(server, name, props, required, desc="A tool."):
    return ToolInfo(
        server=server,
        name=name,
        wire_name=f"{server}__{name}",
        description=desc,
        input_schema={"type": "object", "properties": props, "required": required},
    )


WRITE = _ti(
    "workspace", "write_file",
    {"path": {"type": "string"}, "content": {"type": "string"}},
    ["path", "content"],
    desc="Write a file.",
)
RUN = _ti("code", "run_python", {"code": {"type": "string"}}, ["code"], desc="Run Python.")
LIST = _ti("workspace", "list_files", {"pattern": {"type": "string"}}, [])


# --------------------------------------------------------------------------- #
# parse_tool_blocks
# --------------------------------------------------------------------------- #


def test_parse_no_blocks():
    assert parse_tool_blocks("hello world") == []


def test_parse_qualified_name_and_digits():
    text = '```tool:workspace__write_file\n{"path": "a"}\n```'
    blocks = parse_tool_blocks(text)
    assert blocks == [("workspace__write_file", '{"path": "a"}')]


def test_parse_multiple_blocks():
    text = textwrap.dedent("""\
        thinking
        ```tool:web__web_search
        {"query": "foo"}
        ```
        ```tool:code__run_bash
        echo hi
        ```
    """)
    blocks = parse_tool_blocks(text)
    assert [b[0] for b in blocks] == ["web__web_search", "code__run_bash"]


# --------------------------------------------------------------------------- #
# parse_tool_body — JSON path
# --------------------------------------------------------------------------- #


def test_json_object_body():
    args, err = parse_tool_body('{"path": "r.md", "content": "hi"}', WRITE)
    assert err is None
    assert args == {"path": "r.md", "content": "hi"}


def test_raw_fallback_single_required_string():
    # invalid JSON, but run_python has exactly one required string param
    args, err = parse_tool_body("print('hi')", RUN)
    assert err is None
    assert args == {"code": "print('hi')"}


def test_raw_fallback_used_for_non_object_json():
    # "hello" is valid JSON but not an object → falls back to the raw value
    args, err = parse_tool_body("hello world", RUN)
    assert err is None
    assert args == {"code": "hello world"}


def test_parse_error_when_multi_arg_and_bad_json():
    args, err = parse_tool_body("not json at all", WRITE)
    assert args is None
    assert err.startswith("ERROR: invalid tool arguments")
    assert '"path": string (required)' in err
    assert '"content": string (required)' in err


def test_empty_body_no_required():
    # list_files has no required args → empty body is a parse error with summary
    args, err = parse_tool_body("", LIST)
    assert args is None
    assert "Expected:" in err


# --------------------------------------------------------------------------- #
# arg_summary
# --------------------------------------------------------------------------- #


def test_arg_summary():
    assert arg_summary(WRITE) == '{"path": string (required), "content": string (required)}'
    assert arg_summary(LIST) == '{"pattern": string}'


def test_arg_summary_nested_object():
    ti = _ti("x", "y", {"cfg": {"type": "object", "properties": {"a": {"type": "int"}}}}, [])
    assert arg_summary(ti) == '{"cfg": object}'


# --------------------------------------------------------------------------- #
# render_tool_protocol
# --------------------------------------------------------------------------- #


def test_render_lists_tools_compactly():
    out = render_tool_protocol([WRITE, LIST])
    assert "## Tool use" in out
    assert "workspace__write_file — Write a file. args: {" in out
    assert "workspace__list_files" in out


def test_render_code_guidance_only_when_code_enabled():
    without = render_tool_protocol([WRITE])
    assert "pip install" not in without
    with_code = render_tool_protocol([RUN], code_enabled=True)
    assert "pip install" in with_code
    assert "code__run_bash" in with_code
