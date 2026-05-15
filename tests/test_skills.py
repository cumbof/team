"""Tests for team/skills.py — custom skill plugin loader."""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from team.skills import SkillLoadError, load_skill, load_skills, _verify_checksum


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _write_skill(tmp_path: Path, name: str, content: str) -> str:
    """Write a skill file and return its path string."""
    p = tmp_path / name
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return str(p)


# --------------------------------------------------------------------------- #
# Single-tool format
# --------------------------------------------------------------------------- #


def test_single_tool_format(tmp_path):
    path = _write_skill(tmp_path, "calc.py", """
        TOOL_NAME = "calc"
        TOOL_DESCRIPTION = "Evaluate arithmetic."

        def execute(body, **kwargs):
            return str(eval(body.strip()))
    """)
    tools, descs, _ = load_skill(path)
    assert "calc" in tools
    assert "Evaluate arithmetic" in descs["calc"]
    result = tools["calc"]("2 + 2")
    assert result == "4"


def test_single_tool_default_description(tmp_path):
    path = _write_skill(tmp_path, "t.py", """
        TOOL_NAME = "my_tool"
        def execute(body, **kwargs):
            return "ok"
    """)
    tools, descs, _ = load_skill(path)
    assert "my_tool" in descs
    assert "my_tool" in descs["my_tool"]


# --------------------------------------------------------------------------- #
# Multi-tool format
# --------------------------------------------------------------------------- #


def test_multi_tool_format(tmp_path):
    path = _write_skill(tmp_path, "multi.py", """
        def _a(body, **kw): return "A:" + body
        def _b(body, **kw): return "B:" + body

        TOOLS = {"tool_a": _a, "tool_b": _b}
        TOOL_DESCRIPTIONS = {"tool_a": "does A", "tool_b": "does B"}
    """)
    tools, descs, _ = load_skill(path)
    assert set(tools) == {"tool_a", "tool_b"}
    assert tools["tool_a"]("x") == "A:x"
    assert descs["tool_b"] == "does B"


def test_multi_and_single_coexist(tmp_path):
    """TOOLS dict + TOOL_NAME/execute can coexist in the same file."""
    path = _write_skill(tmp_path, "both.py", """
        def _shared(body, **kw): return "shared"

        TOOLS = {"shared_tool": _shared}
        TOOL_DESCRIPTIONS = {"shared_tool": "shared desc"}

        TOOL_NAME = "extra"
        TOOL_DESCRIPTION = "extra desc"
        def execute(body, **kw): return "extra"
    """)
    tools, _, _ = load_skill(path)
    assert "shared_tool" in tools
    assert "extra" in tools


# --------------------------------------------------------------------------- #
# Error cases
# --------------------------------------------------------------------------- #


def test_missing_file_raises():
    with pytest.raises(SkillLoadError, match="not found"):
        load_skill("/no/such/file.py")


def test_no_exports_raises(tmp_path):
    path = _write_skill(tmp_path, "empty.py", "x = 1")
    with pytest.raises(SkillLoadError, match="must define"):
        load_skill(path)


def test_tools_not_dict_raises(tmp_path):
    path = _write_skill(tmp_path, "bad.py", "TOOLS = 'not a dict'")
    with pytest.raises(SkillLoadError, match="must be a dict"):
        load_skill(path)


def test_tools_non_callable_raises(tmp_path):
    path = _write_skill(tmp_path, "bad2.py", "TOOLS = {'t': 42}")
    with pytest.raises(SkillLoadError, match="not callable"):
        load_skill(path)


def test_execute_not_callable_raises(tmp_path):
    path = _write_skill(tmp_path, "bad3.py", """
        TOOL_NAME = "t"
        execute = "not callable"
    """)
    with pytest.raises(SkillLoadError, match="callable"):
        load_skill(path)


def test_syntax_error_raises(tmp_path):
    path = _write_skill(tmp_path, "syntax.py", "def (:")
    with pytest.raises(SkillLoadError, match="error executing"):
        load_skill(path)


def test_bad_source_type_raises():
    with pytest.raises(SkillLoadError, match="string or dict"):
        load_skill(123)  # type: ignore[arg-type]


def test_dict_both_path_and_url_raises(tmp_path):
    with pytest.raises(SkillLoadError, match="exactly one of"):
        load_skill({"path": "a.py", "url": "http://x.com/b.py"})


def test_dict_missing_path_and_url_raises():
    with pytest.raises(SkillLoadError, match="must have"):
        load_skill({"checksum": "sha256:abc"})


# --------------------------------------------------------------------------- #
# load_skill — dict form with path
# --------------------------------------------------------------------------- #


def test_dict_path_form(tmp_path):
    path = _write_skill(tmp_path, "d.py", """
        TOOL_NAME = "d"
        def execute(body, **kw): return "d"
    """)
    tools, _ , _ = load_skill({"path": path})
    assert "d" in tools


# --------------------------------------------------------------------------- #
# load_skill — plain string URL (mocked)
# --------------------------------------------------------------------------- #


def _make_mock_response(code: str) -> MagicMock:
    mock = MagicMock()
    mock.text = code
    mock.raise_for_status = MagicMock()
    return mock


_SKILL_CODE = textwrap.dedent("""
    TOOL_NAME = "remote_tool"
    TOOL_DESCRIPTION = "A remote tool."
    def execute(body, **kw): return "remote:" + body
""")


def test_remote_url_string(tmp_path):
    with patch("requests.get", return_value=_make_mock_response(_SKILL_CODE)):
        tools, descs, _ = load_skill("https://example.com/skill.py")
    assert "remote_tool" in tools
    assert tools["remote_tool"]("hi") == "remote:hi"


def test_remote_url_dict(tmp_path):
    with patch("requests.get", return_value=_make_mock_response(_SKILL_CODE)):
        tools, *_ = load_skill({"url": "https://example.com/skill.py"})
    assert "remote_tool" in tools


def test_remote_network_error():
    with patch("requests.get", side_effect=ConnectionError("no net")):
        with pytest.raises(SkillLoadError, match="failed to fetch"):
            load_skill("https://example.com/fail.py")


# --------------------------------------------------------------------------- #
# Checksum verification
# --------------------------------------------------------------------------- #


def test_checksum_ok_local(tmp_path):
    import hashlib
    code = "TOOL_NAME = 'ck'\ndef execute(body, **kw): return 'ok'\n"
    digest = hashlib.sha256(code.encode()).hexdigest()
    path = tmp_path / "ck.py"
    path.write_text(code, encoding="utf-8")
    tools, *_ = load_skill({"path": str(path), "checksum": f"sha256:{digest}"})
    assert "ck" in tools


def test_checksum_mismatch_local(tmp_path):
    path = _write_skill(tmp_path, "ck2.py", "TOOL_NAME='x'\ndef execute(b,**k):return'x'\n")
    with pytest.raises(SkillLoadError, match="mismatch"):
        load_skill({"path": str(path), "checksum": "sha256:badhash"})


def test_checksum_bad_format(tmp_path):
    path = _write_skill(tmp_path, "ck3.py", "TOOL_NAME='x'\ndef execute(b,**k):return'x'\n")
    with pytest.raises(SkillLoadError, match="format"):
        load_skill({"path": str(path), "checksum": "nocolon"})


def test_checksum_unknown_algo(tmp_path):
    path = _write_skill(tmp_path, "ck4.py", "TOOL_NAME='x'\ndef execute(b,**k):return'x'\n")
    with pytest.raises(SkillLoadError, match="algorithm"):
        load_skill({"path": str(path), "checksum": "bogus:abc"})


def test_verify_checksum_standalone():
    import hashlib
    data = b"hello"
    digest = hashlib.sha256(data).hexdigest()
    _verify_checksum(data, f"sha256:{digest}", "test")  # should not raise


# --------------------------------------------------------------------------- #
# load_skills (batch)
# --------------------------------------------------------------------------- #


def test_load_skills_merges(tmp_path):
    p1 = _write_skill(tmp_path, "s1.py", """
        TOOL_NAME = "t1"
        def execute(body, **kw): return "1"
    """)
    p2 = _write_skill(tmp_path, "s2.py", """
        TOOL_NAME = "t2"
        def execute(body, **kw): return "2"
    """)
    tools, *_ = load_skills([p1, p2])
    assert "t1" in tools and "t2" in tools


def test_load_skills_skips_bad(tmp_path, caplog):
    """A bad skill should be skipped (logged) rather than aborting the load."""
    p1 = _write_skill(tmp_path, "good.py", """
        TOOL_NAME = "good"
        def execute(body, **kw): return "ok"
    """)
    bad = "/nonexistent/skill.py"
    import logging
    with caplog.at_level(logging.ERROR, logger="team.skills"):
        tools, *_ = load_skills([p1, bad])
    assert "good" in tools
    assert any("skipping" in r.message for r in caplog.records)


def test_load_skills_empty():
    tools, descs, ctx = load_skills([])
    assert tools == {}
    assert descs == {}
    assert ctx == []


# --------------------------------------------------------------------------- #
# Markdown skill (context injection)
# --------------------------------------------------------------------------- #


def test_markdown_skill_returns_no_tools(tmp_path):
    p = tmp_path / "guide.md"
    p.write_text("# Style Guide\n- Use snake_case.\n", encoding="utf-8")
    tools, descs, ctx = load_skill(str(p))
    assert tools == {}
    assert descs == {}
    assert len(ctx) == 1
    assert "snake_case" in ctx[0]


def test_markdown_skill_dict_form(tmp_path):
    p = tmp_path / "rules.md"
    p.write_text("# Rules\nBe concise.\n", encoding="utf-8")
    tools, descs, ctx = load_skill({"path": str(p)})
    assert tools == {}
    assert "Be concise" in ctx[0]


def test_markdown_skill_merged_by_load_skills(tmp_path):
    md = tmp_path / "knowledge.md"
    md.write_text("# Knowledge\nFact: sky is blue.\n", encoding="utf-8")
    py = _write_skill(tmp_path, "t.py", """
        TOOL_NAME = "t"
        def execute(body, **kw): return "ok"
    """)
    tools, descs, ctx = load_skills([str(md), py])
    assert "t" in tools
    assert len(ctx) == 1
    assert "sky is blue" in ctx[0]


def test_markdown_skill_empty_file_produces_no_context(tmp_path):
    p = tmp_path / "empty.md"
    p.write_text("", encoding="utf-8")
    tools, _, ctx = load_skill(str(p))
    assert tools == {}
    assert ctx == []


# --------------------------------------------------------------------------- #
# INJECT_INTO_CONTEXT in Python skill
# --------------------------------------------------------------------------- #


def test_inject_into_context_string(tmp_path):
    path = _write_skill(tmp_path, "ctx.py", """
        TOOL_NAME = "ctx_tool"
        INJECT_INTO_CONTEXT = "## Background\\nImportant fact."

        def execute(body, **kw):
            return "ok"
    """)
    tools, _, ctx = load_skill(path)
    assert "ctx_tool" in tools
    assert len(ctx) == 1
    assert "Important fact" in ctx[0]


def test_inject_into_context_empty_string_ignored(tmp_path):
    path = _write_skill(tmp_path, "noct.py", """
        TOOL_NAME = "noct"
        INJECT_INTO_CONTEXT = "   "

        def execute(body, **kw):
            return "ok"
    """)
    _, _, ctx = load_skill(path)
    assert ctx == []


def test_inject_into_context_non_string_ignored(tmp_path):
    path = _write_skill(tmp_path, "nonstr.py", """
        TOOL_NAME = "nonstr"
        INJECT_INTO_CONTEXT = 42  # not a string — should be silently ignored

        def execute(body, **kw):
            return "ok"
    """)
    _, _, ctx = load_skill(path)
    assert ctx == []


# --------------------------------------------------------------------------- #
# render_system_prompt injection
# --------------------------------------------------------------------------- #


def test_render_system_prompt_includes_injected_context():
    """Injected context strings appear in the rendered system prompt."""
    from team.personas import render_system_prompt
    from team.config import TeamConfig, MemberConfig, WorkflowConfig, Defaults
    from pathlib import Path

    team = TeamConfig(
        name="test",
        goal="Do something.",
        workspace=Path("/tmp/test-ws"),
        workflow=WorkflowConfig(),
        defaults=Defaults(),
        members=[
            MemberConfig(name="alpha", role="Worker", model="m", persona="Be helpful.")
        ],
    )
    prompt = render_system_prompt(
        team,
        team.members[0],
        injected_context=["## My Checklist\n- Item 1\n- Item 2"],
    )
    assert "My Checklist" in prompt
    assert "Item 1" in prompt


def test_render_system_prompt_no_context_unchanged():
    """Omitting injected_context keeps the prompt unchanged vs None."""
    from team.personas import render_system_prompt
    from team.config import TeamConfig, MemberConfig, WorkflowConfig, Defaults
    from pathlib import Path

    team = TeamConfig(
        name="test",
        goal="Do something.",
        workspace=Path("/tmp/test-ws"),
        workflow=WorkflowConfig(),
        defaults=Defaults(),
        members=[
            MemberConfig(name="alpha", role="Worker", model="m", persona="Be helpful.")
        ],
    )
    p_none = render_system_prompt(team, team.members[0], injected_context=None)
    p_empty = render_system_prompt(team, team.members[0], injected_context=[])
    assert p_none == p_empty
