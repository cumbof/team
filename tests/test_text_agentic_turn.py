"""Integration tests for the text-mode agentic loop (Member._run_agentic_turn)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from team.bus import Transcript
from team.ollama_client import TokenUsage
from team.workspace import SharedWorkspace


def _make_member(tmp_path: Path, toolset):
    from team.config import Defaults, MemberConfig, TeamConfig, WorkflowConfig
    from team.member import Member

    cfg = MemberConfig(
        name="agent", role="Agent", model="m", persona="p", tool_mode="text",
    )
    team = TeamConfig(
        name="t", goal="g", workspace=tmp_path,
        workflow=WorkflowConfig(type="round_robin", max_rounds=2),
        defaults=Defaults(tool_mode="text"), members=[cfg],
    )
    runtime = MagicMock()
    runtime.base_url = "http://localhost:11434"
    with patch("team.member.OllamaClient"), \
         patch("team.member.OpenAICompatClient"), \
         patch("team.member.render_system_prompt", return_value="system"):
        member = Member(team, cfg, runtime, toolset=toolset)
    member._ready = True
    return member


def _script(member, replies):
    """Make member._call_llm return each reply in turn."""
    idx = [0]

    def _call_llm(messages, token_callback):
        reply = replies[idx[0]]
        idx[0] += 1
        member.client.last_usage = TokenUsage(prompt_tokens=5, completion_tokens=3)
        return reply

    member._call_llm = _call_llm


def test_text_turn_json_body_dispatch(tmp_path, mcp_toolset):
    ts = mcp_toolset("agent", ["workspace/*"], workspace_path=tmp_path)
    member = _make_member(tmp_path, ts)
    _script(member, [
        '```tool:workspace__write_file\n{"path": "a.txt", "content": "hi there"}\n```',
        "Done writing the file.",
    ])
    result = member.take_turn(Transcript(), SharedWorkspace(tmp_path / "shared"))
    assert result.content == "Done writing the file."
    assert (tmp_path / "a.txt").read_text() == "hi there"


def test_text_turn_raw_fallback_for_code(tmp_path, mcp_toolset):
    ts = mcp_toolset("agent", ["code/*"], workspace_path=tmp_path)
    member = _make_member(tmp_path, ts)
    results = []
    _script(member, [
        "```tool:code__run_python\nprint('hello from raw')\n```",
        "The script printed hello.",
    ])
    member.take_turn(
        Transcript(), SharedWorkspace(tmp_path / "shared"),
        on_tool_result=lambda m, t, r: results.append(r),
    )
    assert any("hello from raw" in r for r in results)


def test_text_turn_unknown_tool_injects_error(tmp_path, mcp_toolset):
    ts = mcp_toolset("agent", ["workspace/*"], workspace_path=tmp_path)
    member = _make_member(tmp_path, ts)
    results = []
    _script(member, [
        "```tool:workspace__nonexistent\n{}\n```",
        "Understood.",
    ])
    member.take_turn(
        Transcript(), SharedWorkspace(tmp_path / "shared"),
        on_tool_result=lambda m, t, r: results.append(r),
    )
    assert any("unknown tool" in r for r in results)


def test_text_turn_bad_json_multi_arg_injects_error(tmp_path, mcp_toolset):
    ts = mcp_toolset("agent", ["workspace/*"], workspace_path=tmp_path)
    member = _make_member(tmp_path, ts)
    results = []
    _script(member, [
        "```tool:workspace__write_file\nnot valid json\n```",
        "Retrying.",
    ])
    member.take_turn(
        Transcript(), SharedWorkspace(tmp_path / "shared"),
        on_tool_result=lambda m, t, r: results.append(r),
    )
    assert any("invalid tool arguments" in r for r in results)
