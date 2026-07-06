"""Tests for native (structured) tool calling.

Covers:
- OllamaClient.chat_native: tool-call path, text-reply path, arg-string parsing
- OpenAICompatClient.chat_native: same contract
- Member._run_native_agentic_turn: dispatch to MemberToolset (wire names),
  multi-round loop, single-shot reply, exhausted rounds, not-enabled error
- ChatMessage.to_dict with tool_calls field
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from team.ollama_client import (
    ChatMessage,
    OllamaClient,
    OllamaError,
    OpenAICompatClient,
    TokenUsage,
    ToolCall,
)

#: A sample function-tool schema (shape the clients receive in the tools= list).
_SAMPLE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "code__run_bash",
        "description": "Run a bash command.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
}


# --------------------------------------------------------------------------- #
# ChatMessage.to_dict with tool_calls
# --------------------------------------------------------------------------- #


def test_chat_message_to_dict_no_tool_calls():
    d = ChatMessage(role="assistant", content="hello").to_dict()
    assert d == {"role": "assistant", "content": "hello"}
    assert "tool_calls" not in d


def test_chat_message_to_dict_with_tool_calls():
    tc = ToolCall(name="code__run_bash", arguments={"command": "ls"})
    d = ChatMessage(role="assistant", content="", tool_calls=[tc]).to_dict()
    assert d["tool_calls"][0]["function"]["name"] == "code__run_bash"
    assert d["tool_calls"][0]["function"]["arguments"] == {"command": "ls"}


def test_tool_message_to_dict():
    d = ChatMessage(role="tool", content="result text").to_dict()
    assert d["role"] == "tool"
    assert d["content"] == "result text"


# --------------------------------------------------------------------------- #
# OllamaClient.chat_native
# --------------------------------------------------------------------------- #


def _ollama_client() -> OllamaClient:
    return OllamaClient(base_url="http://localhost:11434")


def _tool_call_response(name: str, args: dict) -> dict:
    return {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": name, "arguments": args}}],
        },
        "done": True,
        "prompt_eval_count": 10,
        "eval_count": 5,
    }


def _text_response(content: str) -> dict:
    return {
        "message": {"role": "assistant", "content": content},
        "done": True,
        "prompt_eval_count": 8,
        "eval_count": 12,
    }


def test_ollama_chat_native_returns_tool_calls():
    client = _ollama_client()
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = _tool_call_response("code__run_bash", {"command": "ls"})
    with patch.object(client._session, "post", return_value=resp):
        content, tool_calls = client.chat_native(
            "llama3.1:8b", [ChatMessage("user", "list files")], tools=[_SAMPLE_SCHEMA]
        )
    assert content == ""
    assert len(tool_calls) == 1
    assert tool_calls[0].name == "code__run_bash"
    assert tool_calls[0].arguments == {"command": "ls"}


def test_ollama_chat_native_returns_text():
    client = _ollama_client()
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = _text_response("Here is my answer.")
    with patch.object(client._session, "post", return_value=resp):
        content, tool_calls = client.chat_native(
            "llama3.1:8b", [ChatMessage("user", "what is 2+2?")], tools=[_SAMPLE_SCHEMA]
        )
    assert content == "Here is my answer."
    assert tool_calls == []


def test_ollama_chat_native_updates_token_usage():
    client = _ollama_client()
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = _text_response("ok")
    with patch.object(client._session, "post", return_value=resp):
        client.chat_native("m", [ChatMessage("user", "hi")], tools=[])
    assert client.last_usage is not None
    assert client.last_usage.prompt_tokens == 8
    assert client.last_usage.completion_tokens == 12


def test_ollama_chat_native_parses_string_args():
    """Ollama sometimes returns arguments as a JSON string — must be parsed."""
    client = _ollama_client()
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "code__run_bash", "arguments": '{"command": "echo hi"}'}}
            ],
        },
        "done": True,
        "prompt_eval_count": 5,
        "eval_count": 3,
    }
    with patch.object(client._session, "post", return_value=resp):
        _, tool_calls = client.chat_native("m", [ChatMessage("user", "x")], tools=[])
    assert tool_calls[0].arguments == {"command": "echo hi"}


def test_ollama_chat_native_raises_on_4xx():
    client = _ollama_client()
    resp = MagicMock()
    resp.status_code = 404
    resp.text = "not found"
    with patch.object(client._session, "post", return_value=resp):
        with pytest.raises(OllamaError, match="404"):
            client.chat_native("m", [ChatMessage("user", "hi")], tools=[])


# --------------------------------------------------------------------------- #
# OpenAICompatClient.chat_native
# --------------------------------------------------------------------------- #


def _openai_client() -> OpenAICompatClient:
    return OpenAICompatClient(base_url="http://localhost:11435", api_key="test")


def _openai_tool_call_response(name: str, args: dict) -> dict:
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(args)},
                        }
                    ],
                }
            }
        ],
        "usage": {"prompt_tokens": 20, "completion_tokens": 8},
    }


def _openai_text_response(content: str) -> dict:
    return {
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 15, "completion_tokens": 10},
    }


def test_openai_compat_chat_native_returns_tool_calls():
    client = _openai_client()
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = _openai_tool_call_response(
        "workspace__write_file", {"path": "a.txt", "content": "hi"}
    )
    with patch.object(client._session, "post", return_value=resp):
        content, tool_calls = client.chat_native(
            "gpt-4o-mini", [ChatMessage("user", "write a file")], tools=[_SAMPLE_SCHEMA]
        )
    assert content == ""
    assert len(tool_calls) == 1
    assert tool_calls[0].name == "workspace__write_file"
    assert tool_calls[0].arguments == {"path": "a.txt", "content": "hi"}


def test_openai_compat_chat_native_returns_text():
    client = _openai_client()
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = _openai_text_response("Final answer.")
    with patch.object(client._session, "post", return_value=resp):
        content, tool_calls = client.chat_native(
            "gpt-4o-mini", [ChatMessage("user", "hi")], tools=[]
        )
    assert content == "Final answer."
    assert tool_calls == []


def test_openai_compat_chat_native_updates_usage():
    client = _openai_client()
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = _openai_text_response("ok")
    with patch.object(client._session, "post", return_value=resp):
        client.chat_native("m", [ChatMessage("user", "hi")], tools=[])
    assert client.last_usage.prompt_tokens == 15
    assert client.last_usage.completion_tokens == 10


# --------------------------------------------------------------------------- #
# Member._run_native_agentic_turn (via take_turn with tool_mode=native)
# --------------------------------------------------------------------------- #


def _make_member(tmp_path: Path, toolset, tool_mode: str = "native", max_tool_rounds=None):
    """Create a minimal Member wired to a real MemberToolset."""
    from team.config import Defaults, MemberConfig, TeamConfig, WorkflowConfig
    from team.member import Member

    cfg = MemberConfig(
        name="agent",
        role="Agent",
        model="llama3.1:8b",
        persona="You are a helpful agent.",
        tool_mode=tool_mode,
        max_tool_rounds=max_tool_rounds,
    )
    team = TeamConfig(
        name="test-team",
        goal="Test native tool calls.",
        workspace=tmp_path,
        workflow=WorkflowConfig(type="round_robin", max_rounds=2),
        defaults=Defaults(tool_mode=tool_mode),
        members=[cfg],
    )
    runtime = MagicMock()
    runtime.base_url = "http://localhost:11434"

    with patch("team.member.OllamaClient"), \
         patch("team.member.OpenAICompatClient"), \
         patch("team.member.render_system_prompt", return_value="system"):
        member = Member(team, cfg, runtime, toolset=toolset)
    member._ready = True
    return member


def test_native_turn_single_tool_then_reply(tmp_path: Path, mcp_toolset):
    """One tool call round (real echo via code server) followed by a text reply."""
    from team.bus import Transcript
    from team.workspace import SharedWorkspace

    ts = mcp_toolset("agent", ["code/*"], workspace_path=tmp_path)
    member = _make_member(tmp_path, ts)
    transcript = Transcript()
    workspace = SharedWorkspace(tmp_path / "shared")

    call_responses = [
        ("", [ToolCall("code__run_bash", {"command": "echo hello"})]),
        ("The output was: hello", []),
    ]
    call_idx = [0]

    def mock_chat_native(model, messages, tools, **kw):
        resp = call_responses[call_idx[0]]
        call_idx[0] += 1
        member.client.last_usage = TokenUsage(prompt_tokens=10, completion_tokens=5)
        return resp

    member.client.chat_native = mock_chat_native

    tool_calls_seen = []
    tool_results_seen = []

    result = member.take_turn(
        transcript, workspace,
        on_tool_call=lambda m, t, b: tool_calls_seen.append((m, t)),
        on_tool_result=lambda m, t, r: tool_results_seen.append((m, t, r)),
    )

    assert "hello" in result.content.lower() or "output" in result.content.lower()
    assert tool_calls_seen == [("agent", "code__run_bash")]
    assert len(tool_results_seen) == 1
    # the real tool actually ran and echoed hello
    assert "hello" in tool_results_seen[0][2]
    assert result.prompt_tokens == 20   # 10 + 10 (two calls)
    assert result.completion_tokens == 10


def test_native_turn_immediate_text_reply(tmp_path: Path, mcp_toolset):
    """Model returns text on the first call — no tool round needed."""
    from team.bus import Transcript
    from team.workspace import SharedWorkspace

    ts = mcp_toolset("agent", ["web/*"], workspace_path=tmp_path)
    member = _make_member(tmp_path, ts)
    transcript = Transcript()
    workspace = SharedWorkspace(tmp_path / "shared")

    def mock_chat_native(model, messages, tools, **kw):
        member.client.last_usage = TokenUsage(prompt_tokens=15, completion_tokens=20)
        return "Direct answer.", []

    member.client.chat_native = mock_chat_native
    result = member.take_turn(transcript, workspace)

    assert result.content == "Direct answer."
    assert result.prompt_tokens == 15


def test_native_turn_exhausted_rounds(tmp_path: Path, mcp_toolset):
    """When all rounds are used up, a final no-tools call is made."""
    from team.bus import Transcript
    from team.workspace import SharedWorkspace

    ts = mcp_toolset("agent", ["code/*"], workspace_path=tmp_path)
    member = _make_member(tmp_path, ts, max_tool_rounds=2)
    call_count = [0]

    def mock_chat_native(model, messages, tools, **kw):
        member.client.last_usage = TokenUsage(prompt_tokens=5, completion_tokens=2)
        call_count[0] += 1
        if tools:
            return "", [ToolCall("code__run_bash", {"command": "echo x"})]
        return "Final reply after exhaustion.", []

    member.client.chat_native = mock_chat_native
    transcript = Transcript()
    workspace = SharedWorkspace(tmp_path / "shared")
    result = member.take_turn(transcript, workspace)

    assert call_count[0] == 3   # 2 tool rounds + 1 final call
    assert result.content == "Final reply after exhaustion."


def test_native_turn_disabled_tool_returns_error(tmp_path: Path, mcp_toolset):
    """Model requests a tool not enabled for the member — gets an error result."""
    from team.bus import Transcript
    from team.workspace import SharedWorkspace

    ts = mcp_toolset("agent", ["web/*"], workspace_path=tmp_path)  # only web enabled
    member = _make_member(tmp_path, ts)
    transcript = Transcript()
    workspace = SharedWorkspace(tmp_path / "shared")

    responses = [
        ("", [ToolCall("code__run_bash", {"command": "rm -rf /"})]),  # not enabled!
        ("OK, noted.", []),
    ]
    idx = [0]

    def mock_chat_native(model, messages, tools, **kw):
        member.client.last_usage = TokenUsage(5, 5)
        resp = responses[idx[0]]
        idx[0] += 1
        return resp

    member.client.chat_native = mock_chat_native
    tool_results = []
    member.take_turn(
        transcript, workspace,
        on_tool_result=lambda m, t, r: tool_results.append(r),
    )
    assert any("not enabled" in r for r in tool_results)


# --------------------------------------------------------------------------- #
# config: tool_mode parsing and validation
# --------------------------------------------------------------------------- #


def test_config_default_tool_mode(tmp_path: Path):
    from team.config import load_team

    p = tmp_path / "team.yaml"
    p.write_text("name: t\ngoal: g\nmembers:\n  - name: a\n    role: R\n    model: m\n    persona: p\n")
    assert load_team(p).defaults.tool_mode == "text"


def test_config_native_tool_mode(tmp_path: Path):
    from team.config import load_team

    p = tmp_path / "team.yaml"
    p.write_text(
        "name: t\ngoal: g\ndefaults:\n  tool_mode: native\n"
        "members:\n  - name: a\n    role: R\n    model: m\n    persona: p\n"
    )
    assert load_team(p).defaults.tool_mode == "native"


def test_config_per_member_tool_mode(tmp_path: Path):
    from team.config import load_team

    p = tmp_path / "team.yaml"
    p.write_text(
        "name: t\ngoal: g\n"
        "members:\n  - name: a\n    role: R\n    model: m\n    persona: p\n    tool_mode: native\n"
    )
    assert load_team(p).members[0].tool_mode == "native"


def test_config_invalid_tool_mode_raises(tmp_path: Path):
    from team.config import TeamConfigError, load_team

    p = tmp_path / "team.yaml"
    p.write_text(
        "name: t\ngoal: g\ndefaults:\n  tool_mode: fancy\n"
        "members:\n  - name: a\n    role: R\n    model: m\n    persona: p\n"
    )
    with pytest.raises(TeamConfigError, match="tool_mode"):
        load_team(p)
