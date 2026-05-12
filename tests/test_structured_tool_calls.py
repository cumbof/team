"""Tests for native (structured) tool calling — F4 extension.

Covers:
- TOOL_SCHEMAS definitions and completeness
- args_to_body round-trips for every built-in tool
- OllamaClient.chat_native: tool-call path, text-reply path, arg-string parsing
- OpenAICompatClient.chat_native: same contract
- Member._run_native_agentic_turn: multi-round loop, single-shot reply, exhausted rounds
- config.tool_mode: parsing, validation, per-member override
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
    ToolCall,
    TokenUsage,
    OpenAICompatClient,
)
from team.tools import TOOL_SCHEMAS, TOOLS, args_to_body


# --------------------------------------------------------------------------- #
# TOOL_SCHEMAS coverage
# --------------------------------------------------------------------------- #


def test_all_builtin_tools_have_schema():
    """Every entry in TOOLS must have a corresponding entry in TOOL_SCHEMAS."""
    for name in TOOLS:
        assert name in TOOL_SCHEMAS, f"missing schema for tool {name!r}"


def test_schema_structure():
    """Each schema must be a valid OpenAI/Ollama function-tool dict."""
    for name, schema in TOOL_SCHEMAS.items():
        assert schema["type"] == "function", f"{name}: type != 'function'"
        fn = schema["function"]
        assert fn["name"] == name
        assert isinstance(fn["description"], str) and fn["description"]
        params = fn["parameters"]
        assert params["type"] == "object"
        assert "properties" in params
        assert "required" in params


def test_required_fields_are_subset_of_properties():
    for name, schema in TOOL_SCHEMAS.items():
        params = schema["function"]["parameters"]
        for req in params["required"]:
            assert req in params["properties"], (
                f"{name}: required field {req!r} not in properties"
            )


# --------------------------------------------------------------------------- #
# args_to_body round-trips
# --------------------------------------------------------------------------- #


def test_args_to_body_run_python():
    body = args_to_body("run_python", {"code": "print('hi')"})
    assert body == "print('hi')"


def test_args_to_body_run_bash():
    body = args_to_body("run_bash", {"command": "ls -la"})
    assert body == "ls -la"


def test_args_to_body_web_search():
    body = args_to_body("web_search", {"query": "climate change"})
    assert "query: climate change" in body


def test_args_to_body_read_url():
    body = args_to_body("read_url", {"url": "https://example.com"})
    assert "url: https://example.com" in body


def test_args_to_body_read_file():
    body = args_to_body("read_file", {"path": "src/main.py"})
    assert "path: src/main.py" in body


def test_args_to_body_write_file():
    body = args_to_body("write_file", {"path": "out.txt", "content": "hello\n"})
    assert "path: out.txt" in body
    assert "---" in body
    assert "hello" in body


def test_args_to_body_append_file():
    body = args_to_body("append_file", {"path": "log.txt", "content": "line\n"})
    assert "path: log.txt" in body
    assert "---" in body
    assert "line" in body


def test_args_to_body_list_files_with_pattern():
    body = args_to_body("list_files", {"pattern": "**/*.py"})
    assert "pattern: **/*.py" in body


def test_args_to_body_list_files_empty():
    body = args_to_body("list_files", {})
    assert body == ""


def test_args_to_body_remember():
    body = args_to_body("remember", {
        "key": "mykey",
        "value": "stored value",
        "tags": "a,b",
        "importance": 0.9,
    })
    assert "key: mykey" in body
    assert "tags: a,b" in body
    assert "importance: 0.9" in body
    assert "---" in body
    assert "stored value" in body


def test_args_to_body_remember_minimal():
    body = args_to_body("remember", {"key": "k", "value": "v"})
    assert "key: k" in body
    assert "---" in body
    assert "v" in body
    assert "tags" not in body


def test_args_to_body_recall():
    body = args_to_body("recall", {"query": "protein", "limit": 3})
    assert "query: protein" in body
    assert "limit: 3" in body


def test_args_to_body_forget():
    body = args_to_body("forget", {"key": "old_key"})
    assert "key: old_key" in body


def test_args_to_body_list_memories():
    body = args_to_body("list_memories", {"tag": "results", "limit": 10})
    assert "tag: results" in body
    assert "limit: 10" in body


def test_args_to_body_assert_belief():
    body = args_to_body("assert_belief", {
        "claim": "The sky is blue.",
        "confidence": 0.95,
        "evidence": "Spectroscopy data.",
    })
    assert "confidence: 0.95" in body
    assert "evidence: Spectroscopy data." in body
    assert "---" in body
    assert "The sky is blue." in body


def test_args_to_body_contest_belief():
    body = args_to_body("contest_belief", {"id": "abc123", "reason": "Too vague."})
    assert "id: abc123" in body
    assert "reason: Too vague." in body


def test_args_to_body_accept_belief():
    body = args_to_body("accept_belief", {"id": "def456"})
    assert "id: def456" in body


def test_args_to_body_list_beliefs():
    body = args_to_body("list_beliefs", {"status": "accepted"})
    assert "status: accepted" in body


def test_args_to_body_list_beliefs_empty():
    body = args_to_body("list_beliefs", {})
    assert body == ""


def test_args_to_body_delegate_task():
    body = args_to_body("delegate_task", {
        "url": "http://lab-b:7001",
        "goal": "Analyse data.",
        "context": "Use CSV.",
        "files": "data.csv",
        "timeout": 300,
    })
    assert "url: http://lab-b:7001" in body
    assert "goal: Analyse data." in body
    assert "context: Use CSV." in body
    assert "files: data.csv" in body
    assert "timeout: 300" in body


def test_args_to_body_log_decision():
    body = args_to_body("log_decision", {
        "title": "Chose pandas",
        "rationale": "Already a dep.",
        "alternatives": "polars, dask",
    })
    assert "title: Chose pandas" in body
    assert "rationale: Already a dep." in body
    assert "alternatives: polars, dask" in body


def test_args_to_body_read_decisions():
    assert args_to_body("read_decisions", {}) == ""


def test_args_to_body_unknown_tool_returns_json():
    body = args_to_body("my_custom_skill", {"foo": "bar", "n": 42})
    data = json.loads(body)
    assert data["foo"] == "bar"
    assert data["n"] == 42


# --------------------------------------------------------------------------- #
# ChatMessage.to_dict with tool_calls
# --------------------------------------------------------------------------- #


def test_chat_message_to_dict_no_tool_calls():
    msg = ChatMessage(role="assistant", content="hello")
    d = msg.to_dict()
    assert d == {"role": "assistant", "content": "hello"}
    assert "tool_calls" not in d


def test_chat_message_to_dict_with_tool_calls():
    tc = ToolCall(name="run_bash", arguments={"command": "ls"})
    msg = ChatMessage(role="assistant", content="", tool_calls=[tc])
    d = msg.to_dict()
    assert d["tool_calls"][0]["function"]["name"] == "run_bash"
    assert d["tool_calls"][0]["function"]["arguments"] == {"command": "ls"}


def test_tool_message_to_dict():
    msg = ChatMessage(role="tool", content="result text")
    d = msg.to_dict()
    assert d["role"] == "tool"
    assert d["content"] == "result text"


# --------------------------------------------------------------------------- #
# OllamaClient.chat_native
# --------------------------------------------------------------------------- #


def _ollama_client() -> OllamaClient:
    return OllamaClient(base_url="http://localhost:11434")


def _tool_call_response(name: str, args: dict) -> dict:
    """Build a fake Ollama /api/chat response with tool_calls."""
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
    resp.json.return_value = _tool_call_response("run_bash", {"command": "ls"})
    with patch.object(client._session, "post", return_value=resp):
        content, tool_calls = client.chat_native(
            "llama3.1:8b",
            [ChatMessage("user", "list files")],
            tools=[TOOL_SCHEMAS["run_bash"]],
        )
    assert content == ""
    assert len(tool_calls) == 1
    assert tool_calls[0].name == "run_bash"
    assert tool_calls[0].arguments == {"command": "ls"}


def test_ollama_chat_native_returns_text():
    client = _ollama_client()
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = _text_response("Here is my answer.")
    with patch.object(client._session, "post", return_value=resp):
        content, tool_calls = client.chat_native(
            "llama3.1:8b",
            [ChatMessage("user", "what is 2+2?")],
            tools=[TOOL_SCHEMAS["run_bash"]],
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
                {"function": {"name": "run_bash", "arguments": '{"command": "echo hi"}'}}
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
        "choices": [{
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(args),  # OpenAI sends as JSON string
                    },
                }],
            }
        }],
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
    resp.json.return_value = _openai_tool_call_response("write_file", {"path": "a.txt", "content": "hi"})
    with patch.object(client._session, "post", return_value=resp):
        content, tool_calls = client.chat_native(
            "gpt-4o-mini",
            [ChatMessage("user", "write a file")],
            tools=[TOOL_SCHEMAS["write_file"]],
        )
    assert content == ""
    assert len(tool_calls) == 1
    assert tool_calls[0].name == "write_file"
    assert tool_calls[0].arguments == {"path": "a.txt", "content": "hi"}


def test_openai_compat_chat_native_returns_text():
    client = _openai_client()
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = _openai_text_response("Final answer.")
    with patch.object(client._session, "post", return_value=resp):
        content, tool_calls = client.chat_native(
            "gpt-4o-mini",
            [ChatMessage("user", "hi")],
            tools=[],
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


def _make_member(tmp_path: Path, tools: list[str], tool_mode: str = "native"):
    """Create a minimal Member for testing without Docker."""
    from team.config import Defaults, MemberConfig, TeamConfig, WorkflowConfig
    from team.member import Member

    cfg = MemberConfig(
        name="agent",
        role="Agent",
        model="llama3.1:8b",
        persona="You are a helpful agent.",
        tools=tools,
        tool_mode=tool_mode,
    )
    team = TeamConfig(
        name="test-team",
        goal="Test native tool calls.",
        workspace=tmp_path,
        workflow=WorkflowConfig(type="round_robin", max_rounds=2),
        defaults=Defaults(tools=[], tool_mode=tool_mode),
        members=[cfg],
    )
    runtime = MagicMock()
    runtime.base_url = "http://localhost:11434"

    with patch("team.member.OllamaClient"), \
         patch("team.member.OpenAICompatClient"), \
         patch("team.member.load_skills", return_value=({}, {}, [])), \
         patch("team.member.render_system_prompt", return_value="system"):
        member = Member(team, cfg, runtime)
    member._ready = True
    return member


def test_native_turn_single_tool_then_reply(tmp_path: Path):
    """One tool call round followed by a text reply."""
    from team.bus import Transcript
    from team.workspace import SharedWorkspace

    member = _make_member(tmp_path, ["run_bash"])
    transcript = Transcript()
    workspace = SharedWorkspace(tmp_path / "shared")

    call_responses = [
        ("", [ToolCall("run_bash", {"command": "echo hello"})]),
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
        on_tool_result=lambda m, t, r: tool_results_seen.append((m, t)),
    )

    assert "hello" in result.content.lower() or "output" in result.content.lower()
    assert len(tool_calls_seen) == 1
    assert tool_calls_seen[0] == ("agent", "run_bash")
    assert len(tool_results_seen) == 1
    assert result.prompt_tokens == 20   # 10 + 10 (two calls)
    assert result.completion_tokens == 10


def test_native_turn_immediate_text_reply(tmp_path: Path):
    """Model returns text on the first call — no tool round needed."""
    from team.bus import Transcript
    from team.workspace import SharedWorkspace

    member = _make_member(tmp_path, ["web_search"])
    transcript = Transcript()
    workspace = SharedWorkspace(tmp_path / "shared")

    def mock_chat_native(model, messages, tools, **kw):
        member.client.last_usage = TokenUsage(prompt_tokens=15, completion_tokens=20)
        return "Direct answer.", []

    member.client.chat_native = mock_chat_native

    result = member.take_turn(transcript, workspace)

    assert result.content == "Direct answer."
    assert result.prompt_tokens == 15


def test_native_turn_exhausted_rounds(tmp_path: Path):
    """When all rounds are used up, a final no-tools call is made."""
    from team.bus import Transcript
    from team.workspace import SharedWorkspace
    from team.config import Defaults, MemberConfig, TeamConfig, WorkflowConfig
    from team.member import Member

    cfg = MemberConfig(
        name="agent",
        role="Agent",
        model="llama3.1:8b",
        persona="Test.",
        tools=["run_bash"],
        max_tool_rounds=2,
        tool_mode="native",
    )
    team = TeamConfig(
        name="t", goal="g", workspace=tmp_path,
        workflow=WorkflowConfig(type="round_robin", max_rounds=2),
        defaults=Defaults(tools=[], tool_mode="native"),
        members=[cfg],
    )
    runtime = MagicMock()
    runtime.base_url = "http://localhost:11434"
    with patch("team.member.OllamaClient"), \
         patch("team.member.OpenAICompatClient"), \
         patch("team.member.load_skills", return_value=({}, {}, [])), \
         patch("team.member.render_system_prompt", return_value="system"):
        member = Member(team, cfg, runtime)
    member._ready = True

    call_count = [0]

    def mock_chat_native(model, messages, tools, **kw):
        member.client.last_usage = TokenUsage(prompt_tokens=5, completion_tokens=2)
        call_count[0] += 1
        if tools:  # keeps requesting tool calls
            return "", [ToolCall("run_bash", {"command": "echo x"})]
        else:  # final call with no tools
            return "Final reply after exhaustion.", []

    member.client.chat_native = mock_chat_native

    transcript = Transcript()
    workspace = SharedWorkspace(tmp_path / "shared")
    result = member.take_turn(transcript, workspace)

    # 2 tool rounds + 1 final call = 3 calls total
    assert call_count[0] == 3
    assert result.content == "Final reply after exhaustion."


def test_native_turn_disabled_tool_returns_error(tmp_path: Path):
    """Model requests a tool not in the enabled list — gets an error result."""
    from team.bus import Transcript
    from team.workspace import SharedWorkspace

    member = _make_member(tmp_path, ["web_search"])  # only web_search enabled
    transcript = Transcript()
    workspace = SharedWorkspace(tmp_path / "shared")

    tool_results = []

    responses = [
        ("", [ToolCall("run_bash", {"command": "rm -rf /"})]),  # not enabled!
        ("OK, noted.", []),
    ]
    idx = [0]

    def mock_chat_native(model, messages, tools, **kw):
        member.client.last_usage = TokenUsage(5, 5)
        resp = responses[idx[0]]
        idx[0] += 1
        return resp

    member.client.chat_native = mock_chat_native
    member.take_turn(
        transcript, workspace,
        on_tool_result=lambda m, t, r: tool_results.append(r),
    )
    assert any("not enabled" in r for r in tool_results)


# --------------------------------------------------------------------------- #
# config: tool_mode parsing and validation
# --------------------------------------------------------------------------- #


def _make_yaml(tmp_path: Path, tool_mode_str: str = "") -> Path:
    p = tmp_path / "team.yaml"
    p.write_text(
        f"name: t\ngoal: g\nmembers:\n  - name: a\n    role: R\n    model: m\n    persona: p\n"
        f"{tool_mode_str}"
    )
    return p


def test_config_default_tool_mode(tmp_path: Path):
    from team.config import load_team
    p = _make_yaml(tmp_path)
    cfg = load_team(p)
    assert cfg.defaults.tool_mode == "text"


def test_config_native_tool_mode(tmp_path: Path):
    from team.config import load_team
    p = tmp_path / "team.yaml"
    p.write_text(
        "name: t\ngoal: g\n"
        "defaults:\n  tool_mode: native\n"
        "members:\n  - name: a\n    role: R\n    model: m\n    persona: p\n"
    )
    cfg = load_team(p)
    assert cfg.defaults.tool_mode == "native"


def test_config_per_member_tool_mode(tmp_path: Path):
    from team.config import load_team
    p = tmp_path / "team.yaml"
    p.write_text(
        "name: t\ngoal: g\n"
        "members:\n  - name: a\n    role: R\n    model: m\n    persona: p\n    tool_mode: native\n"
    )
    cfg = load_team(p)
    assert cfg.members[0].tool_mode == "native"


def test_config_invalid_tool_mode_raises(tmp_path: Path):
    from team.config import load_team, TeamConfigError
    p = tmp_path / "team.yaml"
    p.write_text(
        "name: t\ngoal: g\n"
        "defaults:\n  tool_mode: fancy\n"
        "members:\n  - name: a\n    role: R\n    model: m\n    persona: p\n"
    )
    with pytest.raises(TeamConfigError, match="tool_mode"):
        load_team(p)
