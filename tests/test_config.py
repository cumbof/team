import textwrap
from pathlib import Path

import pytest

from team.config import TeamConfigError, load_team


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "team.yaml"
    p.write_text(textwrap.dedent(body), encoding="utf-8")
    return p


def test_load_minimal(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        """
        name: t1
        goal: do stuff
        members:
          - name: a
            role: Worker
            model: llama3:8b
            persona: be useful
        """,
    )
    cfg = load_team(p)
    assert cfg.name == "t1"
    assert len(cfg.members) == 1
    assert cfg.members[0].name == "a"
    assert cfg.workflow.type == "round_robin"
    assert cfg.workflow.max_rounds == 6


def test_invalid_member_name(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        """
        name: t1
        goal: g
        members:
          - name: BadName
            role: r
            model: m
            persona: p
        """,
    )
    with pytest.raises(TeamConfigError):
        load_team(p)


def test_duplicate_members(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        """
        name: t1
        goal: g
        members:
          - {name: a, role: r, model: m, persona: p}
          - {name: a, role: r, model: m, persona: p}
        """,
    )
    with pytest.raises(TeamConfigError):
        load_team(p)


def test_review_loop_requires_pair(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        """
        name: t1
        goal: g
        workflow: {type: review_loop, producer: a, reviewer: ghost}
        members:
          - {name: a, role: r, model: m, persona: p}
        """,
    )
    with pytest.raises(TeamConfigError):
        load_team(p)


def test_manager_must_exist(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        """
        name: t1
        goal: g
        workflow: {type: manager, manager: ghost}
        members:
          - {name: a, role: r, model: m, persona: p}
        """,
    )
    with pytest.raises(TeamConfigError):
        load_team(p)


def test_member_lookup(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        """
        name: t1
        goal: g
        members:
          - {name: a, role: Lead, model: m, persona: p}
          - {name: b, role: Eng,  model: m, persona: p}
        """,
    )
    cfg = load_team(p)
    assert cfg.member_names() == ["a", "b"]
    assert cfg.member("b").role == "Eng"
    with pytest.raises(KeyError):
        cfg.member("nope")


# --------------------------------------------------------------------------- #
# F10: Remote Ollama
# --------------------------------------------------------------------------- #


def test_member_ollama_url_is_parsed(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        """
        name: t1
        goal: g
        members:
          - name: a
            role: r
            model: m
            persona: p
            ollama_url: http://192.168.1.10:11434
        """,
    )
    cfg = load_team(p)
    assert cfg.members[0].ollama_url == "http://192.168.1.10:11434"


def test_defaults_ollama_url_is_parsed(tmp_path: Path) -> None:
    """defaults.ollama_url routes all members to a host Ollama instance."""
    p = _write(
        tmp_path,
        """
        name: t1
        goal: g
        defaults:
          ollama_url: http://localhost:11434
        members:
          - {name: a, role: r, model: m, persona: p}
          - {name: b, role: r, model: m, persona: p}
        """,
    )
    cfg = load_team(p)
    assert cfg.defaults.ollama_url == "http://localhost:11434"


# --------------------------------------------------------------------------- #
# F1: OpenAI-compat backend
# --------------------------------------------------------------------------- #


def test_member_backend_and_api_base_parsed(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        """
        name: t1
        goal: g
        members:
          - name: a
            role: r
            model: gpt-4o
            persona: p
            backend: openai_compat
            api_base: https://api.openai.com
            api_key: env:OPENAI_API_KEY
        """,
    )
    cfg = load_team(p)
    m = cfg.members[0]
    assert m.backend == "openai_compat"
    assert m.api_base == "https://api.openai.com"
    assert m.api_key == "env:OPENAI_API_KEY"


# --------------------------------------------------------------------------- #
# F2: Context window management
# --------------------------------------------------------------------------- #


def test_context_strategy_in_defaults(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        """
        name: t1
        goal: g
        defaults:
          context_strategy: sliding_window
          context_budget: 20
        members:
          - {name: a, role: r, model: m, persona: p}
        """,
    )
    cfg = load_team(p)
    assert cfg.defaults.context_strategy == "sliding_window"
    assert cfg.defaults.context_budget == 20


def test_context_strategy_per_member(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        """
        name: t1
        goal: g
        members:
          - name: a
            role: r
            model: m
            persona: p
            context_strategy: truncate
            context_budget: 4096
        """,
    )
    cfg = load_team(p)
    assert cfg.members[0].context_strategy == "truncate"
    assert cfg.members[0].context_budget == 4096


# --------------------------------------------------------------------------- #
# F3: Debate workflow config
# --------------------------------------------------------------------------- #


def test_debate_workflow_valid(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        """
        name: t1
        goal: g
        workflow:
          type: debate
          max_rounds: 2
          pro: alice
          con: bob
          judge: carol
        members:
          - {name: alice, role: r, model: m, persona: p}
          - {name: bob, role: r, model: m, persona: p}
          - {name: carol, role: r, model: m, persona: p}
        """,
    )
    cfg = load_team(p)
    assert cfg.workflow.type == "debate"
    assert cfg.workflow.options["pro"] == "alice"
    assert cfg.workflow.options["judge"] == "carol"



# --------------------------------------------------------------------------- #
# MCP servers + tool patterns + extra_context (v0.18)
# --------------------------------------------------------------------------- #


def test_mcp_servers_parsed(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        """
        name: t1
        goal: g
        mcp_servers:
          github:
            transport: stdio
            command: npx
            args: ["-y", "@modelcontextprotocol/server-github"]
            env: { GITHUB_TOKEN: "env:GH" }
          kb:
            transport: http
            url: https://kb.example.com/mcp
            headers: { Authorization: "env:KB" }
          helper:
            transport: entry_point
            entry_point: mypkg.servers:build
        members:
          - name: a
            role: W
            model: m
            persona: p
        """,
    )
    cfg = load_team(p)
    names = {s.name: s for s in cfg.mcp_servers}
    assert set(names) == {"github", "kb", "helper"}
    assert names["github"].transport == "stdio"
    assert names["github"].command == "npx"
    assert names["kb"].url.endswith("/mcp")
    assert names["helper"].entry_point == "mypkg.servers:build"


def test_mcp_server_reserved_name_rejected(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        """
        name: t1
        goal: g
        mcp_servers:
          code:
            transport: stdio
            command: foo
        members:
          - {name: a, role: W, model: m, persona: p}
        """,
    )
    with pytest.raises(TeamConfigError, match="reserved"):
        load_team(p)


def test_mcp_server_missing_command(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        """
        name: t1
        goal: g
        mcp_servers:
          x:
            transport: stdio
        members:
          - {name: a, role: W, model: m, persona: p}
        """,
    )
    with pytest.raises(TeamConfigError, match="requires 'command'"):
        load_team(p)


def test_mcp_server_bad_transport(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        """
        name: t1
        goal: g
        mcp_servers:
          x:
            transport: carrier-pigeon
            command: foo
        members:
          - {name: a, role: W, model: m, persona: p}
        """,
    )
    with pytest.raises(TeamConfigError, match="transport must be"):
        load_team(p)


def test_qualified_tool_patterns_accepted(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        """
        name: t1
        goal: g
        defaults:
          tools: [code/*, workspace/read_file]
        members:
          - name: a
            role: W
            model: m
            persona: p
            tools: [web/*, github/search_repositories]
        """,
    )
    cfg = load_team(p)
    assert cfg.defaults.tools == ["code/*", "workspace/read_file"]
    assert cfg.members[0].tools == ["web/*", "github/search_repositories"]


def test_bare_tool_name_rejected(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        """
        name: t1
        goal: g
        defaults:
          tools: [web_search]
        members:
          - {name: a, role: W, model: m, persona: p}
        """,
    )
    with pytest.raises(TeamConfigError, match="web/web_search"):
        load_team(p)


def test_extra_context_parsed(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        """
        name: t1
        goal: g
        defaults:
          extra_context: [context/checklist.md]
        members:
          - name: a
            role: W
            model: m
            persona: p
            extra_context: [context/extra.md]
        """,
    )
    cfg = load_team(p)
    assert cfg.defaults.extra_context == ["context/checklist.md"]
    assert cfg.members[0].extra_context == ["context/extra.md"]


def test_expand_env(monkeypatch) -> None:
    from team.config import expand_env

    monkeypatch.setenv("MY_TOKEN", "secret")
    assert expand_env("env:MY_TOKEN") == "secret"
    assert expand_env("env:MISSING", default="") == ""
    assert expand_env("literal") == "literal"
    assert expand_env(None) is None
