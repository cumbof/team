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

