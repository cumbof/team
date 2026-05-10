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
