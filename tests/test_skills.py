"""Tests for team/skills.py — the Markdown-only ``team.skills`` registry."""

from __future__ import annotations

import sys
import textwrap
from importlib.metadata import EntryPoint
from pathlib import Path

import pytest

import team.skills as skills_mod
from team.skills import SkillLoadError, resolve_skill


@pytest.fixture(autouse=True)
def _reset_registry():
    """Force a fresh entry-point scan for every test."""
    skills_mod._REGISTRY = None
    yield
    skills_mod._REGISTRY = None


def _install_skill_module(tmp_path: Path, module_name: str, skill_file: Path) -> None:
    """Write a thin wrapper module setting SKILL_FILE and put it on sys.path."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / f"{module_name}.py").write_text(
        f'SKILL_FILE = {str(skill_file)!r}\n', encoding="utf-8"
    )
    sys.path.insert(0, str(tmp_path))


@pytest.fixture()
def cleanup_sys_path():
    before = list(sys.path)
    yield
    sys.path[:] = before


# --------------------------------------------------------------------------- #
# _load_registry
# --------------------------------------------------------------------------- #


class TestLoadRegistry:
    def test_scans_team_skills_entry_points(self, monkeypatch):
        eps = [EntryPoint(name="review_checklist", value="myext.skills.review_checklist", group="team.skills")]
        monkeypatch.setattr(skills_mod, "entry_points", lambda group: eps if group == "team.skills" else [])
        registry = skills_mod._load_registry()
        assert registry == {"review_checklist": "myext.skills.review_checklist"}

    def test_caches_after_first_call(self, monkeypatch):
        calls = []

        def fake_entry_points(group):
            calls.append(group)
            return []

        monkeypatch.setattr(skills_mod, "entry_points", fake_entry_points)
        skills_mod._load_registry()
        skills_mod._load_registry()
        assert len(calls) == 1

    def test_empty_when_nothing_registered(self, monkeypatch):
        monkeypatch.setattr(skills_mod, "entry_points", lambda group: [])
        assert skills_mod._load_registry() == {}


# --------------------------------------------------------------------------- #
# resolve_skill
# --------------------------------------------------------------------------- #


class TestResolveSkill:
    def test_unregistered_name_raises(self):
        skills_mod._REGISTRY = {}
        with pytest.raises(SkillLoadError, match="no skill named 'missing' is registered"):
            resolve_skill("missing")

    def test_unregistered_name_lists_known(self):
        skills_mod._REGISTRY = {"a": "mod.a", "b": "mod.b"}
        with pytest.raises(SkillLoadError, match=r"Registered skills: \['a', 'b'\]"):
            resolve_skill("missing")

    def test_resolves_to_absolute_path(self, tmp_path, cleanup_sys_path):
        md = tmp_path / "review_checklist.md"
        md.write_text("# Checklist\n")
        _install_skill_module(tmp_path, "wrapper_ok", md)
        skills_mod._REGISTRY = {"review_checklist": "wrapper_ok"}
        result = resolve_skill("review_checklist")
        assert result == str(md)

    def test_import_error_raises_skill_load_error(self):
        skills_mod._REGISTRY = {"broken": "no_such_module_xyz"}
        with pytest.raises(SkillLoadError, match="failed to import"):
            resolve_skill("broken")

    def test_missing_skill_file_attr_raises(self, tmp_path, cleanup_sys_path):
        tmp_path.mkdir(parents=True, exist_ok=True)
        (tmp_path / "wrapper_no_attr.py").write_text("X = 1\n", encoding="utf-8")
        sys.path.insert(0, str(tmp_path))
        skills_mod._REGISTRY = {"no_attr": "wrapper_no_attr"}
        with pytest.raises(SkillLoadError, match="does not define SKILL_FILE"):
            resolve_skill("no_attr")

    def test_skill_file_pointing_at_missing_path_raises(self, tmp_path, cleanup_sys_path):
        _install_skill_module(tmp_path, "wrapper_missing_file", tmp_path / "nonexistent.md")
        skills_mod._REGISTRY = {"gone": "wrapper_missing_file"}
        with pytest.raises(SkillLoadError, match="points at a missing file"):
            resolve_skill("gone")


# --------------------------------------------------------------------------- #
# integration with member._load_extra_context
# --------------------------------------------------------------------------- #


class TestExtraContextFallback:
    def test_falls_back_to_registered_skill_when_path_missing(self, tmp_path, cleanup_sys_path, monkeypatch):
        from team.config import TeamConfig, Defaults, MemberConfig, WorkflowConfig
        from team.member import Member

        md = tmp_path / "checklist.md"
        md.write_text("# Review checklist\n- Check tests\n")
        _install_skill_module(tmp_path, "wrapper_integration", md)
        skills_mod._REGISTRY = {"checklist": "wrapper_integration"}

        team_yaml_dir = tmp_path / "proj"
        team_yaml_dir.mkdir()
        cfg = TeamConfig(
            name="t",
            goal="g",
            workspace=tmp_path / "ws",
            workflow=WorkflowConfig(type="round_robin", max_rounds=1),
            defaults=Defaults(extra_context=["checklist"]),
            members=[MemberConfig(name="a", role="A", model="x", persona="p")],
            source_path=team_yaml_dir / "team.yaml",
        )

        member_cfg = cfg.members[0]
        m = object.__new__(Member)
        m.config = member_cfg
        m.team = cfg
        loaded = Member._load_extra_context(m)
        assert any("Check tests" in c for c in loaded)

    def test_plain_path_still_takes_priority(self, tmp_path, cleanup_sys_path):
        from team.config import TeamConfig, Defaults, MemberConfig, WorkflowConfig
        from team.member import Member

        team_yaml_dir = tmp_path / "proj"
        team_yaml_dir.mkdir()
        (team_yaml_dir / "local.md").write_text("local content here\n")
        skills_mod._REGISTRY = {"local.md": "should_not_be_used"}

        cfg = TeamConfig(
            name="t",
            goal="g",
            workspace=tmp_path / "ws",
            workflow=WorkflowConfig(type="round_robin", max_rounds=1),
            defaults=Defaults(extra_context=["local.md"]),
            members=[MemberConfig(name="a", role="A", model="x", persona="p")],
            source_path=team_yaml_dir / "team.yaml",
        )

        member_cfg = cfg.members[0]
        m = object.__new__(Member)
        m.config = member_cfg
        m.team = cfg
        loaded = Member._load_extra_context(m)
        assert any("local content here" in c for c in loaded)
