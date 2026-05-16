"""Tests for team/extension.py — the make_extension_commands factory.

Covers:
- Returned group name, description, and version option
- All four subcommands: init, scenarios, skills, personas
- Skill type detection (_skill_type helper)
- Persona YAML loading (_persona_info helper)
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml
from click.testing import CliRunner

from team.extension import _persona_info, _skill_type, make_extension_commands


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture()
def ext_dirs(tmp_path: Path):
    """Create minimal skills/, personas/, and examples/ directories."""
    sd = tmp_path / "skills"
    pd = tmp_path / "personas"
    ed = tmp_path / "examples"
    sd.mkdir()
    pd.mkdir()
    ed.mkdir()
    return sd, pd, ed


@pytest.fixture()
def simple_group(ext_dirs):
    """A make_extension_commands group with minimal config."""
    sd, pd, ed = ext_dirs
    return make_extension_commands(
        package_name="team-test",
        group_name="mytest",
        description="Test extension.",
        skills_dir=lambda: sd,
        personas_dir=lambda: pd,
        examples_dir=lambda: ed,
        skill_descriptions={"tool_a": "Tool A.", "ctx_b": "Context B."},
        scenario_descriptions={"scenario-x": "Scenario X."},
    )


# --------------------------------------------------------------------------- #
# _skill_type
# --------------------------------------------------------------------------- #


class TestSkillType:
    def test_py_file_is_python_tools(self, tmp_path):
        (tmp_path / "my_tool.py").write_text("TOOL_NAME = 'x'\n")
        assert _skill_type("my_tool", tmp_path) == "Python (tools)"

    def test_md_file_is_markdown_context(self, tmp_path):
        (tmp_path / "my_ctx.md").write_text("# context")
        assert _skill_type("my_ctx", tmp_path) == "Markdown (context)"

    def test_skill_wrapper_py_is_markdown_context(self, tmp_path):
        (tmp_path / "my_ctx_skill.py").write_text("SKILL_FILE = 'x.md'")
        assert _skill_type("my_ctx", tmp_path) == "Markdown (context)"

    def test_unknown_name_returns_registered(self, tmp_path):
        assert _skill_type("no_file_here", tmp_path) == "Registered"


# --------------------------------------------------------------------------- #
# _persona_info
# --------------------------------------------------------------------------- #


class TestPersonaInfo:
    def test_reads_role_and_description(self, tmp_path):
        p = tmp_path / "my.yaml"
        p.write_text("role: Engineer\ndescription: An engineer.\npersona: You are.\n")
        role, desc = _persona_info(p)
        assert role == "Engineer"
        assert "engineer" in desc.lower()

    def test_missing_keys_return_empty(self, tmp_path):
        p = tmp_path / "sparse.yaml"
        p.write_text("persona: You are.")
        role, desc = _persona_info(p)
        assert role == ""
        assert desc == ""

    def test_bad_yaml_returns_empty(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text(": : :")
        role, desc = _persona_info(p)
        assert role == ""
        assert desc == ""


# --------------------------------------------------------------------------- #
# make_extension_commands — group metadata
# --------------------------------------------------------------------------- #


class TestGroupMetadata:
    def test_group_name(self, simple_group):
        assert simple_group.name == "mytest"

    def test_help_shows_description(self, simple_group):
        result = CliRunner().invoke(simple_group, ["--help"])
        assert result.exit_code == 0
        assert "Test extension" in result.output

    def test_subcommands_present(self, simple_group):
        assert "init" in simple_group.commands
        assert "scenarios" in simple_group.commands
        assert "skills" in simple_group.commands
        assert "personas" in simple_group.commands

    def test_no_extra_subcommands(self, simple_group):
        assert set(simple_group.commands.keys()) == {"init", "scenarios", "skills", "personas"}


# --------------------------------------------------------------------------- #
# scenarios subcommand
# --------------------------------------------------------------------------- #


class TestScenariosCmd:
    def test_lists_yaml_stems(self, ext_dirs):
        sd, pd, ed = ext_dirs
        (ed / "alpha.yaml").write_text("name: alpha\n")
        (ed / "beta.yaml").write_text("name: beta\n")
        grp = make_extension_commands(
            package_name="team-t", group_name="t", description="T",
            skills_dir=lambda: sd, personas_dir=lambda: pd, examples_dir=lambda: ed,
        )
        result = CliRunner().invoke(grp, ["scenarios"])
        assert result.exit_code == 0
        assert "alpha" in result.output
        assert "beta" in result.output

    def test_shows_descriptions(self, ext_dirs):
        sd, pd, ed = ext_dirs
        (ed / "alpha.yaml").write_text("name: alpha\n")
        grp = make_extension_commands(
            package_name="team-t", group_name="t", description="T",
            skills_dir=lambda: sd, personas_dir=lambda: pd, examples_dir=lambda: ed,
            scenario_descriptions={"alpha": "Does alpha things."},
        )
        result = CliRunner().invoke(grp, ["scenarios"])
        assert "Does alpha things" in result.output

    def test_empty_examples_dir(self, ext_dirs):
        sd, pd, ed = ext_dirs
        grp = make_extension_commands(
            package_name="team-t", group_name="t", description="T",
            skills_dir=lambda: sd, personas_dir=lambda: pd, examples_dir=lambda: ed,
        )
        result = CliRunner().invoke(grp, ["scenarios"])
        assert result.exit_code == 0


# --------------------------------------------------------------------------- #
# skills subcommand
# --------------------------------------------------------------------------- #


class TestSkillsCmd:
    def test_lists_all_registered_skills(self, simple_group):
        result = CliRunner().invoke(simple_group, ["skills"])
        assert result.exit_code == 0
        assert "tool_a" in result.output
        assert "ctx_b" in result.output

    def test_shows_descriptions(self, simple_group):
        result = CliRunner().invoke(simple_group, ["skills"])
        assert "Tool A." in result.output
        assert "Context B." in result.output

    def test_empty_skill_descriptions(self, ext_dirs):
        sd, pd, ed = ext_dirs
        grp = make_extension_commands(
            package_name="team-t", group_name="t", description="T",
            skills_dir=lambda: sd, personas_dir=lambda: pd, examples_dir=lambda: ed,
        )
        result = CliRunner().invoke(grp, ["skills"])
        assert result.exit_code == 0

    def test_skill_type_python_detected(self, ext_dirs):
        sd, pd, ed = ext_dirs
        (sd / "my_tool.py").write_text("TOOL_NAME = 'x'\n")
        grp = make_extension_commands(
            package_name="team-t", group_name="t", description="T",
            skills_dir=lambda: sd, personas_dir=lambda: pd, examples_dir=lambda: ed,
            skill_descriptions={"my_tool": "A tool."},
        )
        result = CliRunner().invoke(grp, ["skills"])
        assert "Python (tools)" in result.output

    def test_skill_type_markdown_detected(self, ext_dirs):
        sd, pd, ed = ext_dirs
        (sd / "my_ctx.md").write_text("# ctx")
        grp = make_extension_commands(
            package_name="team-t", group_name="t", description="T",
            skills_dir=lambda: sd, personas_dir=lambda: pd, examples_dir=lambda: ed,
            skill_descriptions={"my_ctx": "Context."},
        )
        result = CliRunner().invoke(grp, ["skills"])
        assert "Markdown (context)" in result.output


# --------------------------------------------------------------------------- #
# personas subcommand
# --------------------------------------------------------------------------- #


class TestPersonasCmd:
    def test_lists_persona_yamls(self, ext_dirs):
        sd, pd, ed = ext_dirs
        (pd / "wizard.yaml").write_text(
            "role: Wizard\ndescription: A wizard.\npersona: You are a wizard.\n"
        )
        grp = make_extension_commands(
            package_name="team-t", group_name="t", description="T",
            skills_dir=lambda: sd, personas_dir=lambda: pd, examples_dir=lambda: ed,
        )
        result = CliRunner().invoke(grp, ["personas"])
        assert result.exit_code == 0
        assert "@wizard" in result.output
        assert "Wizard" in result.output

    def test_empty_personas_dir(self, ext_dirs):
        sd, pd, ed = ext_dirs
        grp = make_extension_commands(
            package_name="team-t", group_name="t", description="T",
            skills_dir=lambda: sd, personas_dir=lambda: pd, examples_dir=lambda: ed,
        )
        result = CliRunner().invoke(grp, ["personas"])
        assert result.exit_code == 0


# --------------------------------------------------------------------------- #
# init subcommand
# --------------------------------------------------------------------------- #


class TestInitCmd:
    def _grp_with_scenario(self, ext_dirs):
        sd, pd, ed = ext_dirs
        (ed / "demo.yaml").write_text(textwrap.dedent("""\
            name: demo
            goal: test
            workflow:
              type: round_robin
              max_rounds: 1
            members:
              - name: a
                role: A
                model: x
        """))
        return make_extension_commands(
            package_name="team-t", group_name="t", description="T",
            skills_dir=lambda: sd, personas_dir=lambda: pd, examples_dir=lambda: ed,
        )

    def test_creates_yaml_file(self, ext_dirs, tmp_path):
        grp = self._grp_with_scenario(ext_dirs)
        result = CliRunner().invoke(grp, ["init", "--scenario", "demo", "--output-dir", str(tmp_path)])
        assert result.exit_code == 0
        assert (tmp_path / "demo.yaml").is_file()

    def test_content_matches_template(self, ext_dirs, tmp_path):
        grp = self._grp_with_scenario(ext_dirs)
        CliRunner().invoke(grp, ["init", "--scenario", "demo", "--output-dir", str(tmp_path)])
        content = (tmp_path / "demo.yaml").read_text()
        assert "name: demo" in content

    def test_refuses_overwrite_without_force(self, ext_dirs, tmp_path):
        grp = self._grp_with_scenario(ext_dirs)
        (tmp_path / "demo.yaml").write_text("existing")
        result = CliRunner().invoke(grp, ["init", "--scenario", "demo", "--output-dir", str(tmp_path)])
        assert result.exit_code != 0
        assert "already" in result.output and "--force" in result.output
        assert (tmp_path / "demo.yaml").read_text() == "existing"

    def test_force_overwrites(self, ext_dirs, tmp_path):
        grp = self._grp_with_scenario(ext_dirs)
        (tmp_path / "demo.yaml").write_text("existing")
        result = CliRunner().invoke(
            grp, ["init", "--scenario", "demo", "--output-dir", str(tmp_path), "--force"]
        )
        assert result.exit_code == 0
        assert (tmp_path / "demo.yaml").read_text() != "existing"

    def test_missing_scenario_exits_nonzero(self, ext_dirs, tmp_path):
        sd, pd, ed = ext_dirs
        grp = make_extension_commands(
            package_name="team-t", group_name="t", description="T",
            skills_dir=lambda: sd, personas_dir=lambda: pd, examples_dir=lambda: ed,
        )
        # No scenarios exist — Choice validator will reject it
        result = CliRunner().invoke(grp, ["init", "--scenario", "nonexistent", "--output-dir", str(tmp_path)])
        assert result.exit_code != 0
