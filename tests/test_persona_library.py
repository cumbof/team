"""Tests for the predefined persona library and its YAML integration."""
from __future__ import annotations

import textwrap

import pytest

from team.persona_library import PERSONAS, list_all, resolve


# --------------------------------------------------------------------------- #
# persona_library module
# --------------------------------------------------------------------------- #


class TestPersonaLibrary:
    def test_has_expected_keys(self):
        expected = {
            "pi", "postdoc", "phd", "reviewer", "statistician",
            "bioinformatician", "architect", "engineer", "qa", "devops",
            "tech_writer", "analyst", "writer", "manager", "ethicist",
            "ml_researcher",
        }
        assert expected.issubset(set(PERSONAS))

    def test_each_entry_has_required_fields(self):
        for key, entry in PERSONAS.items():
            assert "role" in entry, f"{key}: missing 'role'"
            assert "description" in entry, f"{key}: missing 'description'"
            assert "persona" in entry, f"{key}: missing 'persona'"
            assert entry["role"].strip(), f"{key}: empty 'role'"
            assert entry["description"].strip(), f"{key}: empty 'description'"
            assert len(entry["persona"]) > 50, f"{key}: persona text too short"

    def test_resolve_known_key(self):
        role, persona = resolve("pi")
        assert role == "Principal Investigator"
        assert "Principal Investigator" in persona or len(persona) > 50

    def test_resolve_engineer(self):
        role, persona = resolve("engineer")
        assert role == "Software Engineer"
        assert "code" in persona.lower() or "senior" in persona.lower()

    def test_resolve_unknown_key_raises(self):
        with pytest.raises(KeyError):
            resolve("nonexistent_persona_xyz")

    def test_list_all_returns_list_of_dicts(self):
        items = list_all()
        assert isinstance(items, list)
        assert all(isinstance(i, dict) for i in items)
        assert all("key" in i and "role" in i and "description" in i for i in items)

    def test_list_all_contains_all_keys(self):
        keys = {i["key"] for i in list_all()}
        assert set(PERSONAS.keys()) == keys

    def test_no_duplicate_roles(self):
        roles = [v["role"] for v in PERSONAS.values()]
        assert len(roles) == len(set(roles)), "Duplicate role names found"


# --------------------------------------------------------------------------- #
# YAML config integration (@-shorthand)
# --------------------------------------------------------------------------- #


def _make_yaml(persona_value: str, role_value: str | None = None) -> str:
    role_line = f"    role: {role_value}" if role_value else ""
    return textwrap.dedent(f"""\
        name: testteam
        goal: Test goal.
        workspace: /tmp/testteam_personas
        members:
          - name: alice
            model: llama3.1:8b
            persona: "{persona_value}"
        {role_line}
    """)


class TestPersonaShorthandConfig:
    def test_at_shorthand_expands_role_and_persona(self, tmp_path):
        from team.config import load_team

        yaml_text = textwrap.dedent("""\
            name: testteam
            goal: Test goal.
            workspace: /tmp/testteam_personas
            members:
              - name: alice
                model: llama3.1:8b
                persona: "@pi"
        """)
        p = tmp_path / "team.yaml"
        p.write_text(yaml_text)
        cfg = load_team(p)
        alice = cfg.member("alice")
        assert alice.role == "Principal Investigator"
        assert len(alice.persona) > 50
        assert alice.persona == PERSONAS["pi"]["persona"]

    def test_at_shorthand_role_override(self, tmp_path):
        from team.config import load_team

        yaml_text = textwrap.dedent("""\
            name: testteam
            goal: Test goal.
            workspace: /tmp/testteam_personas
            members:
              - name: alice
                model: llama3.1:8b
                persona: "@pi"
                role: "Lab Director"
        """)
        p = tmp_path / "team.yaml"
        p.write_text(yaml_text)
        cfg = load_team(p)
        assert cfg.member("alice").role == "Lab Director"
        # persona text must still come from library
        assert cfg.member("alice").persona == PERSONAS["pi"]["persona"]

    def test_at_shorthand_multiple_members(self, tmp_path):
        from team.config import load_team

        yaml_text = textwrap.dedent("""\
            name: testteam
            goal: Test goal.
            workspace: /tmp/testteam_personas
            members:
              - name: alice
                model: llama3.1:70b
                persona: "@pi"
              - name: bob
                model: llama3.1:8b
                persona: "@phd"
              - name: carol
                model: qwen2.5:7b
                persona: "@reviewer"
        """)
        p = tmp_path / "team.yaml"
        p.write_text(yaml_text)
        cfg = load_team(p)
        assert cfg.member("alice").role == "Principal Investigator"
        assert cfg.member("bob").role == "PhD Student"
        assert cfg.member("carol").role == "Critical Reviewer"

    def test_unknown_shorthand_raises_config_error(self, tmp_path):
        from team.config import TeamConfigError, load_team

        yaml_text = textwrap.dedent("""\
            name: testteam
            goal: Test goal.
            workspace: /tmp/testteam_personas
            members:
              - name: alice
                model: llama3.1:8b
                persona: "@nonexistent_xyz"
        """)
        p = tmp_path / "team.yaml"
        p.write_text(yaml_text)
        with pytest.raises(TeamConfigError, match="unknown persona library key"):
            load_team(p)

    def test_error_message_lists_available_keys(self, tmp_path):
        from team.config import TeamConfigError, load_team

        yaml_text = textwrap.dedent("""\
            name: testteam
            goal: Test goal.
            workspace: /tmp/testteam_personas
            members:
              - name: alice
                model: llama3.1:8b
                persona: "@badkey"
        """)
        p = tmp_path / "team.yaml"
        p.write_text(yaml_text)
        with pytest.raises(TeamConfigError) as exc_info:
            load_team(p)
        msg = str(exc_info.value)
        assert "Available:" in msg
        assert "pi" in msg

    def test_plain_persona_still_requires_role(self, tmp_path):
        from team.config import TeamConfigError, load_team

        yaml_text = textwrap.dedent("""\
            name: testteam
            goal: Test goal.
            workspace: /tmp/testteam_personas
            members:
              - name: alice
                model: llama3.1:8b
                persona: "I am a helper."
        """)
        p = tmp_path / "team.yaml"
        p.write_text(yaml_text)
        with pytest.raises(TeamConfigError, match="role"):
            load_team(p)

    def test_plain_persona_with_role_works(self, tmp_path):
        from team.config import load_team

        yaml_text = textwrap.dedent("""\
            name: testteam
            goal: Test goal.
            workspace: /tmp/testteam_personas
            members:
              - name: alice
                role: Helper
                model: llama3.1:8b
                persona: "I am a helpful assistant."
        """)
        p = tmp_path / "team.yaml"
        p.write_text(yaml_text)
        cfg = load_team(p)
        assert cfg.member("alice").role == "Helper"
        assert cfg.member("alice").persona == "I am a helpful assistant."

    def test_all_library_keys_load_without_error(self, tmp_path):
        """Every key in PERSONAS must be usable in a YAML config."""
        from team.config import load_team

        for key in PERSONAS:
            yaml_text = textwrap.dedent(f"""\
                name: testteam
                goal: Test goal.
                workspace: /tmp/testteam_personas
                members:
                  - name: alice
                    model: llama3.1:8b
                    persona: "@{key}"
            """)
            p = tmp_path / f"team_{key}.yaml"
            p.write_text(yaml_text)
            cfg = load_team(p)
            assert cfg.member("alice").role == PERSONAS[key]["role"]
            assert cfg.member("alice").persona == PERSONAS[key]["persona"]


# --------------------------------------------------------------------------- #
# CLI: team personas
# --------------------------------------------------------------------------- #


class TestPersonasCLI:
    def test_list_all_personas(self):
        from click.testing import CliRunner
        from team.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["personas"])
        assert result.exit_code == 0
        assert "pi" in result.output
        assert "Principal Investigator" in result.output
        assert "engineer" in result.output
        assert "persona:" in result.output.lower() or "@" in result.output

    def test_show_specific_persona(self):
        from click.testing import CliRunner
        from team.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["personas", "pi"])
        assert result.exit_code == 0
        assert "Principal Investigator" in result.output
        assert "@pi" in result.output

    def test_show_engineer_persona(self):
        from click.testing import CliRunner
        from team.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["personas", "engineer"])
        assert result.exit_code == 0
        assert "Software Engineer" in result.output

    def test_unknown_key_exits_nonzero(self):
        from click.testing import CliRunner
        from team.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["personas", "nonexistent_xyz"])
        assert result.exit_code != 0
        assert "Unknown persona key" in result.output or "nonexistent_xyz" in result.output

    def test_list_output_has_all_keys(self):
        from click.testing import CliRunner
        from team.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["personas"])
        assert result.exit_code == 0
        for key in PERSONAS:
            assert key in result.output, f"Key {key!r} not in personas list output"
