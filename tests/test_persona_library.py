"""Tests for the predefined persona library and its YAML integration."""
from __future__ import annotations

import os
import textwrap

import pytest

import team.persona_library as _lib
from team.persona_library import PERSONAS, list_all, resolve


def _reset_cache():
    """Clear the module-level persona cache so tests start fresh."""
    _lib._CACHE = None


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
        assert len(persona) > 50

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
        assert all("key" in i and "role" in i and "description" in i and "source" in i for i in items)

    def test_list_all_contains_all_keys(self):
        keys = {i["key"] for i in list_all()}
        assert set(PERSONAS.keys()) == keys

    def test_no_duplicate_roles(self):
        roles = [v["role"] for v in PERSONAS.values()]
        assert len(roles) == len(set(roles)), "Duplicate role names found"

    def test_personas_loaded_from_yaml_files(self):
        """Verify personas come from YAML files, not a hardcoded dict."""
        from team.persona_library import _BUILTIN_DIR
        assert _BUILTIN_DIR.is_dir(), f"personas/ dir missing at {_BUILTIN_DIR}"
        yaml_keys = {p.stem for p in _BUILTIN_DIR.glob("*.yaml")}
        assert "pi" in yaml_keys
        assert "engineer" in yaml_keys
        # Every YAML file must appear in the loaded library
        for key in yaml_keys:
            assert key in PERSONAS, f"{key}.yaml exists but not loaded"

    def test_builtin_dir_has_expected_files(self):
        from team.persona_library import _BUILTIN_DIR
        files = {p.stem for p in _BUILTIN_DIR.glob("*.yaml")}
        expected = {
            "pi", "postdoc", "phd", "reviewer", "statistician",
            "bioinformatician", "ml_researcher", "architect", "engineer",
            "qa", "devops", "tech_writer", "analyst", "writer",
            "manager", "ethicist",
        }
        assert expected == files


# --------------------------------------------------------------------------- #
# Custom persona directory (TEAM_PERSONA_DIR)
# --------------------------------------------------------------------------- #


class TestCustomPersonaDir:
    def test_custom_dir_adds_new_persona(self, tmp_path, monkeypatch):
        persona_dir = tmp_path / "my_personas"
        persona_dir.mkdir()
        (persona_dir / "chemist.yaml").write_text(
            "role: Chemist\ndescription: Synthesis expert.\npersona: You are a chemist.\n"
        )
        _reset_cache()
        monkeypatch.setenv("TEAM_PERSONA_DIR", str(persona_dir))
        try:
            assert "chemist" in _lib._load()
            role, persona = resolve("chemist")
            assert role == "Chemist"
            assert "chemist" in persona.lower()
        finally:
            _reset_cache()

    def test_custom_dir_overrides_builtin(self, tmp_path, monkeypatch):
        persona_dir = tmp_path / "my_personas"
        persona_dir.mkdir()
        (persona_dir / "pi.yaml").write_text(
            "role: Lab Chief\ndescription: Custom PI.\npersona: You are the custom PI.\n"
        )
        _reset_cache()
        monkeypatch.setenv("TEAM_PERSONA_DIR", str(persona_dir))
        try:
            role, persona = resolve("pi")
            assert role == "Lab Chief"
            assert "custom PI" in persona
        finally:
            _reset_cache()

    def test_invalid_yaml_file_is_skipped(self, tmp_path, monkeypatch):
        persona_dir = tmp_path / "my_personas"
        persona_dir.mkdir()
        (persona_dir / "broken.yaml").write_text(": [unclosed")
        _reset_cache()
        monkeypatch.setenv("TEAM_PERSONA_DIR", str(persona_dir))
        try:
            import warnings
            with warnings.catch_warnings(record=True):
                loaded = _lib._load()
            # broken.yaml should be silently skipped
            assert "broken" not in loaded
        finally:
            _reset_cache()

    def test_missing_fields_skipped_with_warning(self, tmp_path, monkeypatch):
        persona_dir = tmp_path / "my_personas"
        persona_dir.mkdir()
        (persona_dir / "incomplete.yaml").write_text("role: Foo\n")
        _reset_cache()
        monkeypatch.setenv("TEAM_PERSONA_DIR", str(persona_dir))
        try:
            import warnings
            with warnings.catch_warnings(record=True):
                loaded = _lib._load()
            assert "incomplete" not in loaded
        finally:
            _reset_cache()

    def test_nonexistent_custom_dir_is_ignored(self, monkeypatch):
        _reset_cache()
        monkeypatch.setenv("TEAM_PERSONA_DIR", "/nonexistent_xyz_dir")
        try:
            loaded = _lib._load()
            # Built-in personas still load fine
            assert "pi" in loaded
        finally:
            _reset_cache()

    def test_source_field_present_in_loaded_personas(self):
        loaded = _lib._load()
        for key, entry in loaded.items():
            assert "source" in entry, f"@{key}: missing 'source' field"

    def test_builtin_personas_have_builtin_source(self):
        _reset_cache()
        try:
            loaded = _lib._load()
            assert loaded["pi"]["source"] == "built-in"
        finally:
            _reset_cache()

    def test_env_dir_personas_have_env_source(self, tmp_path, monkeypatch):
        persona_dir = tmp_path / "custom_personas"
        persona_dir.mkdir()
        (persona_dir / "testpersona.yaml").write_text(
            "role: Tester\ndescription: Test persona.\npersona: You are a tester.\n"
        )
        _reset_cache()
        monkeypatch.setenv("TEAM_PERSONA_DIR", str(persona_dir))
        try:
            loaded = _lib._load()
            assert loaded["testpersona"]["source"] == "env (TEAM_PERSONA_DIR)"
        finally:
            _reset_cache()

    def test_conflict_emits_warning(self, tmp_path, monkeypatch, caplog):
        import logging
        persona_dir = tmp_path / "custom_personas"
        persona_dir.mkdir()
        (persona_dir / "pi.yaml").write_text(
            "role: Custom PI\ndescription: Overridden PI.\npersona: You are the custom PI.\n"
        )
        _reset_cache()
        monkeypatch.setenv("TEAM_PERSONA_DIR", str(persona_dir))
        try:
            with caplog.at_level(logging.WARNING, logger="team.persona_library"):
                _lib._load()
            assert any("@pi" in r.message and "overrides" in r.message for r in caplog.records)
        finally:
            _reset_cache()

    def test_list_all_has_source_field(self):
        items = list_all()
        for item in items:
            assert "source" in item, f"list_all() entry for @{item.get('key')} missing 'source'"


# --------------------------------------------------------------------------- #
# YAML config integration (@-shorthand)
# --------------------------------------------------------------------------- #


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
        # Persona text still comes from library
        assert len(cfg.member("alice").persona) > 50

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
        assert "@" in result.output

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
        assert "nonexistent_xyz" in result.output

    def test_list_output_has_all_keys(self):
        from click.testing import CliRunner
        from team.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["personas"])
        assert result.exit_code == 0
        for key in PERSONAS:
            assert key in result.output, f"Key {key!r} not in personas list output"



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
        assert all("key" in i and "role" in i and "description" in i and "source" in i for i in items)

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
        assert "@pi" in result.output
        # Role may wrap across lines in a narrow terminal — check words separately
        assert "Principal" in result.output
        assert "Investigator" in result.output
        assert "engineer" in result.output
        assert "built-in" in result.output  # Source column present
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
