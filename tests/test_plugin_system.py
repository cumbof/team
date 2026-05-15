"""Tests for the entry-point plugin system.

Covers:
- ``team.skills`` entry-point scanning (_load_skill_registry, _load_skill_by_name)
- ``team.persona_dirs`` entry-point scanning (_persona_dirs)
- ``team.commands`` CLI plugin loading (_load_plugin_commands / cli)
- ``load_skill()`` dispatch for registered-name forms (plain string and ``name:`` dict)
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

import team.skills as skills_mod
from team.skills import SkillLoadError, load_skill


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _make_ep(name: str, value: str, load_return=None):
    """Build a fake entry-point object."""
    ep = MagicMock()
    ep.name = name
    ep.value = value
    if load_return is not None:
        ep.load.return_value = load_return
    return ep


def _reset_registry():
    """Force _load_skill_registry to re-scan on next call."""
    skills_mod._SKILL_REGISTRY = None


# --------------------------------------------------------------------------- #
# _load_skill_registry
# --------------------------------------------------------------------------- #


class TestLoadSkillRegistry:
    def setup_method(self):
        _reset_registry()

    def teardown_method(self):
        _reset_registry()

    def test_returns_empty_when_no_eps(self):
        with patch("team.skills.entry_points", return_value=[]):
            registry = skills_mod._load_skill_registry()
        assert registry == {}

    def test_maps_ep_names_to_values(self):
        eps = [
            _make_ep("bioblend", "team_galaxy.skills.bioblend"),
            _make_ep("planemo", "team_galaxy.skills.planemo"),
        ]
        with patch("team.skills.entry_points", return_value=eps):
            registry = skills_mod._load_skill_registry()
        assert registry["bioblend"] == "team_galaxy.skills.bioblend"
        assert registry["planemo"] == "team_galaxy.skills.planemo"

    def test_result_is_cached(self):
        ep = _make_ep("s1", "some.module")
        with patch("team.skills.entry_points", return_value=[ep]) as m:
            skills_mod._load_skill_registry()
            skills_mod._load_skill_registry()
        assert m.call_count == 1  # entry_points only called once

    def test_importlib_error_yields_empty(self):
        # If importlib.metadata raises, we get an empty dict (graceful degradation).
        with patch("team.skills.entry_points", side_effect=RuntimeError("no metadata")):
            registry = skills_mod._load_skill_registry()
        assert registry == {}


# --------------------------------------------------------------------------- #
# _load_skill_by_name
# --------------------------------------------------------------------------- #


class TestLoadSkillByName:
    def setup_method(self):
        _reset_registry()

    def teardown_method(self):
        _reset_registry()

    def test_raises_on_unknown_name(self):
        with patch("team.skills.entry_points", return_value=[]):
            with pytest.raises(SkillLoadError, match="unknown skill name"):
                skills_mod._load_skill_by_name("no_such_skill")

    def test_raises_on_import_failure(self):
        ep = _make_ep("broken", "definitely.not.a.real.module.xyz")
        with patch("team.skills.entry_points", return_value=[ep]):
            with pytest.raises(SkillLoadError, match="failed to import"):
                skills_mod._load_skill_by_name("broken")

    def test_loads_py_skill_module(self, tmp_path):
        """A registered skill that is a normal .py skill file can be loaded."""
        skill_file = tmp_path / "my_skill.py"
        skill_file.write_text(textwrap.dedent("""\
            TOOL_NAME = "my_tool"
            TOOL_DESCRIPTION = "A test tool."

            def execute(body, **kwargs):
                return "result:" + body
        """), encoding="utf-8")

        mod = ModuleType("fake_skill_module")
        mod.__file__ = str(skill_file)

        ep = _make_ep("my_skill", "fake_skill_module")
        with patch("team.skills.entry_points", return_value=[ep]):
            with patch("importlib.import_module", return_value=mod):
                tools, descs, ctx = skills_mod._load_skill_by_name("my_skill")

        assert "my_tool" in tools
        assert tools["my_tool"]("hello") == "result:hello"
        assert ctx == []

    def test_loads_md_skill_via_skill_file_attr(self, tmp_path):
        """A module that sets SKILL_FILE to a .md path produces injected context."""
        md_file = tmp_path / "context.md"
        md_file.write_text("# Galaxy context\nSome background.", encoding="utf-8")

        mod = ModuleType("md_skill_module")
        mod.__file__ = str(tmp_path / "md_skill_module.py")
        mod.SKILL_FILE = str(md_file)

        ep = _make_ep("my_context", "md_skill_module")
        with patch("team.skills.entry_points", return_value=[ep]):
            with patch("importlib.import_module", return_value=mod):
                tools, descs, ctx = skills_mod._load_skill_by_name("my_context")

        assert tools == {}
        assert descs == {}
        assert len(ctx) == 1
        assert "Galaxy context" in ctx[0]

    def test_empty_md_file_returns_empty_context(self, tmp_path):
        md_file = tmp_path / "empty.md"
        md_file.write_text("   \n  ", encoding="utf-8")

        mod = ModuleType("empty_md_mod")
        mod.__file__ = str(tmp_path / "empty_md_mod.py")
        mod.SKILL_FILE = str(md_file)

        ep = _make_ep("empty_skill", "empty_md_mod")
        with patch("team.skills.entry_points", return_value=[ep]):
            with patch("importlib.import_module", return_value=mod):
                tools, descs, ctx = skills_mod._load_skill_by_name("empty_skill")

        assert ctx == []


# --------------------------------------------------------------------------- #
# load_skill dispatch for plain-name and name-dict forms
# --------------------------------------------------------------------------- #


class TestLoadSkillDispatch:
    def setup_method(self):
        _reset_registry()

    def teardown_method(self):
        _reset_registry()

    def _registry_with_tool(self, tmp_path, skill_name="mysimple"):
        """Return (ep list, module) for a one-tool skill registered as skill_name."""
        sf = tmp_path / "simple.py"
        sf.write_text(textwrap.dedent("""\
            TOOL_NAME = "simple"
            TOOL_DESCRIPTION = "Simple tool."
            def execute(body, **kwargs):
                return "ok"
        """), encoding="utf-8")
        mod = ModuleType("simple_mod")
        mod.__file__ = str(sf)
        ep = _make_ep(skill_name, "simple_mod")
        return [ep], mod

    def test_plain_string_name_resolved_via_registry(self, tmp_path):
        eps, mod = self._registry_with_tool(tmp_path)
        with patch("team.skills.entry_points", return_value=eps):
            with patch("importlib.import_module", return_value=mod):
                tools, _, _ = load_skill("mysimple")
        assert "simple" in tools

    def test_dict_name_key_resolved_via_registry(self, tmp_path):
        eps, mod = self._registry_with_tool(tmp_path)
        with patch("team.skills.entry_points", return_value=eps):
            with patch("importlib.import_module", return_value=mod):
                tools, _, _ = load_skill({"name": "mysimple"})
        assert "simple" in tools

    def test_plain_path_string_still_works(self, tmp_path):
        sf = tmp_path / "direct.py"
        sf.write_text(textwrap.dedent("""\
            TOOL_NAME = "direct"
            TOOL_DESCRIPTION = "Direct."
            def execute(body, **kwargs):
                return "direct"
        """), encoding="utf-8")
        tools, _, _ = load_skill(str(sf))
        assert "direct" in tools

    def test_dict_path_key_still_works(self, tmp_path):
        sf = tmp_path / "pathed.py"
        sf.write_text(textwrap.dedent("""\
            TOOL_NAME = "pathed"
            TOOL_DESCRIPTION = "Pathed."
            def execute(body, **kwargs):
                return "pathed"
        """), encoding="utf-8")
        tools, _, _ = load_skill({"path": str(sf)})
        assert "pathed" in tools

    def test_dict_with_multiple_keys_raises(self, tmp_path):
        with pytest.raises(SkillLoadError, match="exactly one of"):
            load_skill({"path": "/some/path.py", "url": "http://example.com"})

    def test_dict_with_name_and_path_raises(self, tmp_path):
        with pytest.raises(SkillLoadError, match="exactly one of"):
            load_skill({"name": "myskill", "path": "/some/path.py"})

    def test_unknown_name_raises(self):
        with patch("team.skills.entry_points", return_value=[]):
            with pytest.raises(SkillLoadError, match="unknown skill name"):
                load_skill("totally_unknown_skill_xyz")

    def test_dict_missing_all_keys_raises(self):
        with pytest.raises(SkillLoadError, match="'name', 'path', or 'url'"):
            load_skill({"checksum": "abc123"})

    def test_checksum_warning_for_name_form(self, tmp_path, caplog):
        import logging
        eps, mod = self._registry_with_tool(tmp_path)
        with patch("team.skills.entry_points", return_value=eps):
            with patch("importlib.import_module", return_value=mod):
                with caplog.at_level(logging.WARNING, logger="team.skills"):
                    load_skill({"name": "mysimple", "checksum": "sha256:abc"})
        assert "checksum is not supported" in caplog.text


# --------------------------------------------------------------------------- #
# _persona_dirs entry-point scanning
# --------------------------------------------------------------------------- #


class TestPersonaDirsEntryPoints:
    def test_ep_dir_included_between_env_and_builtin(self, tmp_path, monkeypatch):
        from team.persona_library import _persona_dirs

        ep_dir = tmp_path / "plugin_personas"
        ep_dir.mkdir()

        monkeypatch.delenv("TEAM_PERSONA_DIR", raising=False)

        def get_ep_dir():
            return ep_dir

        ep = _make_ep("mypkg", "mypkg:personas_dir", load_return=get_ep_dir)

        with patch("team.persona_library.entry_points", return_value=[ep]):
            dirs = list(_persona_dirs())

        assert ep_dir in dirs

    def test_env_var_dir_takes_priority(self, tmp_path, monkeypatch):
        from team.persona_library import _persona_dirs

        env_dir = tmp_path / "env_personas"
        env_dir.mkdir()
        ep_dir = tmp_path / "plugin_personas"
        ep_dir.mkdir()

        monkeypatch.setenv("TEAM_PERSONA_DIR", str(env_dir))

        def get_ep_dir():
            return ep_dir

        ep = _make_ep("mypkg", "mypkg:personas_dir", load_return=get_ep_dir)

        with patch("team.persona_library.entry_points", return_value=[ep]):
            dirs = list(_persona_dirs())

        assert dirs[0] == env_dir  # env dir first

    def test_no_ep_dirs_when_none_registered(self, tmp_path, monkeypatch):
        from team.persona_library import _persona_dirs

        monkeypatch.delenv("TEAM_PERSONA_DIR", raising=False)

        with patch("team.persona_library.entry_points", return_value=[]):
            dirs = list(_persona_dirs())

        # Should still contain the built-in personas dir
        assert len(dirs) >= 1

    def test_bad_ep_loader_is_skipped(self, tmp_path, monkeypatch):
        from team.persona_library import _persona_dirs

        monkeypatch.delenv("TEAM_PERSONA_DIR", raising=False)

        ep = _make_ep("bad_pkg", "bad_pkg:personas_dir")
        ep.load.side_effect = ImportError("module not found")

        with patch("team.persona_library.entry_points", return_value=[ep]):
            dirs = list(_persona_dirs())  # Should not raise

        assert isinstance(dirs, list)


# --------------------------------------------------------------------------- #
# CLI plugin loading
# --------------------------------------------------------------------------- #


class TestCLIPluginLoading:
    def test_valid_command_added_to_cli(self):
        import click
        from team.cli import cli

        @click.group()
        def my_plugin_group():
            """My plugin group."""

        ep = _make_ep("myplugin", "mypkg.commands:myplugin", load_return=my_plugin_group)

        # Temporarily remove if already added (idempotency guard)
        cli.commands.pop("myplugin", None)

        with patch("team.cli.entry_points", return_value=[ep]):
            from team.cli import _load_plugin_commands
            _load_plugin_commands()

        assert "myplugin" in cli.commands
        # Clean up
        cli.commands.pop("myplugin", None)

    def test_bad_ep_loader_does_not_crash_cli(self):
        from team.cli import _load_plugin_commands

        ep = _make_ep("bad", "bad.module:cmd")
        ep.load.side_effect = ImportError("not found")

        with patch("team.cli.entry_points", return_value=[ep]):
            _load_plugin_commands()  # Should not raise
