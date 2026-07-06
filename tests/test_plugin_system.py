"""Tests for the entry-point plugin system.

Covers:
- ``team.mcp_servers`` entry-point resolution (orchestrator._resolve_entry_point)
- ``team.persona_dirs`` entry-point scanning (_persona_dirs)
- ``team.commands`` CLI plugin loading (_load_plugin_commands / cli)
"""

from __future__ import annotations

from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from team.orchestrator import _resolve_entry_point


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


# --------------------------------------------------------------------------- #
# _resolve_entry_point (team.mcp_servers)
# --------------------------------------------------------------------------- #


class TestResolveEntryPoint:
    def test_module_attr_form(self):
        fake_server = MagicMock(name="fastmcp")
        factory = MagicMock(return_value=fake_server)
        mod = ModuleType("fake_srv_mod")
        mod.build = factory
        with patch("importlib.import_module", return_value=mod):
            result = _resolve_entry_point("fake_srv_mod:build")
        assert result is fake_server
        factory.assert_called_once()

    def test_registered_name_form(self):
        fake_server = MagicMock(name="fastmcp")
        factory = MagicMock(return_value=fake_server)
        ep = _make_ep("helpers", "pkg.servers:build", load_return=factory)
        with patch("importlib.metadata.entry_points", return_value=[ep]):
            result = _resolve_entry_point("helpers")
        assert result is fake_server

    def test_unknown_name_raises(self):
        with patch("importlib.metadata.entry_points", return_value=[]):
            with pytest.raises(ValueError, match="team.mcp_servers"):
                _resolve_entry_point("totally_unknown_server")


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

        assert ep_dir in [d for d, _ in dirs]

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

        assert dirs[0][0] == env_dir  # env dir first

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
