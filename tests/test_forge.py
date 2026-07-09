"""Tests for the ``team forge`` command scaffold."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from team.cli import cli, _forge_normalize, _forge_files


# --------------------------------------------------------------------------- #
# _forge_normalize
# --------------------------------------------------------------------------- #


class TestForgeNormalize:
    def test_plain_name(self):
        dir_name, pkg_name, ext_name = _forge_normalize("bioinformatics")
        assert dir_name == "team-bioinformatics"
        assert pkg_name == "team_bioinformatics"
        assert ext_name == "bioinformatics"

    def test_team_prefix_stripped(self):
        dir_name, pkg_name, ext_name = _forge_normalize("team-bioinformatics")
        assert dir_name == "team-bioinformatics"
        assert pkg_name == "team_bioinformatics"
        assert ext_name == "bioinformatics"

    def test_hyphens_converted_to_underscores(self):
        _, pkg_name, ext_name = _forge_normalize("my-ext")
        assert pkg_name == "team_my_ext"
        assert ext_name == "my_ext"

    def test_invalid_identifier_raises(self):
        import click
        with pytest.raises(click.BadParameter):
            _forge_normalize("123invalid")

    def test_spaces_treated_as_hyphens(self):
        dir_name, _, _ = _forge_normalize("my ext")
        assert dir_name == "team-my-ext"


# --------------------------------------------------------------------------- #
# _forge_files
# --------------------------------------------------------------------------- #


class TestForgeFiles:
    @pytest.fixture()
    def files(self):
        return _forge_files("team-myext", "team_myext", "myext")

    def test_expected_keys(self, files):
        expected = {
            "pyproject.toml",
            "team_myext/__init__.py",
            "team_myext/commands.py",
            "team_myext/servers/.gitkeep",
            "team_myext/skills/.gitkeep",
            "team_myext/personas/.gitkeep",
            "team_myext/examples/.gitkeep",
            "tests/__init__.py",
            "tests/test_package.py",
            "README.md",
            "LICENSE",
            ".gitignore",
        }
        assert set(files.keys()) == expected

    def test_pyproject_has_entry_points(self, files):
        toml = files["pyproject.toml"]
        assert 'team.persona_dirs' in toml
        assert 'team.mcp_servers' in toml
        assert 'team.skills' in toml
        assert 'team.commands' in toml

    def test_pyproject_package_name(self, files):
        assert '"team-myext"' in files["pyproject.toml"]

    def test_pyproject_dep_on_team_core(self, files):
        assert "team-core" in files["pyproject.toml"]

    def test_init_has_path_helpers(self, files):
        init = files["team_myext/__init__.py"]
        assert "def servers_dir" in init
        assert "def skills_dir" in init
        assert "def personas_dir" in init
        assert "def examples_dir" in init

    def test_commands_uses_factory(self, files):
        cmd = files["team_myext/commands.py"]
        assert "make_extension_commands" in cmd
        assert "from team.extension import make_extension_commands" in cmd

    def test_commands_has_correct_variable_name(self, files):
        cmd = files["team_myext/commands.py"]
        assert "myext = make_extension_commands(" in cmd

    def test_test_file_imports_correct_package(self, files):
        test = files["tests/test_package.py"]
        assert "from team_myext" in test
        assert "from team_myext.commands import myext" in test

    def test_license_is_mit(self, files):
        assert "MIT License" in files["LICENSE"]

    def test_gitignore_has_pycache(self, files):
        assert "__pycache__" in files[".gitignore"]

    def test_readme_has_package_name(self, files):
        assert "team-myext" in files["README.md"]

    def test_different_name_produces_correct_identifiers(self):
        files = _forge_files("team-galaxy-lite", "team_galaxy_lite", "galaxy_lite")
        assert "galaxy_lite = make_extension_commands(" in files["team_galaxy_lite/commands.py"]
        assert "from team_galaxy_lite import" in files["tests/test_package.py"]


# --------------------------------------------------------------------------- #
# team forge CLI
# --------------------------------------------------------------------------- #


class TestForgeCli:
    def test_creates_directory(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(cli, ["forge", "myext", "--output-dir", str(tmp_path)])
        assert result.exit_code == 0, result.output
        assert (tmp_path / "team-myext").is_dir()

    def test_all_files_created(self, tmp_path):
        runner = CliRunner()
        runner.invoke(cli, ["forge", "myext", "--output-dir", str(tmp_path)])
        root = tmp_path / "team-myext"
        assert (root / "pyproject.toml").is_file()
        assert (root / "team_myext" / "__init__.py").is_file()
        assert (root / "team_myext" / "commands.py").is_file()
        assert (root / "team_myext" / "servers" / ".gitkeep").is_file()
        assert (root / "team_myext" / "skills" / ".gitkeep").is_file()
        assert (root / "team_myext" / "personas" / ".gitkeep").is_file()
        assert (root / "team_myext" / "examples" / ".gitkeep").is_file()
        assert (root / "tests" / "test_package.py").is_file()
        assert (root / "README.md").is_file()
        assert (root / "LICENSE").is_file()

    def test_team_prefix_in_input_handled(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(cli, ["forge", "team-myext", "--output-dir", str(tmp_path)])
        assert result.exit_code == 0
        # Should not create team-team-myext
        assert (tmp_path / "team-myext").is_dir()
        assert not (tmp_path / "team-team-myext").exists()

    def test_refuses_overwrite_without_force(self, tmp_path):
        (tmp_path / "team-myext").mkdir()
        runner = CliRunner()
        result = runner.invoke(cli, ["forge", "myext", "--output-dir", str(tmp_path)])
        assert result.exit_code != 0
        assert "already exists" in result.output

    def test_force_overwrites(self, tmp_path):
        runner = CliRunner()
        runner.invoke(cli, ["forge", "myext", "--output-dir", str(tmp_path)])
        # Run again with --force
        result = runner.invoke(cli, ["forge", "myext", "--output-dir", str(tmp_path), "--force"])
        assert result.exit_code == 0

    def test_invalid_name_raises_usage_error(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(cli, ["forge", "123-invalid", "--output-dir", str(tmp_path)])
        assert result.exit_code != 0

    def test_output_is_parseable_pyproject(self, tmp_path):
        """pyproject.toml must be parseable with tomllib / tomli."""
        runner = CliRunner()
        runner.invoke(cli, ["forge", "myext", "--output-dir", str(tmp_path)])
        toml_path = tmp_path / "team-myext" / "pyproject.toml"
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib  # type: ignore[no-redef]
            except ImportError:
                pytest.skip("no toml parser available")
        data = tomllib.loads(toml_path.read_text(encoding="utf-8"))
        assert data["project"]["name"] == "team-myext"
        assert "team.commands" in data.get("project", {}).get("entry-points", {})

    def test_generated_commands_py_is_importable_in_isolation(self, tmp_path):
        """commands.py content should have no syntax errors."""
        runner = CliRunner()
        runner.invoke(cli, ["forge", "myext", "--output-dir", str(tmp_path)])
        cmd_path = tmp_path / "team-myext" / "team_myext" / "commands.py"
        code = cmd_path.read_text(encoding="utf-8")
        # Just compile — don't import (would need the package installed)
        compile(code, str(cmd_path), "exec")  # raises SyntaxError on bad code

    def test_generated_init_py_is_importable_in_isolation(self, tmp_path):
        runner = CliRunner()
        runner.invoke(cli, ["forge", "myext", "--output-dir", str(tmp_path)])
        init_path = tmp_path / "team-myext" / "team_myext" / "__init__.py"
        code = init_path.read_text(encoding="utf-8")
        compile(code, str(init_path), "exec")

    def test_generated_test_py_is_importable_in_isolation(self, tmp_path):
        runner = CliRunner()
        runner.invoke(cli, ["forge", "myext", "--output-dir", str(tmp_path)])
        test_path = tmp_path / "team-myext" / "tests" / "test_package.py"
        code = test_path.read_text(encoding="utf-8")
        compile(code, str(test_path), "exec")
