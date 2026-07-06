"""Extension factory for ``team-*`` plugin packages.

This module provides :func:`make_extension_commands`, a factory that returns a
fully-wired :class:`click.Group` with the four standard subcommands expected of
every ``team-*`` extension package:

* ``init``      — copy a scenario YAML template into the user's working directory
* ``scenarios`` — list available scenario templates
* ``servers``   — list the package's MCP servers with descriptions
* ``personas``  — list available persona YAML files

Extension authors call the factory once in their ``commands.py`` and register
the returned group via the ``team.commands`` entry-point group::

    # myext/commands.py
    from team.extension import make_extension_commands
    from myext import examples_dir, personas_dir, servers_dir

    _SERVER_DESCRIPTIONS = {
        "my_server": "Exposes some useful tools.",
    }
    _SCENARIO_DESCRIPTIONS = {
        "my-scenario": "Runs a multi-step analysis.",
    }

    myext = make_extension_commands(
        package_name="team-myext",
        group_name="myext",
        description="myext extensions for the team multi-agent framework.",
        servers_dir=servers_dir,
        personas_dir=personas_dir,
        examples_dir=examples_dir,
        server_descriptions=_SERVER_DESCRIPTIONS,
        scenario_descriptions=_SCENARIO_DESCRIPTIONS,
    )

    # pyproject.toml:
    # [project.entry-points."team.commands"]
    # myext = "myext.commands:myext"

After ``pip install team-myext``, ``team myext --help`` becomes available.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import click
import yaml
from rich.console import Console
from rich.table import Table

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #


def _persona_info(yaml_path: Path) -> tuple[str, str]:
    """Return (role, description) from a persona YAML file; empty strings on error."""
    try:
        raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            return raw.get("role", ""), raw.get("description", "")
    except Exception:  # noqa: BLE001
        pass
    return "", ""


# --------------------------------------------------------------------------- #
# Public factory
# --------------------------------------------------------------------------- #


def make_extension_commands(
    *,
    package_name: str,
    group_name: str,
    description: str,
    servers_dir: Callable[[], Path],
    personas_dir: Callable[[], Path],
    examples_dir: Callable[[], Path],
    server_descriptions: dict[str, str] | None = None,
    scenario_descriptions: dict[str, str] | None = None,
) -> click.Group:
    """Build and return a Click command group for a ``team-*`` extension.

    Parameters
    ----------
    package_name:
        PyPI / distribution name (e.g. ``"team-galaxy"``).  Used by
        ``click.version_option`` to read the installed version.
    group_name:
        Name of the Click group / CLI sub-command (e.g. ``"galaxy"``).
        This is what the user types: ``team <group_name> --help``.
    description:
        One-line description shown in ``team --help``.
    servers_dir:
        Zero-argument callable returning the :class:`~pathlib.Path` to the
        extension's ``servers/`` directory.
    personas_dir:
        Zero-argument callable returning the :class:`~pathlib.Path` to the
        extension's ``personas/`` directory.
    examples_dir:
        Zero-argument callable returning the :class:`~pathlib.Path` to the
        extension's ``examples/`` directory.
    server_descriptions:
        Mapping of registered MCP server *name* → short description string.
        Used by the ``servers`` subcommand table.  Defaults to ``{}``.
    scenario_descriptions:
        Mapping of scenario *stem* (YAML filename without extension) →
        short description string.  Used by the ``scenarios`` subcommand
        table.  Defaults to ``{}``.

    Returns
    -------
    click.Group
        A fully-wired Click group with ``init``, ``scenarios``, ``servers``,
        and ``personas`` subcommands.  Assign the return value to a
        module-level variable whose name matches *group_name*; that variable
        is what the ``team.commands`` entry point should reference.
    """
    _server_descs: dict[str, str] = server_descriptions or {}
    _scenario_descs: dict[str, str] = scenario_descriptions or {}
    _console = Console()

    @click.group(name=group_name, help=description)
    @click.version_option(package_name=package_name)
    def _group() -> None:
        pass

    # ------------------------------------------------------------------ #
    # init
    # ------------------------------------------------------------------ #

    @_group.command("init")
    @click.option(
        "--scenario",
        "-s",
        required=True,
        type=click.Choice(
            [p.stem for p in sorted(examples_dir().glob("*.yaml"))],
            case_sensitive=False,
        ),
        help="Scenario template to initialise.",
    )
    @click.option(
        "--output-dir",
        "-o",
        default=".",
        show_default=True,
        help="Directory to write the scenario YAML file to.",
    )
    @click.option(
        "--force",
        is_flag=True,
        default=False,
        help="Overwrite an existing file.",
    )
    def _init(scenario: str, output_dir: str, force: bool) -> None:
        """Copy a scenario template to OUTPUT_DIR, ready to run with ``team run``.

        Server references in the generated YAML use short registered names that
        resolve automatically when the extension is installed — no paths to edit.
        Edit the ``goal`` and ``model`` fields before running.
        """
        src = examples_dir() / f"{scenario}.yaml"
        if not src.is_file():
            _console.print(f"[red]Scenario template not found:[/red] {src}")
            raise SystemExit(1)

        dest = Path(output_dir).expanduser().resolve() / f"{scenario}.yaml"
        if dest.exists() and not force:
            _console.print(
                f"[yellow]{dest} already exists.[/yellow] Use --force to overwrite."
            )
            raise SystemExit(1)

        content = src.read_text(encoding="utf-8")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")

        _console.print(f"[green]Created[/green] {dest}")
        _console.print("")
        _console.print("Next steps:")
        _console.print("  1. Edit the [bold]goal[/bold] and [bold]model[/bold] fields.")
        _console.print(f"  2. Run:  [bold]team run {dest.name}[/bold]")

    # ------------------------------------------------------------------ #
    # scenarios
    # ------------------------------------------------------------------ #

    @_group.command("scenarios")
    def _scenarios() -> None:
        """List available scenario templates."""
        table = Table(title=f"Available {package_name} scenarios", show_lines=True)
        table.add_column("Name", style="bold cyan")
        table.add_column("Description")

        for yaml_path in sorted(examples_dir().glob("*.yaml")):
            name = yaml_path.stem
            desc = _scenario_descs.get(name, "")
            table.add_row(name, desc)

        _console.print(table)
        _console.print("")
        _console.print(f"Initialise a scenario:  [bold]team {group_name} init --scenario <name>[/bold]")

    # ------------------------------------------------------------------ #
    # servers
    # ------------------------------------------------------------------ #

    @_group.command("servers")
    def _servers() -> None:
        """List the package's MCP servers and descriptions."""
        table = Table(title=f"Available {package_name} MCP servers", show_lines=True)
        table.add_column("Name", style="bold cyan")
        table.add_column("Description")

        for name, desc in sorted(_server_descs.items()):
            table.add_row(name, desc)

        _console.print(table)
        _console.print("")
        _console.print(
            "Use in YAML:  [bold]mcp_servers:\\n  "
            f"{next(iter(_server_descs), 'my_server')}:\\n    transport: entry_point\\n"
            f"    entry_point: {next(iter(_server_descs), 'my_server')}[/bold]"
        )

    # ------------------------------------------------------------------ #
    # personas
    # ------------------------------------------------------------------ #

    @_group.command("personas")
    def _personas() -> None:
        """List available personas with descriptions."""
        table = Table(title=f"Available {package_name} personas", show_lines=True)
        table.add_column("Key", style="bold cyan")
        table.add_column("Role")
        table.add_column("Description")

        for yaml_path in sorted(personas_dir().glob("*.yaml")):
            role, desc = _persona_info(yaml_path)
            table.add_row(f"@{yaml_path.stem}", role, desc)

        _console.print(table)
        _console.print("")
        _console.print("Use in YAML:  [bold]persona: \"@<name>\"[/bold]")
        _console.print("(Personas are auto-discovered when the extension is installed.)")

    return _group
