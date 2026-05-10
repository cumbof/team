"""Command-line interface: ``team``."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from team._version import __version__
from team.config import TeamConfigError, load_team
from team.orchestrator import Orchestrator

console = Console()


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, markup=False)],
    )


def _load(team_file: str):
    try:
        return load_team(team_file)
    except TeamConfigError as e:
        console.print(f"[red]config error:[/red] {e}")
        sys.exit(2)


# --------------------------------------------------------------------------- #
# Top-level group
# --------------------------------------------------------------------------- #


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="team")
@click.option("-v", "--verbose", is_flag=True, help="Verbose logging.")
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """Orchestrate a cluster of containerized local LLMs."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    _setup_logging(verbose)


# --------------------------------------------------------------------------- #
# init
# --------------------------------------------------------------------------- #


_TEMPLATE = """\
name: my-team
goal: |
  Describe the high-level objective of this team here.

workspace: ./runs/my-team

workflow:
  type: round_robin       # round_robin | manager | review_loop
  max_rounds: 4

defaults:
  ollama_image: ollama/ollama:latest
  context_window: 8192
  temperature: 0.4
  gpus: none              # "all", "none", or [0, 1]

members:
  - name: lead
    role: Project Lead
    model: llama3.1:8b
    persona: |
      You coordinate the team and make final decisions.
  - name: worker
    role: Engineer
    model: qwen2.5-coder:7b
    persona: |
      You implement code and produce concrete artifacts.
"""


@cli.command()
@click.argument("path", type=click.Path(dir_okay=False, writable=True), default="team.yaml")
def init(path: str) -> None:
    """Write a starter team YAML to PATH (default: team.yaml)."""
    p = Path(path)
    if p.exists():
        console.print(f"[red]refusing to overwrite[/red] {p}")
        sys.exit(1)
    p.write_text(_TEMPLATE, encoding="utf-8")
    console.print(f"[green]wrote[/green] {p}")


# --------------------------------------------------------------------------- #
# validate
# --------------------------------------------------------------------------- #


@cli.command()
@click.argument("team_file", type=click.Path(exists=True, dir_okay=False))
def validate(team_file: str) -> None:
    """Parse and validate a team YAML file without touching Docker."""
    cfg = _load(team_file)
    console.print(f"[green]ok[/green] — team '{cfg.name}' with {len(cfg.members)} member(s)")
    for m in cfg.members:
        console.print(f"  • @{m.name} ({m.role}) — model={m.model}")
    console.print(f"  workflow={cfg.workflow.type} max_rounds={cfg.workflow.max_rounds}")
    console.print(f"  workspace={cfg.workspace}")


# --------------------------------------------------------------------------- #
# up / down / status
# --------------------------------------------------------------------------- #


@cli.command()
@click.argument("team_file", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--prepare-timeout",
    default=600,
    show_default=True,
    help="Seconds to wait for each member's Ollama daemon to be ready and its model to be pulled.",
)
def up(team_file: str, prepare_timeout: int) -> None:
    """Start one Ollama container per member and pull required models."""
    cfg = _load(team_file)
    orch = Orchestrator(cfg)
    orch.up(prepare_deadline_seconds=prepare_timeout)
    console.print("[green]team is up[/green]")
    _print_status(orch)


@cli.command()
@click.argument("team_file", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--purge",
    is_flag=True,
    help="Also remove per-member model volumes (forces a fresh pull next time).",
)
def down(team_file: str, purge: bool) -> None:
    """Stop and remove all containers (and optionally volumes) for a team."""
    cfg = _load(team_file)
    Orchestrator(cfg).down(remove_volumes=purge)
    console.print("[green]team is down[/green]")


@cli.command()
@click.argument("team_file", type=click.Path(exists=True, dir_okay=False))
def status(team_file: str) -> None:
    """Show container status for each member."""
    cfg = _load(team_file)
    _print_status(Orchestrator(cfg))


def _print_status(orch: Orchestrator) -> None:
    table = Table(title=f"team {orch.team.name}")
    for col in ("member", "role", "model", "container", "status"):
        table.add_column(col)
    for row in orch.status():
        table.add_row(row["member"], row["role"], row["model"], row["container"], row["status"])
    console.print(table)


def _setup_streaming(orch: Orchestrator, console: Console) -> None:
    """Wire token-by-token output to the console for live turns."""
    import sys

    def on_turn_start(name: str) -> None:
        try:
            role = orch.team.member(name).role
            console.print(f"\n[bold cyan]@{name}[/bold cyan] [dim]({role})[/dim]")
        except KeyError:
            console.print(f"\n[bold cyan]@{name}[/bold cyan]")

    def on_token(token: str) -> None:
        sys.stdout.write(token)
        sys.stdout.flush()

    def on_turn_end(name: str) -> None:
        sys.stdout.write("\n")
        sys.stdout.flush()

    orch._on_turn_start = on_turn_start
    orch._on_token = on_token
    orch._on_turn_end = on_turn_end


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #


@cli.command()
@click.argument("team_file", type=click.Path(exists=True, dir_okay=False))
@click.option("--no-up", is_flag=True, help="Assume containers are already running.")
@click.option("--keep-up", is_flag=True, help="Leave containers running after the workflow finishes.")
@click.option(
    "--prepare-timeout",
    default=600,
    show_default=True,
    help="Seconds to wait for each member's Ollama daemon to be ready and its model to be pulled.",
)
@click.option(
    "--resume",
    is_flag=True,
    help="Resume from an existing transcript, skipping already-completed turns.",
)
@click.option(
    "--no-stream",
    is_flag=True,
    help="Disable token-by-token streaming output; wait for each full reply before printing.",
)
def run(team_file: str, no_up: bool, keep_up: bool, prepare_timeout: int, resume: bool, no_stream: bool) -> None:
    """Bring the team up and execute its workflow until completion."""
    cfg = _load(team_file)
    orch = Orchestrator(cfg, resume=resume)
    if resume and orch._replay_queue:
        console.print(
            f"[cyan]resuming[/cyan] — replaying {len(orch._replay_queue)} completed turn(s)"
        )
    if not no_stream:
        _setup_streaming(orch, console)
    if not no_up:
        orch.up(prepare_deadline_seconds=prepare_timeout)
    else:
        # Re-attach to existing containers + prepare clients.
        runtimes = orch.containers.start_all()
        from team.member import Member
        for rt in runtimes:
            orch.members[rt.member.name] = Member(cfg, rt.member, rt)
        for m in orch.members.values():
            m.prepare(deadline_seconds=prepare_timeout)
        orch._kickoff()  # noqa: SLF001 — internal but safe

    console.print(f"[bold]running workflow[/bold]: {cfg.workflow.type}")
    try:
        orch.run()
    finally:
        if not keep_up:
            orch.down()
    console.print(f"[green]done[/green] — transcript: {orch.transcript_path()}")


# --------------------------------------------------------------------------- #
# logs
# --------------------------------------------------------------------------- #


@cli.command()
@click.argument("team_file", type=click.Path(exists=True, dir_okay=False))
@click.option("--member", "member_name", help="Show logs for a single member only.")
@click.option("--tail", default=200, show_default=True, help="Lines to tail per container.")
def logs(team_file: str, member_name: str | None, tail: int) -> None:
    """Print docker container logs for one or all members."""
    import docker
    from docker.errors import NotFound

    cfg = _load(team_file)
    client = docker.from_env()
    targets = [m for m in cfg.members if member_name is None or m.name == member_name]
    for m in targets:
        name = f"team-{cfg.name}-{m.name}"
        console.rule(f"@{m.name} ({name})")
        try:
            c = client.containers.get(name)
        except NotFound:
            console.print(f"[yellow]no container[/yellow] {name}")
            continue
        out = c.logs(tail=tail).decode("utf-8", errors="replace")
        console.print(out)


# --------------------------------------------------------------------------- #
# transcript
# --------------------------------------------------------------------------- #


@cli.command()
@click.argument("team_file", type=click.Path(exists=True, dir_okay=False))
def transcript(team_file: str) -> None:
    """Render the persisted transcript of the most recent run."""
    cfg = _load(team_file)
    p = cfg.workspace / "transcript.jsonl"
    if not p.is_file():
        console.print(f"[yellow]no transcript found[/yellow] at {p}")
        return
    import json
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        t = json.loads(line)
        console.rule(f"turn {t['index']} — @{t['speaker']} ({t['role']})")
        console.print(t["content"])
        if t.get("files_written"):
            console.print(f"[dim]wrote: {', '.join(t['files_written'])}[/dim]")


# --------------------------------------------------------------------------- #
# export
# --------------------------------------------------------------------------- #


@cli.command()
@click.argument("team_file", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--format", "fmt",
    type=click.Choice(["markdown", "html"], case_sensitive=False),
    default="markdown",
    show_default=True,
    help="Output format.",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(dir_okay=False, writable=True),
    default=None,
    help="Output file. Defaults to <workspace>/report.md (or .html).",
)
def export(team_file: str, fmt: str, output_path: str | None) -> None:
    """Export the run transcript and shared artifacts to a Markdown or HTML report."""
    from team.export import export_run

    cfg = _load(team_file)
    transcript_p = cfg.workspace / "transcript.jsonl"
    if not transcript_p.is_file():
        console.print(f"[yellow]no transcript found[/yellow] at {transcript_p}")
        sys.exit(1)

    text = export_run(cfg, fmt=fmt)  # type: ignore[arg-type]

    if output_path is None:
        ext = "html" if fmt == "html" else "md"
        out = cfg.workspace / f"report.{ext}"
    else:
        out = Path(output_path)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    console.print(f"[green]exported[/green] → {out}")


if __name__ == "__main__":  # pragma: no cover
    cli()
