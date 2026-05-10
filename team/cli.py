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
# new (wizard)
# --------------------------------------------------------------------------- #


@cli.command()
@click.argument("path", type=click.Path(dir_okay=False, writable=True), default="team.yaml")
def new(path: str) -> None:
    """Interactively create a new team YAML via a guided wizard."""
    p = Path(path)
    if p.exists():
        console.print(f"[red]refusing to overwrite[/red] {p}")
        sys.exit(1)
    from team.wizard import run_wizard
    run_wizard(p)


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
# visualize
# --------------------------------------------------------------------------- #


@cli.command()
@click.argument("team_file", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--format", "fmt",
    type=click.Choice(["ascii", "mermaid"], case_sensitive=False),
    default="ascii",
    show_default=True,
    help="Output format.",
)
def visualize(team_file: str, fmt: str) -> None:
    """Print an ASCII or Mermaid diagram of the team workflow."""
    from team.visualize import render_diagram
    cfg = _load(team_file)
    console.print(render_diagram(cfg, fmt=fmt))


# --------------------------------------------------------------------------- #
# check
# --------------------------------------------------------------------------- #


@cli.command()
@click.argument("team_file", type=click.Path(exists=True, dir_okay=False))
def check(team_file: str) -> None:
    """Run preflight checks for a team spec without starting containers."""
    from team.checks import Status, run_all_checks

    cfg = _load(team_file)
    results = run_all_checks(cfg)

    table = Table(title=f"Preflight checks — team '{cfg.name}'")
    table.add_column("Check", style="bold")
    table.add_column("Status", justify="center")
    table.add_column("Details")

    icons = {
        Status.OK:   "[green]✓  ok[/green]",
        Status.WARN: "[yellow]⚠  warn[/yellow]",
        Status.FAIL: "[red]✗  fail[/red]",
    }
    for r in results:
        table.add_row(r.name, icons[r.status], r.detail)
    console.print(table)

    failures = [r for r in results if r.status == Status.FAIL]
    warnings = [r for r in results if r.status == Status.WARN]
    if failures:
        console.print(f"[red]{len(failures)} critical check(s) failed — fix before running.[/red]")
        sys.exit(1)
    if warnings:
        console.print(f"[yellow]{len(warnings)} warning(s) — run may still work but review above.[/yellow]")
    else:
        console.print("[green]all checks passed[/green]")


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


def _print_token_summary(orch: Orchestrator) -> None:
    """Print a token usage summary table after a workflow completes (F6)."""
    totals = orch._token_totals
    if not any(v["prompt"] + v["completion"] for v in totals.values()):
        return  # all-zero (e.g. pure replay run or backend doesn't report tokens)
    table = Table(title="Token usage (live turns)")
    table.add_column("member")
    table.add_column("prompt", justify="right")
    table.add_column("completion", justify="right")
    table.add_column("total", justify="right")
    grand_p = grand_c = 0
    for name, counts in totals.items():
        p, c = counts["prompt"], counts["completion"]
        grand_p += p
        grand_c += c
        table.add_row(f"@{name}", str(p), str(c), str(p + c))
    table.add_section()
    table.add_row("[bold]total[/bold]", str(grand_p), str(grand_c), str(grand_p + grand_c))
    console.print(table)


def _setup_streaming(orch: Orchestrator, console: Console) -> None:
    """Wire token-by-token output to the console for live turns.

    Three closures are attached to the orchestrator as hook attributes.
    They are called by ``Orchestrator.run_turn()`` during live (non-replay)
    turns.  Using closures rather than subclassing keeps the orchestrator
    decoupled from the CLI.
    """
    import sys

    def on_turn_start(name: str) -> None:
        try:
            role = orch.team.member(name).role
            console.print(f"\n[bold cyan]@{name}[/bold cyan] [dim]({role})[/dim]")
        except KeyError:
            console.print(f"\n[bold cyan]@{name}[/bold cyan]")

    def on_token(token: str) -> None:
        # Write directly to stdout (bypassing Rich) so the token appears
        # immediately without any markup processing or buffering.
        sys.stdout.write(token)
        sys.stdout.flush()

    def on_turn_end(name: str) -> None:
        sys.stdout.write("\n")
        sys.stdout.flush()

    orch._on_turn_start = on_turn_start
    orch._on_token = on_token
    orch._on_turn_end = on_turn_end

    def on_tool_call(member_name: str, tool_name: str, body: str) -> None:
        # Ensure we're on a fresh line (streaming may not have ended with \n).
        sys.stdout.write("\n")
        sys.stdout.flush()
        preview = body[:80].replace("\n", " ").strip()
        if len(body) > 80:
            preview += "…"
        console.print(
            f"  [bold magenta]🔧 tool:[/bold magenta] [magenta]{tool_name}[/magenta]"
            f" [dim]{preview}[/dim]"
        )

    def on_tool_result(member_name: str, tool_name: str, result: str) -> None:
        preview = result[:120].replace("\n", " ").strip()
        if len(result) > 120:
            preview += "…"
        console.print(f"  [dim]   ↳ {preview}[/dim]")

    orch._on_tool_call = on_tool_call
    orch._on_tool_result = on_tool_result


def _setup_interactive(orch: Orchestrator, console: Console) -> None:
    """Attach a round-end callback that pauses for human input.

    After every workflow round the user is prompted for an optional directive.
    Anything typed is injected into the transcript before the next round
    begins so every member sees it in their next turn.  Pressing Enter with
    no text continues without interruption.
    """
    max_rounds = orch.team.workflow.max_rounds

    def on_round_end(round_idx: int) -> None:
        console.print(
            f"\n[bold yellow]── round {round_idx + 1}/{max_rounds} complete ──[/bold yellow]"
        )
        console.print(
            "[dim]Enter a directive for the team (or press Enter to continue):[/dim] ",
            end="",
        )
        try:
            text = input()
        except (EOFError, KeyboardInterrupt):
            # Non-interactive environment (piped stdin) or user pressed Ctrl-C —
            # treat as "no directive" and let the run continue.
            return
        if text.strip():
            orch.inject_directive(text)
            console.print("[bold yellow]↳ directive injected[/bold yellow]")

    orch._on_round_end = on_round_end


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
@click.option(
    "--interactive",
    is_flag=True,
    help="Pause at the end of each round and prompt for an optional human directive.",
)
def run(team_file: str, no_up: bool, keep_up: bool, prepare_timeout: int, resume: bool, no_stream: bool, interactive: bool) -> None:
    """Bring the team up and execute its workflow until completion."""
    cfg = _load(team_file)
    orch = Orchestrator(cfg, resume=resume)
    if resume and orch._replay_queue:
        console.print(
            f"[cyan]resuming[/cyan] — replaying {len(orch._replay_queue)} completed turn(s)"
        )
    if not no_stream:
        _setup_streaming(orch, console)
    if interactive:
        _setup_interactive(orch, console)
    if not no_up:
        orch.up(prepare_deadline_seconds=prepare_timeout)
    else:
        # Re-attach to containers that were started by a previous `team up`
        # without stopping and re-starting them.  We still need to build the
        # Member objects and wait for Ollama to be ready.
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
    _print_token_summary(orch)
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
