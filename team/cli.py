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


@cli.command()
@click.argument("team_file", type=click.Path(exists=True, dir_okay=False))
def stats(team_file: str) -> None:
    """Show statistics for the most recent run of a team.

    Displays per-speaker turn counts, token usage, run duration, and the
    total number of files written to the shared workspace.
    """
    import json as _json

    from team.bus import Transcript

    cfg = _load(team_file)
    p = cfg.workspace / "transcript.jsonl"
    if not p.is_file():
        console.print(f"[yellow]no transcript found[/yellow] at {p}")
        return

    t = Transcript(persist_path=None)
    from team.bus import Turn as _Turn
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            data = _json.loads(line)
            t.turns.append(_Turn(**data))
        except (ValueError, TypeError):
            continue

    s = t.stats()

    # --- summary row ---
    dur = s["duration_seconds"]
    dur_str = f"{dur:.1f}s" if dur is not None else "—"

    console.print(
        f"\n[bold]Team:[/bold] {cfg.name}  "
        f"[dim]{s['total_turns']} turns · "
        f"{s['total_prompt_tokens'] + s['total_completion_tokens']:,} tokens · "
        f"duration {dur_str} · "
        f"{s['files_written']} file(s) written[/dim]\n"
    )

    # --- per-speaker table ---
    table = Table(title="Turns & token usage by speaker")
    table.add_column("Speaker")
    table.add_column("Turns", justify="right")
    table.add_column("Prompt tokens", justify="right")
    table.add_column("Completion tokens", justify="right")
    table.add_column("Total tokens", justify="right")

    for speaker, count in sorted(s["turns_by_speaker"].items()):
        tok = s["tokens_by_speaker"].get(speaker, {"prompt": 0, "completion": 0})
        p_tok, c_tok = tok["prompt"], tok["completion"]
        table.add_row(
            f"@{speaker}",
            str(count),
            str(p_tok),
            str(c_tok),
            str(p_tok + c_tok),
        )

    table.add_section()
    grand_p = s["total_prompt_tokens"]
    grand_c = s["total_completion_tokens"]
    table.add_row(
        "[bold]total[/bold]",
        str(s["total_turns"]),
        str(grand_p),
        str(grand_c),
        str(grand_p + grand_c),
    )
    console.print(table)


# --------------------------------------------------------------------------- #
# serve (bridge server)
# --------------------------------------------------------------------------- #


@cli.command()
@click.argument("team_file", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--port",
    default=None,
    type=int,
    show_default=True,
    help=(
        "TCP port to listen on.  Defaults to bridge.listen_port in the "
        "team YAML (7000 if not set)."
    ),
)
def serve(team_file: str, port: int | None) -> None:
    """Start a bridge server so this team can accept delegated tasks.

    Remote teams can submit work to this server using the ``delegate_task``
    tool.  Each incoming task triggers a full run of this team's workflow
    inside an isolated sub-workspace; the results (workspace files and a
    summary) are returned to the requesting team when the run completes.

    Example — Lab B exposes its cluster on port 7001:

        team serve lab-b.yaml --port 7001

    Lab A can then delegate sub-tasks using the built-in tool:

        ```tool:delegate_task
        url: http://lab-b.example.com:7001
        goal: Perform the survival analysis on the BRCA dataset.
        files: data/preprocessed.csv
        timeout: 600
        ```
    """
    import signal

    from team.bridge_server import BridgeServer

    cfg = _load(team_file)
    listen_port = port if port is not None else cfg.bridge.listen_port
    max_conc = cfg.bridge.max_concurrent_tasks
    cfg_path = Path(team_file).expanduser().resolve()

    server = BridgeServer(
        cfg_path=cfg_path,
        port=listen_port,
        max_concurrent_tasks=max_conc,
        workspace_root=cfg.workspace / "bridge_workspaces",
    )
    server.start()
    console.print(
        f"[green]bridge server started[/green] — "
        f"team [bold]{cfg.name}[/bold] listening on port [bold]{listen_port}[/bold]"
    )
    console.print(
        f"[dim]max concurrent tasks: {max_conc} · "
        f"workspace: {cfg.workspace / 'bridge_workspaces'}[/dim]"
    )
    console.print("[dim]Press Ctrl-C to stop.[/dim]")

    def _stop(sig: int, frame: object) -> None:  # noqa: ARG001
        console.print("\n[yellow]shutting down bridge server…[/yellow]")
        server.stop()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    server.join()
    console.print("[green]bridge server stopped[/green]")


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


# --------------------------------------------------------------------------- #
# checkpoints
# --------------------------------------------------------------------------- #


@cli.command()
@click.argument("team_file", type=click.Path(exists=True, dir_okay=False))
def checkpoints(team_file: str) -> None:
    """List all workspace checkpoints for a team run.

    A checkpoint is created automatically before each live member turn, so
    you can restore the shared workspace to any of these points in time.
    Use ``team restore`` to roll back to a specific checkpoint.
    """
    from team.workspace import CheckpointManager

    cfg = _load(team_file)
    mgr = CheckpointManager(cfg.workspace)
    items = mgr.list_checkpoints()

    if not items:
        console.print("[yellow]no checkpoints found[/yellow] — run the team first.")
        return

    table = Table(title=f"Checkpoints — team '{cfg.name}'")
    table.add_column("ID", style="bold")
    table.add_column("Turn", justify="right")
    table.add_column("Before member's turn")
    table.add_column("Timestamp")
    table.add_column("Files", justify="right")

    for cp in items:
        ts_fmt = (
            f"{cp.timestamp[:4]}-{cp.timestamp[4:6]}-{cp.timestamp[6:8]} "
            f"{cp.timestamp[9:11]}:{cp.timestamp[11:13]}:{cp.timestamp[13:15]}"
            if len(cp.timestamp) == 15
            else cp.timestamp
        )
        table.add_row(cp.id, str(cp.turn), f"@{cp.member}", ts_fmt, str(cp.file_count))

    console.print(table)
    console.print(
        f"[dim]Use [bold]team restore {team_file} <ID>[/bold] to roll back the shared workspace.[/dim]"
    )


# --------------------------------------------------------------------------- #
# restore
# --------------------------------------------------------------------------- #


@cli.command()
@click.argument("team_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("checkpoint_id")
def restore(team_file: str, checkpoint_id: str) -> None:
    """Restore the shared workspace to a previous checkpoint.

    CHECKPOINT_ID is the full checkpoint name shown by ``team checkpoints``.
    The current contents of the shared workspace are replaced with the
    snapshot; this cannot be undone (unless a later checkpoint captures the
    current state).
    """
    from team.workspace import CheckpointManager

    cfg = _load(team_file)
    mgr = CheckpointManager(cfg.workspace)

    try:
        cp = mgr.restore(checkpoint_id)
    except ValueError as exc:
        console.print(f"[red]error:[/red] {exc}")
        sys.exit(1)

    console.print(
        f"[green]restored[/green] checkpoint [bold]{cp.id}[/bold] "
        f"— {cp.file_count} file(s) now in the shared workspace."
    )


if __name__ == "__main__":  # pragma: no cover
    cli()
