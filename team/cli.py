"""Command-line interface: ``team``."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
from rich.console import Console, Group
from rich.live import Live
from rich.logging import RichHandler
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from team._version import __version__
from team.config import TeamConfigError, load_team
from team.orchestrator import Orchestrator

console = Console()

# Colour palette — one colour per team member, cycles if there are more members than colours.
_MEMBER_COLORS = [
    "bright_cyan", "bright_green", "bright_yellow", "bright_magenta",
    "cornflower_blue", "orange1", "spring_green1", "plum1", "gold1", "violet",
]


def _get_member_color(name: str, members: list) -> str:
    idx = next((i for i, m in enumerate(members) if m.name == name), 0)
    return _MEMBER_COLORS[idx % len(_MEMBER_COLORS)]


def _print_team_banner(cfg, cons: Console) -> None:
    """Render a rich overview panel before the run starts."""
    body = Text()
    body.append(cfg.goal.strip(), style="italic")
    body.append("\n\n")
    body.append("Workflow  ", style="dim")
    body.append(cfg.workflow.type, style="bold")
    body.append(f"  ·  max {cfg.workflow.max_rounds} rounds"
                f"  ·  {len(cfg.members)} member(s)\n", style="dim")
    for i, m in enumerate(cfg.members):
        color = _MEMBER_COLORS[i % len(_MEMBER_COLORS)]
        body.append(f"\n  ● @{m.name}", style=f"bold {color}")
        body.append(f"  {m.role}", style="default")
        body.append(f"  [{m.model}]", style="dim italic")
    cons.print(Panel(body, title=f"[bold]🤖  {cfg.name}[/bold]",
                     border_style="blue", padding=(1, 2)))
    cons.print()

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
# Helpers for GPU / host-ollama overrides
# --------------------------------------------------------------------------- #


def _apply_no_gpu(cfg) -> None:
    """Force gpus='none' on all members and defaults, disabling NVIDIA device requests."""
    cfg.defaults.gpus = "none"
    for m in cfg.members:
        m.gpus = "none"


def _apply_host_ollama(cfg, url: str) -> None:
    """Route all members to an external Ollama instance, bypassing Docker."""
    cfg.defaults.ollama_url = url


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
@click.option(
    "--no-gpu",
    is_flag=True,
    help="Disable GPU device requests for all containers (CPU-only). "
         "Useful on macOS / systems without NVIDIA hardware.",
)
@click.option(
    "--host-ollama",
    "host_ollama",
    default=None,
    metavar="URL",
    help="Skip Docker entirely and connect all members to an already-running Ollama "
         "instance at URL (e.g. http://localhost:11434). Recommended on Apple Silicon "
         "where the native Ollama app uses Metal GPU acceleration.",
)
def up(team_file: str, prepare_timeout: int, no_gpu: bool, host_ollama: str | None) -> None:
    """Start one Ollama container per member and pull required models."""
    cfg = _load(team_file)
    if host_ollama:
        _apply_host_ollama(cfg, host_ollama)
    elif no_gpu:
        _apply_no_gpu(cfg)
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
    table = Table(title=f"[bold]team · {orch.team.name}[/bold]",
                  border_style="dim", header_style="bold dim")
    for col in ("member", "role", "model", "container", "status"):
        table.add_column(col)
    for i, row in enumerate(orch.status()):
        color = _MEMBER_COLORS[i % len(_MEMBER_COLORS)]
        table.add_row(
            f"[bold {color}]@{row['member']}[/bold {color}]",
            row["role"], row["model"], row["container"], row["status"],
        )
    console.print(table)


def _print_token_summary(orch: Orchestrator) -> None:
    """Print a token usage summary table after a workflow completes."""
    totals = orch._token_totals
    if not any(v["prompt"] + v["completion"] for v in totals.values()):
        return
    table = Table(title="[bold]Token usage[/bold]",
                  border_style="dim", header_style="bold dim")
    table.add_column("member")
    table.add_column("prompt", justify="right")
    table.add_column("completion", justify="right")
    table.add_column("total", justify="right")
    grand_p = grand_c = 0
    for i, (name, counts) in enumerate(totals.items()):
        color = _get_member_color(name, orch.team.members)
        p, c = counts["prompt"], counts["completion"]
        grand_p += p
        grand_c += c
        table.add_row(
            f"[bold {color}]@{name}[/bold {color}]",
            str(p), str(c), str(p + c),
        )
    table.add_section()
    table.add_row("[bold]total[/bold]", str(grand_p), str(grand_c),
                  str(grand_p + grand_c))
    console.print(table)


def _setup_streaming(orch: Orchestrator, cons: Console) -> None:
    """Wire live panel rendering for each member turn.

    Each member turn is rendered inside a colour-coded Rich Panel that updates
    token-by-token as the LLM streams its reply.  Tool invocations and their
    results are appended to the same panel so the user sees a single coherent
    view per turn.  A dim Rule separates rounds.
    """
    # Mutable closure state — avoids global variables.
    state: dict = {
        "live": None,
        "text": Text(),
        "tools": [],
        "name": "",
        "role": "",
        "color": "cyan",
    }

    def _panel() -> Panel:
        parts: list = []
        if len(state["text"]) > 0:
            parts.append(state["text"])
        parts.extend(state["tools"])
        body = Group(*parts) if parts else Text("thinking…", style="dim italic")
        title = (
            f"[bold {state['color']}]@{state['name']}[/bold {state['color']}]"
            f" [dim]· {state['role']}[/dim]"
        )
        return Panel(body, title=title, border_style=state["color"], padding=(0, 1))

    def on_turn_start(name: str) -> None:
        state["name"] = name
        state["text"] = Text()
        state["tools"] = []
        try:
            state["role"] = orch.team.member(name).role
        except KeyError:
            state["role"] = ""
        state["color"] = _get_member_color(name, orch.team.members)
        live = Live(_panel(), console=cons, refresh_per_second=12, transient=False)
        live.start()
        state["live"] = live

    def on_token(token: str) -> None:
        state["text"].append(token)
        if state["live"]:
            state["live"].update(_panel())

    def on_turn_end(_name: str) -> None:
        live = state["live"]
        if live:
            live.update(_panel())
            live.stop()
            state["live"] = None

    def on_tool_call(_member: str, tool_name: str, body: str) -> None:
        preview = body[:70].replace("\n", " ").strip()
        if len(body) > 70:
            preview += "…"
        row = Text()
        row.append("\n  🔧 ", style="bold magenta")
        row.append(tool_name, style="bold magenta")
        row.append(f"  {preview}", style="dim")
        state["tools"].append(row)
        if state["live"]:
            state["live"].update(_panel())

    def on_tool_result(_member: str, _tool: str, result: str) -> None:
        preview = result[:120].replace("\n", " ").strip()
        if len(result) > 120:
            preview += "…"
        row = Text()
        row.append("     ↳ ", style="dim green")
        row.append(preview, style="dim")
        state["tools"].append(row)
        if state["live"]:
            state["live"].update(_panel())

    def on_round_end(round_idx: int) -> None:
        max_rounds = orch.team.workflow.max_rounds
        cons.print()
        cons.print(Rule(
            f"[dim]round {round_idx + 1} of {max_rounds} complete[/dim]",
            style="dim",
        ))
        cons.print()

    orch._on_turn_start = on_turn_start
    orch._on_token = on_token
    orch._on_turn_end = on_turn_end
    orch._on_tool_call = on_tool_call
    orch._on_tool_result = on_tool_result
    orch._on_round_end = on_round_end


def _setup_interactive(orch: Orchestrator, cons: Console) -> None:
    """Layer interactive prompts on top of the round-end hook.

    Chains with any existing ``_on_round_end`` (e.g. the round separator
    printed by ``_setup_streaming``) so both fire in order.
    """
    _prev = orch._on_round_end

    def on_round_end(round_idx: int) -> None:
        if _prev:
            _prev(round_idx)
        cons.print(
            "[dim]Enter a directive for the team (or press Enter to continue):[/dim] ",
            end="",
        )
        try:
            text = input()
        except (EOFError, KeyboardInterrupt):
            return
        if text.strip():
            orch.inject_directive(text)
            cons.print("[bold yellow]↳ directive injected[/bold yellow]")

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
@click.option(
    "--no-gpu",
    is_flag=True,
    help="Disable GPU device requests for all containers (CPU-only). "
         "Useful on macOS / systems without NVIDIA hardware.",
)
@click.option(
    "--host-ollama",
    "host_ollama",
    default=None,
    metavar="URL",
    help="Skip Docker entirely and connect all members to an already-running Ollama "
         "instance at URL (e.g. http://localhost:11434). Recommended on Apple Silicon "
         "where the native Ollama app uses Metal GPU acceleration.",
)
def run(team_file: str, no_up: bool, keep_up: bool, prepare_timeout: int, resume: bool, no_stream: bool, interactive: bool, no_gpu: bool, host_ollama: str | None) -> None:
    """Bring the team up and execute its workflow until completion."""
    cfg = _load(team_file)
    if host_ollama:
        _apply_host_ollama(cfg, host_ollama)
    elif no_gpu:
        _apply_no_gpu(cfg)
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
        with console.status("[bold blue]starting containers and pulling models…[/bold blue]"):
            orch.up(prepare_deadline_seconds=prepare_timeout)
        console.print("[green]✓[/green] team is up\n")
    else:
        runtimes = orch.containers.start_all()
        from team.member import Member
        for rt in runtimes:
            orch.members[rt.member.name] = Member(cfg, rt.member, rt)
        for m in orch.members.values():
            m.prepare(deadline_seconds=prepare_timeout)
        orch._kickoff()  # noqa: SLF001 — internal but safe

    _print_team_banner(cfg, console)
    console.print(Rule(f"[dim]round 1 of {cfg.workflow.max_rounds}[/dim]", style="dim"))
    console.print()
    try:
        orch.run()
    finally:
        if not keep_up:
            orch.down()
    console.print()
    _print_token_summary(orch)
    console.print(f"[green]✓ done[/green]  transcript → {orch.transcript_path()}")


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
        secret=cfg.bridge.secret,
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
        name = t["speaker"]
        color = _get_member_color(name, cfg.members)
        title = (
            f"[bold {color}]@{name}[/bold {color}]"
            f" [dim]· {t['role']}  ·  turn {t['index']}[/dim]"
        )
        body = Text(t["content"])
        if t.get("files_written"):
            body.append(f"\n\n📄 wrote: {', '.join(t['files_written'])}", style="dim green")
        console.print(Panel(body, title=title, border_style=color, padding=(0, 1)))
        console.print()


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


# --------------------------------------------------------------------------- #
# rollback
# --------------------------------------------------------------------------- #


@cli.command()
@click.argument("team_file", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--to",
    "checkpoint_id",
    default=None,
    metavar="ID",
    help="Checkpoint ID to restore (from the list shown without --to).",
)
@click.option(
    "--yes",
    is_flag=True,
    default=False,
    help="Skip the confirmation prompt when restoring.",
)
def rollback(team_file: str, checkpoint_id: str | None, yes: bool) -> None:
    """List checkpoints and optionally roll the workspace back to one.

    Without ``--to``, shows all available checkpoints as a table (same as
    ``team checkpoints``).  With ``--to ID``, atomically replaces the shared
    workspace with the snapshot at that checkpoint, after asking for
    confirmation (skip with ``--yes``).

    Example workflow:

    \\b
        team rollback myteam.yaml               # list all checkpoints
        team rollback myteam.yaml --to 0005_alice_20250510T183000
        team rollback myteam.yaml --to 0005_alice_20250510T183000 --yes

    After rolling back, use ``team run --resume`` to continue from the
    restored state with a different prompt or model configuration.
    """
    from team.workspace import CheckpointManager

    cfg = _load(team_file)
    mgr = CheckpointManager(cfg.workspace)
    items = mgr.list_checkpoints()

    if not items:
        console.print("[yellow]no checkpoints found[/yellow] — run the team first.")
        return

    if checkpoint_id is None:
        # Just list.
        table = Table(title=f"Checkpoints — team '{cfg.name}'  (use --to ID to restore)")
        table.add_column("ID", style="bold")
        table.add_column("Turn", justify="right")
        table.add_column("Before member")
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
        return

    # Validate the checkpoint exists.
    matching = [cp for cp in items if cp.id == checkpoint_id]
    if not matching:
        console.print(f"[red]checkpoint not found:[/red] {checkpoint_id!r}")
        console.print(f"[dim]Run [bold]team rollback {team_file}[/bold] to see available checkpoints.[/dim]")
        sys.exit(1)
    cp_info = matching[0]

    # Confirm unless --yes.
    if not yes:
        ts_fmt = (
            f"{cp_info.timestamp[:4]}-{cp_info.timestamp[4:6]}-{cp_info.timestamp[6:8]} "
            f"{cp_info.timestamp[9:11]}:{cp_info.timestamp[11:13]}:{cp_info.timestamp[13:15]}"
            if len(cp_info.timestamp) == 15
            else cp_info.timestamp
        )
        console.print(
            f"[yellow]About to restore checkpoint [bold]{checkpoint_id}[/bold][/yellow]\n"
            f"  Turn: {cp_info.turn}  ·  Before: @{cp_info.member}  ·  "
            f"  Timestamp: {ts_fmt}  ·  Files: {cp_info.file_count}\n"
            "[red]This will replace the current shared workspace.[/red]"
        )
        try:
            confirmed = click.confirm("Continue?", default=False)
        except click.Abort:
            confirmed = False
        if not confirmed:
            console.print("[dim]Cancelled.[/dim]")
            return

    try:
        restored = mgr.restore(checkpoint_id)
    except ValueError as exc:
        console.print(f"[red]error:[/red] {exc}")
        sys.exit(1)

    console.print(
        f"[green]Rolled back[/green] to checkpoint [bold]{restored.id}[/bold] "
        f"— {restored.file_count} file(s) restored to shared workspace."
    )
    console.print(
        "[dim]Tip: run [bold]team run --resume[/bold] to continue from this point.[/dim]"
    )


# --------------------------------------------------------------------------- #
# beliefs
# --------------------------------------------------------------------------- #


@cli.command()
@click.argument("team_file", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--status",
    default=None,
    type=click.Choice(["pending", "accepted", "contested", "rejected"]),
    help="Filter beliefs by status.",
)
def beliefs(team_file: str, status: str | None) -> None:
    """Display the team's shared belief board.

    Shows all beliefs the team has asserted during their runs, together
    with confidence scores, voting counts, and current consensus status.

    Examples:

    \\b
        team beliefs myteam.yaml                    # all beliefs
        team beliefs myteam.yaml --status accepted  # accepted only
        team beliefs myteam.yaml --status contested # requires attention
    """
    from team.beliefs import BeliefBoard

    cfg = _load(team_file)
    beliefs_path = cfg.workspace / "beliefs.json"
    if not beliefs_path.exists():
        console.print(
            "[yellow]No belief board found.[/yellow]\n"
            "[dim]Enable beliefs in your team YAML:\n"
            "  beliefs:\n"
            "    enabled: true[/dim]"
        )
        return

    board = BeliefBoard(
        path=beliefs_path,
        member_names=cfg.member_names(),
        consensus_threshold=cfg.beliefs.consensus_threshold,
    )
    items = board.list_beliefs(status=status)

    if not items:
        msg = f"No beliefs with status [bold]{status}[/bold]." if status else "The belief board is empty."
        console.print(f"[yellow]{msg}[/yellow]")
        return

    _ICONS = {"accepted": "✓", "contested": "⚡", "rejected": "✗", "pending": "?"}
    _COLORS = {"accepted": "green", "contested": "yellow", "rejected": "red", "pending": "blue"}

    title = f"Belief board — team '{cfg.name}'"
    if status:
        title += f"  [status={status}]"
    table = Table(title=title)
    table.add_column("ID", style="bold")
    table.add_column("Status")
    table.add_column("Claim")
    table.add_column("Confidence", justify="right")
    table.add_column("By")
    table.add_column("For", justify="right")
    table.add_column("Against", justify="right")

    for b in items:
        icon = _ICONS.get(b.status, "?")
        color = _COLORS.get(b.status, "white")
        claim_preview = b.claim[:80] + "…" if len(b.claim) > 80 else b.claim
        table.add_row(
            b.id,
            f"[{color}]{icon} {b.status}[/{color}]",
            claim_preview,
            f"{b.confidence:.0%}",
            f"@{b.author}",
            str(len(b.votes_for)),
            str(len(b.votes_against)),
        )

    console.print(table)
    if any(b.status == "contested" for b in items):
        console.print(
            "[yellow]⚡ Some beliefs are contested — review and resolve via accept_belief / contest_belief tools.[/yellow]"
        )


# --------------------------------------------------------------------------- #
# personas
# --------------------------------------------------------------------------- #


@cli.command()
@click.argument("key", required=False)
def personas(key: str | None) -> None:
    """List predefined personas (or show a full persona by KEY).

    Without arguments, prints a table of all available predefined personas.
    Pass a KEY to display the full persona text.

    Examples:

    \\b
        team personas            # list all predefined personas
        team personas pi         # show the full 'pi' (Principal Investigator) persona
        team personas engineer   # show the full 'engineer' persona
    """
    from team.persona_library import PERSONAS, list_all

    if key is not None:
        if key not in PERSONAS:
            available = ", ".join(sorted(PERSONAS))
            console.print(f"[red]Unknown persona key:[/red] {key!r}\nAvailable: {available}")
            sys.exit(2)
        entry = PERSONAS[key]
        console.print(f"[bold]@{key}[/bold] — [cyan]{entry['role']}[/cyan]")
        console.print(f"[dim]{entry['description']}[/dim]\n")
        console.print(entry["persona"])
        console.print(
            f"\n[dim]Use in YAML:[/dim] [bold]persona: \"@{key}\"[/bold]"
        )
        return

    table = Table(title="Predefined persona library", show_lines=True)
    table.add_column("Key", style="bold cyan", no_wrap=True)
    table.add_column("Role", style="bold")
    table.add_column("Description")
    for entry in list_all():
        table.add_row(f"@{entry['key']}", entry["role"], entry["description"])
    console.print(table)
    console.print(
        "\n[dim]Use in YAML:[/dim]  [bold]persona: \"@<key>\"[/bold]  "
        "(role is set automatically; override with [bold]role:[/bold])\n"
        "[dim]Full persona:[/dim]  [bold]team personas <key>[/bold]"
    )


# --------------------------------------------------------------------------- #
# test
# --------------------------------------------------------------------------- #


@cli.command(name="test")
@click.argument("team_file", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--no-run",
    is_flag=True,
    help="Skip the team run and validate an existing workspace/transcript directly.",
)
@click.option(
    "--goal",
    default=None,
    help="Override the team goal for this test run.",
)
@click.option(
    "--max-rounds",
    "max_rounds",
    default=None,
    type=int,
    help="Override max_rounds for this test run.",
)
@click.option(
    "--no-gpu",
    is_flag=True,
    help="Disable GPU device requests (CPU-only).",
)
@click.option(
    "--host-ollama",
    "host_ollama",
    default=None,
    metavar="URL",
    help="Skip Docker and connect all members to an existing Ollama instance.",
)
def test_cmd(
    team_file: str,
    no_run: bool,
    goal: str | None,
    max_rounds: int | None,
    no_gpu: bool,
    host_ollama: str | None,
) -> None:
    """Run the team then validate assertions from the team YAML's tests: section.

    Assertions are defined in the ``tests:`` section of the team YAML::

        tests:
          - name: creates hello.py
            type: file_exists
            path: hello.py
          - name: script contains print
            type: file_contains
            path: hello.py
            text: "print"
          - name: any agent mentioned Python
            type: transcript_contains
            text: "Python"

    Use ``--no-run`` to validate an existing workspace without re-running the
    team (useful after a manual run).

    Exits with code 0 if all assertions pass, 1 if any fail.
    """
    from team.test_runner import run_assertions

    cfg = _load(team_file)

    if not cfg.tests:
        console.print("[yellow]no tests: section found in team YAML — nothing to assert.[/yellow]")
        return

    if not no_run:
        if goal:
            cfg.goal = goal
        if max_rounds is not None:
            cfg.workflow.max_rounds = max_rounds
        if host_ollama:
            _apply_host_ollama(cfg, host_ollama)
        elif no_gpu:
            _apply_no_gpu(cfg)

        orch = Orchestrator(cfg)
        _setup_streaming(orch, console)
        _print_team_banner(cfg, console)
        console.print(Rule(f"[dim]round 1 of {cfg.workflow.max_rounds}[/dim]", style="dim"))
        console.print()
        with console.status("[bold blue]starting containers and pulling models…[/bold blue]"):
            orch.up()
        console.print("[green]✓[/green] team is up\n")
        try:
            orch.run()
        finally:
            orch.down()
        console.print()

    transcript_path = cfg.workspace / "transcript.jsonl"
    results = run_assertions(cfg.tests, cfg.workspace, transcript_path)

    table = Table(
        title=f"[bold]Test results — {cfg.name}[/bold]",
        border_style="dim",
        header_style="bold dim",
        show_lines=True,
    )
    table.add_column("Test", style="bold")
    table.add_column("Result", justify="center")
    table.add_column("Detail")

    passed = failed = 0
    for r in results:
        if r.passed:
            icon = "[green]✓ pass[/green]"
            passed += 1
        else:
            icon = "[red]✗ fail[/red]"
            failed += 1
        table.add_row(r.name, icon, r.detail)

    console.print(table)

    if failed:
        console.print(f"\n[red]{failed} test(s) failed · {passed} passed[/red]")
        sys.exit(1)
    else:
        console.print(f"\n[green]all {passed} test(s) passed ✓[/green]")


if __name__ == "__main__":  # pragma: no cover
    cli()
