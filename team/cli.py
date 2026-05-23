"""Command-line interface: ``team``."""

from __future__ import annotations

import logging
import shutil
import sys
from importlib.metadata import entry_points
from pathlib import Path

import click
from rich import box as _rich_box
from rich.console import Console, Group
from rich.live import Live

log = logging.getLogger(__name__)
from rich.logging import RichHandler
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from team._version import __version__
from team.config import TeamConfigError, load_team
from team.orchestrator import Orchestrator

# Maximum console width: keeps the UI readable on very wide terminals.
# On narrower terminals the actual terminal width is used instead.
_MAX_CONSOLE_WIDTH = 100

def _make_console() -> Console:
    """Return a Console whose width is capped at *_MAX_CONSOLE_WIDTH*."""
    cols = shutil.get_terminal_size((80, 24)).columns
    return Console(width=min(cols, _MAX_CONSOLE_WIDTH))

console = _make_console()

# Separator length for decorative ─ lines; derived from the console width so
# they never overflow the panel.  4 chars of left-side prefix (e.g. "  └") are
# subtracted so the full line stays within the console width.
_SEP = console.width - 4

# Colour palette — one colour per team member, cycles if there are more members than colours.
_MEMBER_COLORS = [
    "bright_cyan", "bright_green", "bright_yellow", "bright_magenta",
    "cornflower_blue", "orange1", "spring_green1", "plum1", "gold1", "violet",
]


def _get_member_color(name: str, members: list) -> str:
    idx = next((i for i, m in enumerate(members) if m.name == name), 0)
    return _MEMBER_COLORS[idx % len(_MEMBER_COLORS)]


def _print_team_banner(cfg, cons: Console) -> None:
    """Render a compact overview panel before the run starts."""
    header = Text()
    header.append(f"  🤖  {cfg.name}", style="bold white")
    header.append(f"  ·  {cfg.workflow.type}", style="bold bright_blue")
    header.append(
        f"  ·  max {cfg.workflow.max_rounds} rounds"
        f"  ·  {len(cfg.members)} member(s)",
        style="dim",
    )

    goal = Text(f"  {cfg.goal.strip()}", style="italic dim")

    chips = Text("  ")
    for i, m in enumerate(cfg.members):
        color = _MEMBER_COLORS[i % len(_MEMBER_COLORS)]
        chips.append(f" @{m.name} ", style=f"bold {color} on grey11")
        chips.append(f" {m.role} ", style="dim on grey11")
        chips.append(f" {m.model} ", style="dim italic on grey11")
        chips.append("  ")

    body = Group(header, Text(""), goal, Text(""), chips)
    cons.print(Panel(body, border_style="bright_blue", padding=(0, 1), box=_rich_box.ROUNDED))
    cons.print()


def _render_turn_block(
    name: str,
    role: str,
    color: str,
    content: str,
    *,
    model: str = "",
    turn_index: int | None = None,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    files_written: list | None = None,
) -> Group:
    """Return a Group renderable for a completed turn in the new left-bar style."""
    lines: list = []

    hdr = Text()
    hdr.append(f"  @{name} ", style=f"bold {color} on grey11")
    hdr.append(f" {role} ", style="dim on grey11")
    if model:
        hdr.append(f" {model} ", style="dim italic on grey11")
    if turn_index is not None:
        hdr.append(f"  turn {turn_index}", style="dim on grey11")
    if prompt_tokens or completion_tokens:
        hdr.append(
            f"  ↑{prompt_tokens:,} ↓{completion_tokens:,} tok",
            style="dim on grey11",
        )
    lines.append(hdr)

    for line in (content.split("\n") if content.strip() else [""]):
        t = Text()
        t.append("  │ ", style=color)
        t.append(line)
        lines.append(t)

    if files_written:
        t = Text()
        t.append("  │ ", style=color)
        t.append(f"📄 wrote: {', '.join(files_written)}", style="dim green")
        lines.append(t)

    lines.append(Text(f"  └{'─' * _SEP}", style=f"{color} dim"))
    return Group(*lines)


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


def _load_plugin_commands() -> None:
    """Register commands from installed packages that declare ``team.commands`` entry points.

    Each entry point must load to a :class:`click.BaseCommand` (a command or
    group).  The entry-point name becomes the sub-command name under ``team``.

    Example ``pyproject.toml`` declaration in an extension package::

        [project.entry-points."team.commands"]
        galaxy = "team_galaxy.commands:galaxy"

    After ``pip install team-mypkg``, ``team mypkg --help`` becomes available.
    Errors loading individual plugins are logged and skipped so one bad plugin
    does not prevent ``team`` from starting.
    """
    try:
        for ep in entry_points(group="team.commands"):
            try:
                cmd = ep.load()
                cli.add_command(cmd, name=ep.name)
                log.debug("cli: loaded plugin command %r from %s", ep.name, ep.value)
            except Exception as exc:  # noqa: BLE001
                log.warning("cli: failed to load plugin command %r: %s", ep.name, exc)
    except Exception:  # noqa: BLE001
        pass  # importlib.metadata not available in very old environments


_load_plugin_commands()


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
    """Print a token-usage summary in the compact bar-chart style."""
    from team.config import resolve_member_setting
    from team.pricing import estimate_cost, format_cost

    totals = orch._token_totals
    if not any(v["prompt"] + v["completion"] for v in totals.values()):
        return

    grand_p = sum(v["prompt"] for v in totals.values())
    grand_c = sum(v["completion"] for v in totals.values())
    grand_t = grand_p + grand_c

    hdr = Text()
    hdr.append("  📊  Token usage", style="bold")
    hdr.append(f"   {grand_t:,} total  ·  {len(totals)} members", style="dim")
    console.print(hdr)
    console.print()

    bar_width = 28
    grand_cost: float | None = 0.0
    for name, counts in totals.items():
        color = _get_member_color(name, orch.team.members)
        p, c = counts["prompt"], counts["completion"]
        t = p + c
        filled = round(t / grand_t * bar_width) if grand_t else 0
        bar = "█" * filled + "░" * (bar_width - filled)
        try:
            mc = orch.team.member(name)
            backend = resolve_member_setting(mc, orch.team.defaults, "backend") or "ollama"
            is_local = backend == "ollama"
            cost = estimate_cost(mc.model, p, c, is_local=is_local)
        except (KeyError, AttributeError):
            cost = None
            is_local = True
        if cost is None:
            grand_cost = None
        elif grand_cost is not None:
            grand_cost += cost
        row = Text()
        row.append(f"  @{name:<10}", style=f"bold {color}")
        row.append(f" [{bar}] ", style=color)
        row.append(f"{t:>6,} tok", style="bold")
        row.append(f"    ↑{p:,} prompt  ↓{c:,} completion", style="dim")
        if cost is not None and not is_local:
            row.append(f"  {format_cost(cost)}", style="dim")
        console.print(row)

    if grand_cost is not None and grand_cost > 0:
        console.print()
        cost_line = Text()
        cost_line.append("  est. total cost: ", style="dim")
        cost_line.append(format_cost(grand_cost), style="dim bold")
        console.print(cost_line)


def _setup_streaming(orch: Orchestrator, cons: Console) -> None:
    """Wire live left-bar rendering for each member turn.

    Each turn is rendered with a colour-coded header bar and a left gutter
    (│) that grows token-by-token.  Tool calls and results are inlined below
    the content.  A compact round-complete footer separates rounds.
    """
    state: dict = {
        "live": None,
        "text": Text(),
        "tools": [],   # Text lines for tool calls and results
        "name": "",
        "role": "",
        "model": "",
        "color": "cyan",
        "pending_round_header": False,
        "pending_round": 2,
    }

    def _live_group() -> Group:
        lines: list = []
        hdr = Text()
        hdr.append(f"  @{state['name']} ", style=f"bold {state['color']} on grey11")
        hdr.append(f" {state['role']} ", style="dim on grey11")
        if state["model"]:
            hdr.append(f" {state['model']} ", style="dim italic on grey11")
        lines.append(hdr)
        text_str = str(state["text"])
        if text_str.strip():
            for line in text_str.split("\n"):
                t = Text()
                t.append("  │ ", style=state["color"])
                t.append(line)
                lines.append(t)
        else:
            t = Text()
            t.append("  │ ", style=state["color"])
            t.append("thinking…", style="dim italic")
            lines.append(t)
        lines.extend(state["tools"])
        lines.append(Text(f"  └{'─' * _SEP}", style=f"{state['color']} dim"))
        return Group(*lines)

    def on_turn_start(name: str) -> None:
        if state["pending_round_header"]:
            max_rounds = orch.team.workflow.max_rounds
            rh = Text()
            rh.append(f"  ▶  Round {state['pending_round']} / {max_rounds}", style="bold bright_blue")
            rh.append(f"   {len(orch.team.members)} members active", style="dim")
            cons.print(rh)
            cons.print(Text("  " + "─" * _SEP, style="bright_blue dim"))
            cons.print()
            state["pending_round_header"] = False
        state["name"] = name
        state["text"] = Text()
        state["tools"] = []
        try:
            member = orch.team.member(name)
            state["role"] = member.role
            state["model"] = member.model
        except KeyError:
            state["role"] = ""
            state["model"] = ""
        state["color"] = _get_member_color(name, orch.team.members)
        live = Live(_live_group(), console=cons, refresh_per_second=12, transient=False)
        live.start()
        state["live"] = live

    def on_token(token: str) -> None:
        state["text"].append(token)
        if state["live"]:
            state["live"].update(_live_group())

    def on_turn_end(_name: str) -> None:
        live = state["live"]
        if live:
            live.update(_live_group())
            live.stop()
            state["live"] = None
        cons.print()

    def on_tool_call(_member: str, tool_name: str, body: str) -> None:
        preview = body[:65].replace("\n", " ").strip()
        if len(body) > 65:
            preview += "…"
        tool_line = Text()
        tool_line.append("  │  ", style=state["color"])
        tool_line.append(" 🔧 ", style="magenta")
        tool_line.append(tool_name, style="bold magenta")
        tool_line.append(f"  {preview}", style="dim")
        state["tools"].append(tool_line)
        if state["live"]:
            state["live"].update(_live_group())

    def on_tool_result(_member: str, _tool: str, result: str) -> None:
        preview = result[:120].replace("\n", " ").strip()
        if len(result) > 120:
            preview += "…"
        res_line = Text()
        res_line.append("  │     ", style=state["color"])
        res_line.append("↳ ", style="green")
        res_line.append(preview, style="dim green")
        state["tools"].append(res_line)
        if state["live"]:
            state["live"].update(_live_group())

    def on_round_end(round_idx: int) -> None:
        max_rounds = orch.team.workflow.max_rounds
        rf = Text()
        rf.append(f"  ✓  Round {round_idx + 1} complete", style="bold green")
        cons.print(rf)
        cons.print()
        if round_idx + 1 < max_rounds:
            state["pending_round_header"] = True
            state["pending_round"] = round_idx + 2

    def on_parallel_round_start(names: list) -> None:
        color_list = "  ".join(
            f"[bold {_get_member_color(n, orch.team.members)}]@{n}[/bold {_get_member_color(n, orch.team.members)}]"
            for n in names
        )
        ph = Text()
        ph.append("  ⚡  parallel  ", style="bold bright_blue")
        ph.append(color_list)
        cons.print(ph)
        cons.print(Text("  " + "─" * _SEP, style="bright_blue dim"))
        cons.print()

    def on_parallel_round_end(results: list) -> None:
        for name, result in results:
            color = _get_member_color(name, orch.team.members)
            try:
                member = orch.team.member(name)
                role = member.role
                model = member.model
            except KeyError:
                role = ""
                model = ""
            cons.print(
                _render_turn_block(
                    name, role, color,
                    result.content.strip() if result.content else "",
                    model=model,
                    files_written=result.files_written or [],
                )
            )
            cons.print()

    orch._on_turn_start = on_turn_start
    orch._on_token = on_token
    orch._on_turn_end = on_turn_end
    orch._on_tool_call = on_tool_call
    orch._on_tool_result = on_tool_result
    orch._on_round_end = on_round_end
    orch._on_parallel_round_start = on_parallel_round_start
    orch._on_parallel_round_end = on_parallel_round_end


def _setup_interactive(orch: Orchestrator, cons: Console) -> None:
    """Layer interactive prompts on top of the round-end hook.

    Chains with any existing ``_on_round_end`` (e.g. the round separator
    printed by ``_setup_streaming``) so both fire in order.
    """
    _prev = orch._on_round_end

    def on_round_end(round_idx: int) -> None:
        if _prev:
            _prev(round_idx)
        cons.print("  [dim]─ enter a directive, or press Enter to continue:[/dim] ", end="")
        try:
            text = input()
        except (EOFError, KeyboardInterrupt):
            return
        if text.strip():
            orch.inject_directive(text)
            inj = Text()
            inj.append("  ↳ directive injected", style="bold yellow")
            cons.print(inj)

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
    _rh = Text()
    _rh.append(f"  ▶  Round 1 / {cfg.workflow.max_rounds}", style="bold bright_blue")
    _rh.append(f"   {len(cfg.members)} members active", style="dim")
    console.print(_rh)
    console.print(Text("  " + "─" * _SEP, style="bright_blue dim"))
    console.print()
    _budget_hit = False
    _retry_exhausted = False
    try:
        orch.run()
    except Exception as _exc:
        from team.orchestrator import TokenBudgetError
        from team.ollama_client import LLMRetryExhaustedError
        if isinstance(_exc, TokenBudgetError):
            _budget_hit = True
            console.print()
            console.print(
                Panel(
                    f"[yellow]{_exc}[/yellow]\n\n"
                    "[dim]The run stopped before that member's next turn.  "
                    "Increase [bold]token_budget[/bold] in the team YAML "
                    "or use [bold]team run --resume[/bold] with a higher budget "
                    "to continue from where it left off.[/dim]",
                    title="[bold yellow]⚠ token budget exhausted[/bold yellow]",
                    border_style="yellow",
                )
            )
        elif isinstance(_exc, LLMRetryExhaustedError):
            _retry_exhausted = True
            console.print()
            console.print(
                Panel(
                    f"[red]{_exc}[/red]\n\n"
                    "[dim]All retry attempts failed.  Check that the Ollama container is "
                    "healthy and the model is loaded.  You can increase "
                    "[bold]max_retries[/bold] or [bold]retry_backoff[/bold] in the team "
                    "YAML to allow more attempts before giving up.[/dim]",
                    title="[bold red]✗ LLM connection failed[/bold red]",
                    border_style="red",
                )
            )
        else:
            raise
    finally:
        if not keep_up:
            orch.down()
    console.print()
    _print_token_summary(orch)
    if _retry_exhausted:
        _done = Text()
        _done.append("  ✗  Run failed", style="bold red")
        _done.append("   transcript → ", style="dim")
        _done.append(str(orch.transcript_path()), style="dim underline cyan")
    elif _budget_hit:
        _done = Text()
        _done.append("  ⚠  Budget exhausted", style="bold yellow")
        _done.append("   transcript → ", style="dim")
        _done.append(str(orch.transcript_path()), style="dim underline cyan")
    else:
        _done = Text()
        _done.append("  ✅  Done", style="bold green")
        _done.append("   transcript → ", style="dim")
        _done.append(str(orch.transcript_path()), style="dim underline cyan")
    console.print(_done)


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

    from team.config import resolve_member_setting
    from team.pricing import estimate_cost, format_cost

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
    table.add_column("Est. cost", justify="right")

    grand_cost: float | None = 0.0
    for speaker, count in sorted(s["turns_by_speaker"].items()):
        tok = s["tokens_by_speaker"].get(speaker, {"prompt": 0, "completion": 0})
        p_tok, c_tok = tok["prompt"], tok["completion"]
        try:
            mc = cfg.member(speaker)
            backend = resolve_member_setting(mc, cfg.defaults, "backend") or "ollama"
            cost = estimate_cost(mc.model, p_tok, c_tok, is_local=(backend == "ollama"))
        except (KeyError, AttributeError):
            cost = None
        if cost is None:
            grand_cost = None
        elif grand_cost is not None:
            grand_cost += cost
        table.add_row(
            f"@{speaker}",
            str(count),
            str(p_tok),
            str(c_tok),
            str(p_tok + c_tok),
            format_cost(cost),
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
        format_cost(grand_cost),
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
        task_ttl_seconds=cfg.bridge.task_ttl_seconds,
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

    # Auto-register with the configured registry if requested.
    _registry_client = None
    if cfg.registry.auto_register and cfg.registry.url:
        import socket as _socket
        from team.registry import RegistryEntry
        from team.registry_client import RegistryClient, RegistryClientError

        hostname = _socket.gethostname()
        bridge_url = f"http://{hostname}:{listen_port}"
        models = list({m.model for m in cfg.members})
        tools: list[str] = []
        for m in cfg.members:
            for t in (m.tools or cfg.defaults.tools or []):
                if t not in tools:
                    tools.append(t)
        entry = RegistryEntry(
            name=cfg.name,
            url=bridge_url,
            description=cfg.registry.description,
            tags=cfg.registry.tags,
            models=models,
            tools=tools,
        )
        _registry_client = RegistryClient(cfg.registry.url)
        try:
            _registry_client.register(entry)
            _registry_client.start_heartbeat(
                cfg.name,
                interval=float(cfg.registry.heartbeat_interval),
                entry=entry,
            )
            console.print(
                f"[green]registered[/green] with registry at "
                f"[bold]{cfg.registry.url}[/bold]  "
                f"[dim](heartbeat every {cfg.registry.heartbeat_interval}s)[/dim]"
            )
        except RegistryClientError as _exc:
            console.print(f"[yellow]⚠ registry auto-register failed:[/yellow] {_exc}")
            _registry_client = None

    console.print("[dim]Press Ctrl-C to stop.[/dim]")

    def _stop(sig: int, frame: object) -> None:  # noqa: ARG001
        console.print("\n[yellow]shutting down bridge server…[/yellow]")
        if _registry_client is not None:
            try:
                _registry_client.stop_heartbeat()
                _registry_client.deregister(cfg.name)
                console.print(
                    f"[dim]deregistered {cfg.name!r} from {cfg.registry.url}[/dim]"
                )
            except Exception:  # noqa: BLE001
                pass
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
        role = t.get("role", "")
        model = t.get("model", "")
        content = t.get("content", "")
        files = t.get("files_written") or []
        console.print(
            _render_turn_block(
                name, role, color, content,
                model=model,
                turn_index=t.get("index"),
                files_written=files,
            )
        )
        console.print()


# --------------------------------------------------------------------------- #
# export
# --------------------------------------------------------------------------- #


@cli.command()
@click.argument("team_file", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--format", "fmt",
    type=click.Choice(["markdown", "html", "json"], case_sensitive=False),
    default="markdown",
    show_default=True,
    help="Output format.",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(dir_okay=False, writable=True),
    default=None,
    help="Output file. Defaults to <workspace>/report.md / .html / .json.",
)
@click.option(
    "--no-artifacts",
    is_flag=True,
    default=False,
    help="Omit workspace files from the report (useful for large codebases).",
)
def export(team_file: str, fmt: str, output_path: str | None, no_artifacts: bool) -> None:
    """Export the run transcript and shared artifacts to a Markdown, HTML, or JSON report."""
    from team.export import export_run

    cfg = _load(team_file)
    transcript_p = cfg.workspace / "transcript.jsonl"
    if not transcript_p.is_file():
        console.print(f"[yellow]no transcript found[/yellow] at {transcript_p}")
        sys.exit(1)

    text = export_run(cfg, fmt=fmt, no_artifacts=no_artifacts)  # type: ignore[arg-type]

    if output_path is None:
        ext = {"html": "html", "json": "json"}.get(fmt, "md")
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
        console.print(f"[dim]{entry['description']}[/dim]")
        console.print(f"[dim]Source:[/dim] {entry.get('source', 'built-in')}\n")
        console.print(entry["persona"])
        console.print(
            f"\n[dim]Use in YAML:[/dim] [bold]persona: \"@{key}\"[/bold]"
        )
        return

    table = Table(title="Predefined persona library", show_lines=False)
    table.add_column("Source", style="dim", no_wrap=True)
    table.add_column("Key", style="bold cyan", no_wrap=True)
    table.add_column("Role", style="bold")
    table.add_column("Description")

    sorted_entries = sorted(list_all(), key=lambda e: (e.get("source", ""), e["key"]))
    prev_source: str | None = None
    for entry in sorted_entries:
        src = entry.get("source", "built-in")
        if prev_source is not None and src != prev_source:
            table.add_section()  # visible divider only between source groups
        table.add_row(
            src if src != prev_source else "",  # label only on first row of group
            f"@{entry['key']}",
            entry["role"],
            entry["description"],
        )
        prev_source = src
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
        _rh = Text()
        _rh.append(f"  ▶  Round 1 / {cfg.workflow.max_rounds}", style="bold bright_blue")
        _rh.append(f"   {len(cfg.members)} members active", style="dim")
        console.print(_rh)
        console.print(Text("  " + "─" * _SEP, style="bright_blue dim"))
        console.print()
        with console.status("[bold blue]starting containers and pulling models…[/bold blue]"):
            orch.up()
        console.print("[green]✓[/green] team is up\n")
        try:
            orch.run()
        except Exception as _exc:
            from team.orchestrator import TokenBudgetError
            from team.ollama_client import LLMRetryExhaustedError
            if isinstance(_exc, TokenBudgetError):
                console.print(f"\n[yellow]⚠ token budget exhausted:[/yellow] {_exc}")
            elif isinstance(_exc, LLMRetryExhaustedError):
                console.print(
                    f"\n[red]✗ LLM connection failed (all retries exhausted):[/red] {_exc}\n"
                    "[dim]Cannot run assertions — no transcript was produced.[/dim]"
                )
                sys.exit(1)
            else:
                raise
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


# --------------------------------------------------------------------------- #
# replay helpers (terminal I/O — Unix only)
# --------------------------------------------------------------------------- #


def _read_key_unix() -> str:
    """Read a single keypress from stdin in raw terminal mode (Unix only).

    Returns a normalised name:
    * printable character as-is (``'n'``, ``'q'``, …)
    * ``'ENTER'``, ``'SPACE'``, ``'BACKSPACE'``
    * ``'ARROW_LEFT'``, ``'ARROW_RIGHT'``, ``'ARROW_UP'``, ``'ARROW_DOWN'``
    * ``'ESCAPE'`` (bare Esc with no follow-up within 50 ms)
    * ``'UNKNOWN'`` for unrecognised escape sequences
    """
    import select
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.buffer.read(1).decode("utf-8", errors="replace")
        if ch == "\x1b":
            ready, _, _ = select.select([sys.stdin], [], [], 0.05)
            if ready:
                ch2 = sys.stdin.buffer.read(1).decode("utf-8", errors="replace")
                if ch2 == "[":
                    ready2, _, _ = select.select([sys.stdin], [], [], 0.05)
                    if ready2:
                        ch3 = sys.stdin.buffer.read(1).decode("utf-8", errors="replace")
                        return {
                            "A": "ARROW_UP",
                            "B": "ARROW_DOWN",
                            "C": "ARROW_RIGHT",
                            "D": "ARROW_LEFT",
                        }.get(ch3, "UNKNOWN")
                return "UNKNOWN"
            return "ESCAPE"
        if ch in ("\r", "\n"):
            return "ENTER"
        if ch == " ":
            return "SPACE"
        if ch in ("\x7f", "\x08"):
            return "BACKSPACE"
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _readline_raw_unix() -> str:
    """Read characters in raw mode until Enter, echoing them.

    Handles printable characters and backspace.  Returns the collected string
    without the trailing newline.  Used for inline prompts inside the replay
    interactive loop (no shell-level readline needed).
    """
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    chars: list[str] = []
    try:
        tty.setraw(fd)
        while True:
            ch = sys.stdin.buffer.read(1).decode("utf-8", errors="replace")
            if ch in ("\r", "\n"):
                sys.stdout.write("\n")
                sys.stdout.flush()
                break
            if ch in ("\x7f", "\x08"):
                if chars:
                    chars.pop()
                    sys.stdout.write("\b \b")
                    sys.stdout.flush()
            elif ch.isprintable():
                chars.append(ch)
                sys.stdout.write(ch)
                sys.stdout.flush()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return "".join(chars)


def _replay_render_turn(
    session: "Any",
    team_name: str,
    members: list,
    cons: Console,
    *,
    show_stats: bool = False,
) -> None:
    """Clear the screen and render the current turn (or stats) + footer."""
    import time as _time

    cons.clear()
    turn = session.current()
    if turn is None:
        cons.print("[yellow]No turns in transcript.[/yellow]")
        return

    color = _get_member_color(turn.speaker, members)
    total = session.total
    idx = session.index
    pct = f"{(idx + 1) / total * 100:.0f}%" if total else "—"

    header = Text()
    header.append("📼 replay", style="bold")
    header.append(f"  ·  {team_name}", style="dim")
    header.append("  ·  turn ")
    header.append(str(idx + 1), style="bold")
    header.append(f"/{total}  ({pct})", style="dim")
    header.append("  ·  ")
    header.append(f"@{turn.speaker}", style=f"bold {color}")
    header.append(f"  {turn.role}", style="dim")
    cons.print(Rule(header, style="dim"))
    cons.print()

    if show_stats:
        s = session.stats()
        dur = s["duration_seconds"]
        dur_str = f"{dur:.1f}s" if dur is not None else "—"
        cons.print(
            f"[bold]Totals:[/bold]  {s['total_turns']} turns  ·  "
            f"{s['total_prompt_tokens'] + s['total_completion_tokens']:,} tokens  ·  "
            f"{s['files_written']} file(s) written  ·  duration {dur_str}\n"
        )
        tbl = Table(show_lines=False, box=None, padding=(0, 2), show_header=True)
        tbl.add_column("Speaker", style="bold")
        tbl.add_column("Turns", justify="right")
        tbl.add_column("Prompt", justify="right")
        tbl.add_column("Completion", justify="right")
        tbl.add_column("Total", justify="right")
        for sp, count in sorted(s["turns_by_speaker"].items()):
            tok = s["tokens_by_speaker"].get(sp, {"prompt": 0, "completion": 0})
            sp_color = _get_member_color(sp, members)
            tbl.add_row(
                f"[{sp_color}]@{sp}[/{sp_color}]",
                str(count),
                str(tok["prompt"]),
                str(tok["completion"]),
                str(tok["prompt"] + tok["completion"]),
            )
        cons.print(tbl)
    else:
        cons.print(
            _render_turn_block(
                turn.speaker, turn.role, color,
                turn.content.rstrip(),
                model=getattr(turn, "model", ""),
                turn_index=turn.index,
                prompt_tokens=turn.prompt_tokens or 0,
                completion_tokens=turn.completion_tokens or 0,
                files_written=turn.files_written or [],
            )
        )

    # Footer ---------------------------------------------------------------- #
    cons.print()
    at_start = session.at_start
    at_end = session.at_end
    prev_style = "dim" if not at_start else "dim"
    next_style = "dim" if not at_end else "dim"
    nav_prev = f"[{prev_style}]←/p[/{prev_style}] prev"
    nav_next = f"[{next_style}]→/n[/{next_style}] next"
    s_label = "[bold]stats ✓[/bold]" if show_stats else "stats"
    cons.print(
        f"  {nav_prev}   {nav_next}"
        f"   [dim]g[/dim] goto"
        f"   [dim]f[/dim] find speaker"
        f"   [dim]s[/dim] {s_label}"
        f"   [dim]q[/dim] quit",
        highlight=False,
    )


# --------------------------------------------------------------------------- #
# pipeline command
# --------------------------------------------------------------------------- #


@cli.command()
@click.argument("pipeline_file", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Validate the pipeline YAML and print the execution plan without running anything.",
)
def pipeline(pipeline_file: str, dry_run: bool) -> None:
    """Run a multi-team pipeline defined in a YAML file.

    A pipeline chains multiple team runs in sequence, automatically passing
    shared workspace files and transcript summaries from one stage to the next.

    Example pipeline.yaml::

    \b
        name: research-and-write
        stages:
          - id: research
            team: ./teams/researcher.yaml
          - id: writing
            team: ./teams/writer.yaml
            depends_on: [research]
            inject_files: true
            inject_context: true
            goal_override: |
              Write a paper based on: {research.summary}
    """
    from team.pipeline import PipelineError, PipelineRunner, _toposort, load_pipeline

    try:
        cfg = load_pipeline(pipeline_file)
    except PipelineError as exc:
        console.print(f"[red]pipeline error:[/red] {exc}")
        sys.exit(1)

    order = _toposort(cfg.stages)
    stage_by_id = {s.id: s for s in cfg.stages}

    # Print execution plan.
    console.print()
    console.print(
        Panel(
            f"[bold]{cfg.name}[/bold]\n[dim]{cfg.description}[/dim]\n\n"
            + "\n".join(
                f"  [bold bright_cyan]{i + 1}.[/bold bright_cyan] "
                f"[bold]{sid}[/bold]"
                + (
                    f"  [dim]← {', '.join(stage_by_id[sid].depends_on)}[/dim]"
                    if stage_by_id[sid].depends_on
                    else ""
                )
                + f"\n     [dim]{stage_by_id[sid].team_file}[/dim]"
                for i, sid in enumerate(order)
            ),
            title="[bold blue]🔗  pipeline[/bold blue]",
            border_style="blue",
            padding=(1, 2),
        )
    )
    console.print()

    if dry_run:
        console.print("[yellow]dry-run:[/yellow] pipeline validated — no stages were executed.")
        return

    runner = PipelineRunner(cfg, console=console)
    try:
        results = runner.run()
    except PipelineError as exc:
        console.print(f"\n[red]✗ pipeline failed:[/red] {exc}")
        sys.exit(1)

    # Summary table.
    table = Table(title="Pipeline results")
    table.add_column("Stage", style="bold")
    table.add_column("Status")
    table.add_column("Duration", justify="right")
    table.add_column("Artifacts", justify="right")
    table.add_column("Workspace")

    for r in results:
        status = "[green]✓ done[/green]" if r.success else "[red]✗ failed[/red]"
        table.add_row(
            r.stage_id,
            status,
            f"{r.duration_seconds:.1f}s",
            str(len(r.artifacts)),
            str(r.workspace),
        )

    console.print(table)
    console.print()
    console.print(f"[green]✓ pipeline complete[/green]  ·  {len(results)} stage(s)  ·  workspace → {cfg.workspace}")


# --------------------------------------------------------------------------- #
# replay command
# --------------------------------------------------------------------------- #


@cli.command()
@click.argument("team_file", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--from",
    "start_turn",
    default=0,
    type=int,
    show_default=True,
    help="Start at turn N (0-based index).",
)
@click.option(
    "--speaker",
    default=None,
    metavar="NAME",
    help="Jump to the first turn by SPEAKER at startup.",
)
def replay(team_file: str, start_turn: int, speaker: str | None) -> None:
    """Browse a saved transcript turn-by-turn in an interactive viewer.

    Loads the persisted transcript from the team's workspace and lets you
    step through it one turn at a time using keyboard shortcuts.

    \b
    Navigation keys:
      → / n / Space / Enter   advance to next turn
      ← / p / b               go back one turn
      g                        jump to a specific turn number
      f                        find the next turn by a given speaker
      s                        toggle the stats summary panel
      q / Esc                  quit

    Non-interactive mode (piped stdin): all turns are printed and the
    command exits immediately — useful for scripting.

    \b
    Examples:
      team replay myteam.yaml
      team replay myteam.yaml --from 5
      team replay myteam.yaml --speaker alice
    """
    import time as _time

    from team.replay import ReplaySession, load_transcript

    cfg = _load(team_file)
    transcript_p = cfg.workspace / "transcript.jsonl"
    turns = load_transcript(transcript_p)

    if not turns:
        console.print(f"[yellow]no transcript found[/yellow] at {transcript_p}")
        console.print("[dim]Run the team first: team run myteam.yaml[/dim]")
        return

    session = ReplaySession(turns)

    # Apply startup positioning -------------------------------------------- #
    if speaker:
        # Jump to the first turn by the requested speaker.
        if turns[0].speaker.lower() == speaker.lower():
            pass  # already there
        elif not session.find_next(speaker):
            console.print(f"[yellow]no turns by @{speaker} in transcript[/yellow]")
            return
    elif start_turn > 0:
        if not session.jump(start_turn):
            console.print(
                f"[red]turn {start_turn} out of range "
                f"(transcript has {session.total} turn(s), 0–{session.total - 1})[/red]"
            )
            return

    # Non-interactive fallback (e.g. piped stdin) -------------------------- #
    if not sys.stdin.isatty():
        for t in turns:
            color = _get_member_color(t.speaker, cfg.members)
            console.print(
                _render_turn_block(
                    t.speaker, t.role, color, t.content,
                    model=getattr(t, "model", ""),
                    turn_index=t.index,
                    files_written=t.files_written or [],
                )
            )
            console.print()
        return

    # Interactive mode (requires Unix terminal) ---------------------------- #
    try:
        import termios  # noqa: F401
        import tty  # noqa: F401
    except ImportError:
        console.print(
            "[yellow]Interactive replay requires a Unix terminal "
            "(termios/tty not available).[/yellow]"
        )
        console.print("[dim]Falling back to non-interactive output.[/dim]")
        for t in turns:
            color = _get_member_color(t.speaker, cfg.members)
            console.print(
                _render_turn_block(
                    t.speaker, t.role, color, t.content,
                    model=getattr(t, "model", ""),
                    turn_index=t.index,
                    files_written=t.files_written or [],
                )
            )
            console.print()
        return

    show_stats = False
    while True:
        _replay_render_turn(session, cfg.name, cfg.members, console, show_stats=show_stats)

        try:
            key = _read_key_unix()
        except Exception:
            break

        if key in ("n", "SPACE", "ENTER", "ARROW_RIGHT"):
            show_stats = False
            if not session.next():
                console.print()
                console.print("[dim]  ─── end of transcript ───[/dim]")
                _time.sleep(0.7)
        elif key in ("p", "b", "ARROW_LEFT"):
            show_stats = False
            session.prev()
        elif key == "g":
            console.print(
                f"\n  [bold]Go to turn[/bold] [dim](0–{session.total - 1}):[/dim] ",
                end="",
            )
            sys.stdout.flush()
            raw = _readline_raw_unix().strip()
            try:
                target = int(raw)
                if not session.jump(target):
                    console.print(f"[red]  out of range (0–{session.total - 1})[/red]")
                    _time.sleep(0.6)
            except ValueError:
                pass
            show_stats = False
        elif key == "f":
            console.print("\n  [bold]Find speaker[/bold] [dim](name):[/dim] ", end="")
            sys.stdout.flush()
            raw = _readline_raw_unix().strip()
            if raw:
                # Try forward first; if nothing found, try backward (wrap).
                if not session.find_next(raw):
                    if not session.find_prev(raw):
                        console.print(f"[yellow]  no turns by @{raw}[/yellow]")
                        _time.sleep(0.6)
            show_stats = False
        elif key == "s":
            show_stats = not show_stats
        elif key in ("q", "ESCAPE"):
            break
        # Any other key — re-render silently.

    console.clear()
    console.print(
        f"[dim]replay ended  ·  team [bold]{cfg.name}[/bold]  ·  {session.total} turn(s)[/dim]"
    )



# --------------------------------------------------------------------------- #
# forge
# --------------------------------------------------------------------------- #


def _forge_normalize(name: str) -> tuple[str, str, str]:
    """Return (dir_name, pkg_name, ext_name) from a raw extension name.

    Accepts either "bioinformatics" or "team-bioinformatics".
    Returns e.g. ("team-bioinformatics", "team_bioinformatics", "bioinformatics").
    """
    name = name.strip().lower().replace(" ", "-")
    ext_name = name.removeprefix("team-")
    ext_name = ext_name.replace("-", "_")
    if not ext_name.isidentifier():
        raise click.BadParameter(
            f"{ext_name!r} is not a valid Python identifier.  "
            "Use only letters, digits, and underscores."
        )
    dir_name = f"team-{ext_name.replace('_', '-')}"
    pkg_name = f"team_{ext_name}"
    return dir_name, pkg_name, ext_name


def _forge_files(dir_name: str, pkg_name: str, ext_name: str) -> dict[str, str]:
    """Return a mapping of relative path -> file content for a new extension scaffold."""
    import datetime as _dt
    ext_title = ext_name.replace("_", " ").title()
    hyphen_name = ext_name.replace("_", "-")
    year = _dt.date.today().year

    pyproject = (
        "[build-system]\n"
        "requires = [\"setuptools>=64\", \"wheel\"]\n"
        "build-backend = \"setuptools.build_meta\"\n"
        "\n"
        "[project]\n"
        f'name = "team-{hyphen_name}"\n'
        "version = \"0.1.0\"\n"
        f'description = "{ext_title} extensions for the team multi-agent LLM framework."\n'
        "readme = \"README.md\"\n"
        "license = { file = \"LICENSE\" }\n"
        "requires-python = \">=3.9\"\n"
        "authors = [{ name = \"Your Name\" }]\n"
        f'keywords = ["llm", "{hyphen_name}", "multi-agent", "team"]\n'
        "dependencies = [\n"
        "    \"team-core>=0.15.5\",\n"
        "    \"click>=8.1\",\n"
        "    \"rich>=13.0\",\n"
        "]\n"
        "\n"
        "[project.optional-dependencies]\n"
        "dev = [\n"
        "    \"pytest>=7.0\",\n"
        "    \"pytest-mock>=3.10\",\n"
        "]\n"
        "\n"
        "[project.entry-points.\"team.persona_dirs\"]\n"
        f'{ext_name} = "{pkg_name}:personas_dir"\n'
        "\n"
        "[project.entry-points.\"team.skills\"]\n"
        f'# my_skill = "{pkg_name}.skills.my_skill"\n'
        "\n"
        "[project.entry-points.\"team.commands\"]\n"
        f'{ext_name} = "{pkg_name}.commands:{ext_name}"\n'
        "\n"
        "[project.urls]\n"
        f'Homepage = "https://github.com/YOUR_USERNAME/team-{hyphen_name}"\n'
        f'Issues = "https://github.com/YOUR_USERNAME/team-{hyphen_name}/issues"\n'
        "\n"
        "[tool.pytest.ini_options]\n"
        "markers = [\n"
        "    \"integration: end-to-end tests that require external services\",\n"
        "]\n"
        "\n"
        "[tool.setuptools.packages.find]\n"
        f'include = ["{pkg_name}*"]\n'
        "exclude = [\"tests*\", \"examples*\"]\n"
        "\n"
        "[tool.setuptools.package-data]\n"
        f'{pkg_name} = ["skills/*.py", "skills/*.md", "personas/*.yaml", "examples/*.yaml"]\n'
    )

    init_py = (
        f'"""{pkg_name} -- {ext_title} extensions for the team multi-agent LLM framework.\n'
        "\n"
        "Extension points registered\n"
        "---------------------------\n"
        "``team.skills``\n"
        "    Short skill names usable anywhere in a team YAML ``skills:`` list.\n"
        "\n"
        "``team.persona_dirs``\n"
        f'    Personas in ``{pkg_name}/personas/`` are auto-discovered with the ``@name`` shorthand.\n'
        "\n"
        "``team.commands``\n"
        f'    ``team {ext_name}`` subcommand group injected into the ``team`` CLI.\n'
        '"""\n'
        "\n"
        "from __future__ import annotations\n"
        "\n"
        "from pathlib import Path\n"
        "\n"
        "\n"
        "def skills_dir() -> Path:\n"
        f'    """Return the absolute path to the {pkg_name} skills directory."""\n'
        "    return Path(__file__).parent / \"skills\"\n"
        "\n"
        "\n"
        "def personas_dir() -> Path:\n"
        f'    """Return the absolute path to the {pkg_name} personas directory."""\n'
        "    return Path(__file__).parent / \"personas\"\n"
        "\n"
        "\n"
        "def examples_dir() -> Path:\n"
        f'    """Return the absolute path to the {pkg_name} examples directory."""\n'
        "    return Path(__file__).parent / \"examples\"\n"
        "\n"
        "\n"
        "__all__ = [\"skills_dir\", \"personas_dir\", \"examples_dir\"]\n"
    )

    commands_py = (
        f'"""{ext_name} subcommand group -- accessible as ``team {ext_name} <cmd>``."""\n'
        "\n"
        "from __future__ import annotations\n"
        "\n"
        "from team.extension import make_extension_commands\n"
        f'from {pkg_name} import examples_dir, personas_dir, skills_dir\n'
        "\n"
        "_SKILL_DESCRIPTIONS: dict[str, str] = {\n"
        "    # \"my_skill\": \"Does something useful.\",\n"
        "}\n"
        "\n"
        "_SCENARIO_DESCRIPTIONS: dict[str, str] = {\n"
        "    # \"my-scenario\": \"Runs a multi-step analysis.\",\n"
        "}\n"
        "\n"
        f'{ext_name} = make_extension_commands(\n'
        f'    package_name="team-{hyphen_name}",\n'
        f'    group_name="{ext_name}",\n'
        f'    description="{ext_title} extensions for the team multi-agent framework.",\n'
        "    skills_dir=skills_dir,\n"
        "    personas_dir=personas_dir,\n"
        "    examples_dir=examples_dir,\n"
        "    skill_descriptions=_SKILL_DESCRIPTIONS,\n"
        "    scenario_descriptions=_SCENARIO_DESCRIPTIONS,\n"
        ")\n"
    )

    test_py = (
        f'"""Standard extension package tests for team-{hyphen_name}."""\n'
        "\n"
        "from __future__ import annotations\n"
        "\n"
        "from pathlib import Path\n"
        "\n"
        "import pytest\n"
        "import yaml\n"
        "from click.testing import CliRunner\n"
        "\n"
        f'from {pkg_name} import examples_dir, personas_dir, skills_dir\n'
        f'from {pkg_name}.commands import {ext_name}\n'
        "\n"
        "\n"
        "def test_skills_dir_exists():\n"
        "    assert skills_dir().is_dir()\n"
        "\n"
        "\n"
        "def test_personas_dir_exists():\n"
        "    assert personas_dir().is_dir()\n"
        "\n"
        "\n"
        "def test_examples_dir_exists():\n"
        "    assert examples_dir().is_dir()\n"
        "\n"
        "\n"
        "@pytest.mark.parametrize(\n"
        "    \"yaml_path\",\n"
        f'    list((Path(__file__).parent.parent / "{pkg_name}" / "personas").glob("*.yaml")),\n'
        "    ids=lambda p: p.stem,\n"
        ")\n"
        "def test_persona_yaml_has_required_fields(yaml_path):\n"
        "    raw = yaml.safe_load(yaml_path.read_text(encoding=\"utf-8\"))\n"
        "    assert isinstance(raw, dict)\n"
        "    for field in (\"role\", \"description\", \"persona\"):\n"
        "        assert field in raw, f\"{yaml_path.name} missing {field!r}\"\n"
        "\n"
        "\n"
        "@pytest.mark.parametrize(\n"
        "    \"yaml_path\",\n"
        f'    list((Path(__file__).parent.parent / "{pkg_name}" / "examples").glob("*.yaml")),\n'
        "    ids=lambda p: p.stem,\n"
        ")\n"
        "def test_example_yaml_parseable(yaml_path):\n"
        "    raw = yaml.safe_load(yaml_path.read_text(encoding=\"utf-8\"))\n"
        "    assert isinstance(raw, dict)\n"
        "    for key in (\"name\", \"goal\", \"workflow\", \"members\"):\n"
        "        assert key in raw\n"
        "    assert isinstance(raw[\"members\"], list) and len(raw[\"members\"]) >= 1\n"
        "\n"
        "\n"
        "def test_cli_help():\n"
        f'    result = CliRunner().invoke({ext_name}, ["--help"])\n'
        "    assert result.exit_code == 0\n"
        "\n"
        "\n"
        "def test_cli_scenarios():\n"
        f'    result = CliRunner().invoke({ext_name}, ["scenarios"])\n'
        "    assert result.exit_code == 0\n"
        "\n"
        "\n"
        "def test_cli_skills():\n"
        f'    result = CliRunner().invoke({ext_name}, ["skills"])\n'
        "    assert result.exit_code == 0\n"
        "\n"
        "\n"
        "def test_cli_personas():\n"
        f'    result = CliRunner().invoke({ext_name}, ["personas"])\n'
        "    assert result.exit_code == 0\n"
        "\n"
        "\n"
        "def test_cli_init_creates_file(tmp_path):\n"
        "    if not list(examples_dir().glob(\"*.yaml\")):\n"
        "        pytest.skip(\"no example scenarios defined yet\")\n"
        "    scenario = sorted(examples_dir().glob(\"*.yaml\"))[0].stem\n"
        f'    result = CliRunner().invoke({ext_name}, ["init", "--scenario", scenario, "--output-dir", str(tmp_path)])\n'
        "    assert result.exit_code == 0, result.output\n"
        "    assert (tmp_path / f\"{scenario}.yaml\").is_file()\n"
    )

    readme = (
        f'# team-{hyphen_name}\n'
        "\n"
        f'{ext_title} extensions for the [team](https://github.com/cumbof/team) multi-agent LLM framework.\n'
        "\n"
        "## Installation\n"
        "\n"
        "```bash\n"
        f'pip install team-{hyphen_name}\n'
        "```\n"
        "\n"
        "## Quick Start\n"
        "\n"
        "```bash\n"
        f'team {ext_name} scenarios\n'
        f'team {ext_name} init --scenario <scenario-name>\n'
        "team run <scenario-name>.yaml\n"
        "```\n"
        "\n"
        "## Skills\n"
        "\n"
        "```yaml\n"
        "defaults:\n"
        "  skills:\n"
        "    - my_skill\n"
        "```\n"
        "\n"
        "## Personas\n"
        "\n"
        "```yaml\n"
        "members:\n"
        "  - name: agent\n"
        "    persona: \"@my_persona\"\n"
        "```\n"
        "\n"
        "## How it extends team-core\n"
        "\n"
        "```\n"
        "team-core plugin API\n"
        f'|- team.skills          <- registered skill names\n'
        f'|- team.persona_dirs    <- {pkg_name}/personas/ (auto-merged)\n'
        f'`- team.commands        <- team {ext_name} <subcommand>\n'
        "```\n"
        "\n"
        "## Development\n"
        "\n"
        "```bash\n"
        "pip install -e \".[dev]\"\n"
        "pytest -q\n"
        "```\n"
    )

    gitignore = (
        "__pycache__/\n"
        "*.egg-info/\n"
        "*.egg\n"
        ".pytest_cache/\n"
        "dist/\n"
        "build/\n"
        ".venv/\n"
    )

    license_txt = (
        "MIT License\n"
        "\n"
        f"Copyright (c) {year} Your Name\n"
        "\n"
        "Permission is hereby granted, free of charge, to any person obtaining a copy\n"
        "of this software and associated documentation files (the \"Software\"), to deal\n"
        "in the Software without restriction, including without limitation the rights\n"
        "to use, copy, modify, merge, publish, distribute, sublicense, and/or sell\n"
        "copies of the Software, and to permit persons to whom the Software is\n"
        "furnished to do so, subject to the following conditions:\n"
        "\n"
        "The above copyright notice and this permission notice shall be included in all\n"
        "copies or substantial portions of the Software.\n"
        "\n"
        "THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\n"
        "IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\n"
        "FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE\n"
        "AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\n"
        "LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\n"
        "OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE\n"
        "SOFTWARE.\n"
    )

    return {
        "pyproject.toml": pyproject,
        f"{pkg_name}/__init__.py": init_py,
        f"{pkg_name}/commands.py": commands_py,
        f"{pkg_name}/skills/.gitkeep": "",
        f"{pkg_name}/personas/.gitkeep": "",
        f"{pkg_name}/examples/.gitkeep": "",
        "tests/__init__.py": "",
        "tests/test_package.py": test_py,
        "README.md": readme,
        "LICENSE": license_txt,
        ".gitignore": gitignore,
    }


@cli.command("forge")
@click.argument("name")
@click.option(
    "--output-dir", "-o", default=".", show_default=True,
    help="Parent directory in which to create the extension package directory.",
)
@click.option(
    "--force", is_flag=True, default=False, help="Overwrite an existing directory.",
)
def forge(name: str, output_dir: str, force: bool) -> None:
    """Scaffold a new team-* extension package.

    NAME is the extension name -- either "bioinformatics" or the full
    "team-bioinformatics".  The package directory is created as
    team-<name>/ inside OUTPUT_DIR.

    The generated package is a minimal but complete team-* extension:

    \b
    team-<name>/
      pyproject.toml          entry points pre-wired, version 0.1.0
      team_<name>/
        __init__.py           skills_dir / personas_dir / examples_dir helpers
        commands.py           make_extension_commands() call
        skills/               drop .py and .md skill files here
        personas/             drop persona YAML files here
        examples/             drop scenario YAML templates here
      tests/
        test_package.py       standard scaffold tests (pass immediately)
      README.md
      LICENSE                 MIT
      .gitignore

    After scaffolding:

    \b
      cd team-<name>
      pip install -e ".[dev]"
      pytest -q            # all tests pass out of the box
      team <name> --help   # subcommand registered immediately
    """
    try:
        dir_name, pkg_name, ext_name = _forge_normalize(name)
    except click.BadParameter as exc:
        raise click.UsageError(str(exc)) from exc

    dest_root = Path(output_dir).expanduser().resolve() / dir_name
    if dest_root.exists() and not force:
        console.print(f"[yellow]{dest_root} already exists.[/yellow]  Use --force to overwrite.")
        raise SystemExit(1)

    files = _forge_files(dir_name, pkg_name, ext_name)
    created: list[str] = []
    for rel_path, content in files.items():
        full = dest_root / rel_path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content, encoding="utf-8")
        created.append(rel_path)

    console.print(f"\n[green bold]Created[/green bold] [bold]{dest_root}[/bold]\n")
    for rel in created:
        console.print(f"  [dim]{rel}[/dim]")
    console.print()
    console.print("Next steps:")
    console.print(f"  [bold]cd {dir_name}[/bold]")
    console.print(f"  [bold]pip install -e \".[dev]\"[/bold]")
    console.print(f"  [bold]pytest -q[/bold]                # all tests pass out of the box")
    console.print(f"  [bold]team {ext_name} --help[/bold]   # your new subcommand")


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #


@cli.group()
def registry() -> None:
    """Manage the team service-discovery registry.

    The registry is a lightweight HTTP directory where running team clusters
    can advertise their capabilities (tags, models, bridge URL) so other teams
    can discover and delegate work to them.

    Common workflow:

    \\b
        # Start a standalone registry server on port 8000
        team registry start --port 8000

        # Register a team with the registry
        team registry register myteam.yaml http://localhost:8000

        # Find teams that handle biology and statistics
        team registry find http://localhost:8000 --tag biology --tag statistics

        # List all registered teams
        team registry list http://localhost:8000
    """


@registry.command(name="start")
@click.option("--port", default=8000, show_default=True, help="TCP port to listen on.")
@click.option("--host", default="0.0.0.0", show_default=True, help="Bind address.")
@click.option("--secret", default=None, help="Optional HMAC shared secret for write operations.")
@click.option(
    "--entry-ttl",
    "entry_ttl",
    default=300,
    show_default=True,
    type=int,
    help="Seconds after which an entry without a heartbeat is evicted (0 = never).",
)
def registry_start(port: int, host: str, secret: str | None, entry_ttl: int) -> None:
    """Start a standalone team registry server.

    The server listens for team registrations and answers discovery queries.
    Any team can register itself with ``team registry register`` or by setting
    ``registry.auto_register: true`` and ``registry.url`` in its YAML.

    Example:

    \\b
        team registry start --port 8000 --secret mysharedsecret
    """
    import signal

    from team.registry_server import RegistryServer

    server = RegistryServer(
        port=port,
        host=host,
        secret=secret,
        entry_ttl_seconds=float(entry_ttl),
    )
    server.start()
    console.print(
        f"[green]registry server started[/green] — "
        f"listening on [bold]{host}:{port}[/bold]"
    )
    if entry_ttl:
        console.print(f"[dim]entry TTL: {entry_ttl}s  ·  auth: {'yes' if secret else 'no'}[/dim]")
    else:
        console.print("[dim]entry TTL: disabled  ·  entries live forever[/dim]")
    console.print("[dim]Press Ctrl-C to stop.[/dim]")

    def _stop(sig: int, frame: object) -> None:  # noqa: ARG001
        console.print("\n[yellow]shutting down registry server…[/yellow]")
        server.stop()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    server.join()
    console.print("[green]registry server stopped[/green]")


@registry.command(name="register")
@click.argument("team_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("registry_url")
@click.option("--secret", default=None, help="HMAC shared secret (required when server uses --secret).")
@click.option(
    "--heartbeat",
    "heartbeat_interval",
    default=0,
    type=int,
    show_default=True,
    help="Send heartbeats every N seconds to keep the registration alive (0 = no heartbeat).",
)
def registry_register(
    team_file: str,
    registry_url: str,
    secret: str | None,
    heartbeat_interval: int,
) -> None:
    """Register a team with a registry server.

    Reads team name, description, tags, models, and tools from the team YAML
    and registers the team's bridge URL with the registry at REGISTRY_URL.
    The bridge.listen_port from the YAML is used to compute the bridge URL
    unless the YAML sets ``registry.url`` (which is used as the team's
    advertised bridge URL).

    Example:

    \\b
        team registry register myteam.yaml http://registry.example.com:8000
    """
    import socket

    from team.registry import RegistryEntry
    from team.registry_client import RegistryClient, RegistryClientError

    cfg = _load(team_file)

    # Build the advertised bridge URL.
    # Priority: explicit registry.url in YAML > compute from bridge.listen_port.
    bridge_url = cfg.registry.url
    if not bridge_url:
        hostname = socket.gethostname()
        bridge_url = f"http://{hostname}:{cfg.bridge.listen_port}"

    # Collect tools and models.
    models = list({m.model for m in cfg.members})
    tools: list[str] = []
    for m in cfg.members:
        for t in (m.tools or cfg.defaults.tools or []):
            if t not in tools:
                tools.append(t)

    tags = cfg.registry.tags
    description = cfg.registry.description

    entry = RegistryEntry(
        name=cfg.name,
        url=bridge_url,
        description=description,
        tags=tags,
        models=models,
        tools=tools,
    )

    effective_secret = secret or (cfg.registry.url and None)
    client = RegistryClient(registry_url, secret=effective_secret)
    try:
        stored = client.register(entry)
    except RegistryClientError as exc:
        console.print(f"[red]registration failed:[/red] {exc}")
        sys.exit(1)

    console.print(
        f"[green]registered[/green] team [bold]{stored.name}[/bold] "
        f"with registry at [bold]{registry_url}[/bold]"
    )
    console.print(f"  bridge URL: {stored.url}")
    if stored.tags:
        console.print(f"  tags: {', '.join(stored.tags)}")
    if stored.models:
        console.print(f"  models: {', '.join(stored.models)}")

    if heartbeat_interval > 0:
        console.print(
            f"[dim]Sending heartbeats every {heartbeat_interval}s — "
            "press Ctrl-C to stop.[/dim]"
        )
        import signal
        client.start_heartbeat(cfg.name, interval=float(heartbeat_interval), entry=stored)

        def _stop(sig: int, frame: object) -> None:  # noqa: ARG001
            client.stop_heartbeat()
            console.print("\n[yellow]heartbeat stopped[/yellow]")
            sys.exit(0)

        signal.signal(signal.SIGINT, _stop)
        signal.signal(signal.SIGTERM, _stop)
        import threading
        threading.Event().wait()  # block until signal


@registry.command(name="deregister")
@click.argument("team_name")
@click.argument("registry_url")
@click.option("--secret", default=None, help="HMAC shared secret.")
def registry_deregister(team_name: str, registry_url: str, secret: str | None) -> None:
    """Remove a team from the registry.

    Example:

    \\b
        team registry deregister myteam http://localhost:8000
    """
    from team.registry_client import RegistryClient, RegistryClientError

    client = RegistryClient(registry_url, secret=secret)
    try:
        client.deregister(team_name)
    except RegistryClientError as exc:
        console.print(f"[red]deregistration failed:[/red] {exc}")
        sys.exit(1)
    console.print(
        f"[green]deregistered[/green] [bold]{team_name}[/bold] from {registry_url}"
    )


@registry.command(name="list")
@click.argument("registry_url")
def registry_list(registry_url: str) -> None:
    """List all teams currently registered with the registry.

    Example:

    \\b
        team registry list http://localhost:8000
    """
    from team.registry_client import RegistryClient, RegistryClientError

    client = RegistryClient(registry_url)
    try:
        entries = client.list_all()
    except RegistryClientError as exc:
        console.print(f"[red]registry error:[/red] {exc}")
        sys.exit(1)

    if not entries:
        console.print("[yellow]No teams registered.[/yellow]")
        return

    table = Table(
        title=f"Registered teams — {registry_url}",
        border_style="dim",
        header_style="bold dim",
    )
    table.add_column("Name", style="bold")
    table.add_column("Bridge URL")
    table.add_column("Tags")
    table.add_column("Models")
    table.add_column("Description")

    for e in entries:
        table.add_row(
            e.name,
            e.url,
            ", ".join(e.tags) if e.tags else "—",
            ", ".join(e.models) if e.models else "—",
            (e.description[:50] + "…") if len(e.description) > 50 else (e.description or "—"),
        )

    console.print(table)


@registry.command(name="find")
@click.argument("registry_url")
@click.option("--tag", "tags", multiple=True, help="Required capability tag (repeat for AND logic).")
@click.option("--keyword", default=None, help="Substring searched across name, description, and tags.")
def registry_find(registry_url: str, tags: tuple[str, ...], keyword: str | None) -> None:
    """Find teams in the registry matching given tags or a keyword.

    Example:

    \\b
        team registry find http://localhost:8000 --tag biology --tag python
        team registry find http://localhost:8000 --keyword "survival analysis"
    """
    from team.registry_client import RegistryClient, RegistryClientError

    client = RegistryClient(registry_url)
    try:
        entries = client.query(tags=list(tags) or None, keyword=keyword)
    except RegistryClientError as exc:
        console.print(f"[red]registry error:[/red] {exc}")
        sys.exit(1)

    if not entries:
        filters = []
        if tags:
            filters.append(f"tags=[{', '.join(tags)}]")
        if keyword:
            filters.append(f"keyword={keyword!r}")
        desc = " and ".join(filters) if filters else "(no filters)"
        console.print(f"[yellow]No teams match {desc}.[/yellow]")
        return

    table = Table(
        title=f"Matching teams — {registry_url}",
        border_style="dim",
        header_style="bold dim",
    )
    table.add_column("Name", style="bold")
    table.add_column("Bridge URL")
    table.add_column("Tags")
    table.add_column("Models")
    table.add_column("Description")

    for e in entries:
        table.add_row(
            e.name,
            e.url,
            ", ".join(e.tags) if e.tags else "—",
            ", ".join(e.models) if e.models else "—",
            (e.description[:50] + "…") if len(e.description) > 50 else (e.description or "—"),
        )

    console.print(table)
    console.print(
        f"\n[dim]{len(entries)} result(s). Use [bold]delegate_task[/bold] with the bridge URL "
        "to route work to any of these teams.[/dim]"
    )


if __name__ == "__main__":  # pragma: no cover
    cli()
