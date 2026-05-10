"""Workflow diagram renderer (F9).

Supports two output formats:

* ``ascii``   — plain-text box-and-arrow diagram rendered with Rich markup.
* ``mermaid`` — Mermaid flowchart syntax (paste into any Mermaid renderer or
                GitHub Markdown fenced block).

Usage::

    from team.visualize import render_diagram
    print(render_diagram(cfg, fmt="ascii"))
"""

from __future__ import annotations

from team.config import TeamConfig


# --------------------------------------------------------------------------- #
# ASCII renderer
# --------------------------------------------------------------------------- #

def _box(label: str, width: int = 24) -> list[str]:
    """Return a 3-line box containing *label* (Rich markup allowed)."""
    inner = f" {label} "
    pad = max(0, width - len(inner))
    left = pad // 2
    right = pad - left
    bar = "─" * (width + 2)
    return [
        f"┌{bar}┐",
        f"│{' ' * left}{inner}{' ' * right}│",
        f"└{bar}┘",
    ]


def _ascii_round_robin(team: TeamConfig) -> str:
    names = [f"@{m.name}\n{m.role}" for m in team.members]
    lines: list[str] = [
        f"[bold]Workflow:[/bold] round_robin  "
        f"(max_rounds={team.workflow.max_rounds})\n",
    ]
    for i, m in enumerate(team.members):
        lines.append(f"  ┌─────────────────────┐")
        lines.append(f"  │ [cyan]@{m.name:<19}[/cyan] │")
        lines.append(f"  │ {m.role:<21} │")
        lines.append(f"  └─────────────────────┘")
        if i < len(team.members) - 1:
            lines.append(f"           │")
            lines.append(f"           ▼")
    lines.append(f"           │")
    lines.append(f"           └──── repeats ────┘")
    return "\n".join(lines)


def _ascii_manager(team: TeamConfig) -> str:
    manager_name = team.workflow.options.get("manager", "?")
    workers = [m for m in team.members if m.name != manager_name]
    lines = [
        f"[bold]Workflow:[/bold] manager_driven  "
        f"(manager=@{manager_name}, max_rounds={team.workflow.max_rounds})\n",
        f"                ┌──────────────────────┐",
        f"                │ [yellow]@{manager_name:<20}[/yellow] │  ◄── decides NEXT:",
        f"                └──────────┬───────────┘",
        f"                           │",
    ]
    for i, m in enumerate(workers):
        connector = "├" if i < len(workers) - 1 else "└"
        lines.append(f"                           {connector}──► [cyan]@{m.name}[/cyan] ({m.role})")
    return "\n".join(lines)


def _ascii_review_loop(team: TeamConfig) -> str:
    producer = team.workflow.options.get("producer", "?")
    reviewer = team.workflow.options.get("reviewer", "?")
    approve_token = team.workflow.options.get("approve_token", "APPROVED")
    lines = [
        f"[bold]Workflow:[/bold] review_loop  "
        f"(max_rounds={team.workflow.max_rounds}, approve_token={approve_token!r})\n",
        f"  ┌──────────────────────┐",
        f"  │ [cyan]@{producer:<20}[/cyan] │  ◄── draft / revise",
        f"  └──────────┬───────────┘",
        f"             │ draft",
        f"             ▼",
        f"  ┌──────────────────────┐",
        f"  │ [green]@{reviewer:<20}[/green] │  ◄── review",
        f"  └──────────┬───────────┘",
        f"             │ feedback",
        f"             ▼",
        f"       [{approve_token}?]",
        f"       yes ──► done",
        f"       no  ──► back to @{producer}",
    ]
    return "\n".join(lines)


def _ascii_sequential_chain(team: TeamConfig) -> str:
    lines = [
        f"[bold]Workflow:[/bold] sequential_chain  "
        f"(max_rounds={team.workflow.max_rounds})\n",
    ]
    for i, m in enumerate(team.members):
        lines.append(f"  ┌──────────────────────┐")
        lines.append(f"  │ [cyan]@{m.name:<20}[/cyan] │")
        lines.append(f"  │ {m.role:<22} │")
        lines.append(f"  └──────────────────────┘")
        if i < len(team.members) - 1:
            lines.append(f"             │ output")
            lines.append(f"             ▼")
    lines.append(f"             │")
    lines.append(f"             └─── next round ──►  @{team.members[0].name}")
    return "\n".join(lines)


def _ascii_debate(team: TeamConfig) -> str:
    pro_name = team.workflow.options.get("pro", "?")
    con_name = team.workflow.options.get("con", "?")
    judge_name = team.workflow.options.get("judge", "?")
    topic = team.workflow.options.get("topic") or "(team goal)"
    lines = [
        f"[bold]Workflow:[/bold] debate  "
        f"(max_rounds={team.workflow.max_rounds})\n",
        f"  Topic: {topic[:60]}{'…' if len(topic) > 60 else ''}\n",
        f"  ┌──────────────────────┐     ┌──────────────────────┐",
        f"  │ [green]PRO: @{pro_name:<16}[/green] │ ◄──► │ [red]CON: @{con_name:<16}[/red] │",
        f"  └──────────────────────┘     └──────────────────────┘",
        f"               Opening statements",
        f"               ── {team.workflow.max_rounds} rebuttal round(s) ──",
        f"               Closing statements",
        f"                         │",
        f"                         ▼",
        f"               ┌──────────────────────┐",
        f"               │ [yellow]JUDGE: @{judge_name:<14}[/yellow] │",
        f"               └──────────────────────┘",
        f"                    [[TEAM_DONE]]",
    ]
    return "\n".join(lines)


def _ascii(team: TeamConfig) -> str:
    wf = team.workflow.type
    header = (
        f"[bold]Team:[/bold] {team.name}  "
        f"([dim]{len(team.members)} member(s)[/dim])\n"
    )
    dispatch = {
        "round_robin": _ascii_round_robin,
        "manager": _ascii_manager,
        "review_loop": _ascii_review_loop,
        "sequential_chain": _ascii_sequential_chain,
        "debate": _ascii_debate,
    }
    body = dispatch.get(wf, lambda t: f"(no diagram for workflow type {wf!r})")(team)
    members_section = "\n[bold]Members:[/bold]\n" + "\n".join(
        f"  • [cyan]@{m.name}[/cyan] ({m.role}) — {m.model}" for m in team.members
    )
    return header + body + members_section


# --------------------------------------------------------------------------- #
# Mermaid renderer
# --------------------------------------------------------------------------- #

def _mermaid_id(name: str) -> str:
    return name.replace("-", "_")


def _mermaid(team: TeamConfig) -> str:
    wf = team.workflow.type
    lines = ["```mermaid", "flowchart TD"]

    # Node definitions
    for m in team.members:
        mid = _mermaid_id(m.name)
        lines.append(f'    {mid}["@{m.name}<br/><i>{m.role}</i><br/><small>{m.model}</small>"]')

    lines.append("")

    if wf == "round_robin":
        members = team.members
        for i in range(len(members)):
            a = _mermaid_id(members[i].name)
            b = _mermaid_id(members[(i + 1) % len(members)].name)
            lines.append(f"    {a} --> {b}")

    elif wf == "manager":
        mgr = _mermaid_id(team.workflow.options.get("manager", ""))
        for m in team.members:
            if m.name != team.workflow.options.get("manager"):
                lines.append(f"    {mgr} -->|NEXT:| {_mermaid_id(m.name)}")
                lines.append(f"    {_mermaid_id(m.name)} --> {mgr}")

    elif wf == "review_loop":
        p = _mermaid_id(team.workflow.options.get("producer", ""))
        r = _mermaid_id(team.workflow.options.get("reviewer", ""))
        approve = team.workflow.options.get("approve_token", "APPROVED")
        lines.append(f"    {p} -->|draft| {r}")
        lines.append(f"    {r} -->|feedback| {p}")
        lines.append(f"    {r} -->|{approve}| DONE([TEAM_DONE])")

    elif wf == "sequential_chain":
        members = team.members
        for i in range(len(members)):
            a = _mermaid_id(members[i].name)
            b = _mermaid_id(members[(i + 1) % len(members)].name)
            lines.append(f"    {a} -->|output| {b}")

    elif wf == "debate":
        pro = _mermaid_id(team.workflow.options.get("pro", ""))
        con = _mermaid_id(team.workflow.options.get("con", ""))
        judge = _mermaid_id(team.workflow.options.get("judge", ""))
        lines.append(f"    {pro} <-->|rebuttals| {con}")
        lines.append(f"    {pro} -->|closing| {judge}")
        lines.append(f"    {con} -->|closing| {judge}")
        lines.append(f"    {judge} --> DONE([TEAM_DONE])")

    lines.append("```")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def render_diagram(team: TeamConfig, fmt: str = "ascii") -> str:
    """Return a workflow diagram for *team* in the requested *fmt*."""
    if fmt == "mermaid":
        return _mermaid(team)
    return _ascii(team)
