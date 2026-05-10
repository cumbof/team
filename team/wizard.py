"""Interactive wizard for ``team new`` (F8).

Guides the user through creating a new team YAML via a series of prompts,
then validates the result before writing it to disk.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import click
import yaml


# Popular Ollama models suggested in the wizard — easy to extend.
_SUGGESTED_MODELS = [
    "llama3.1:8b",
    "llama3.1:70b",
    "llama3.2:3b",
    "qwen2.5-coder:7b",
    "qwen2.5:14b",
    "gemma3:12b",
    "mistral:7b",
    "phi4:14b",
    "deepseek-r1:8b",
]

_WORKFLOW_DESCRIPTIONS = {
    "round_robin":      "Every member speaks in order, repeated max_rounds times.",
    "manager":          "A designated manager decides who speaks next each turn.",
    "review_loop":      "A producer/reviewer pair iterate until APPROVED or max_rounds.",
    "sequential_chain": "Each member's output becomes the next member's input.",
    "debate":           "Two members argue opposing sides; a judge delivers a verdict.",
}


def _prompt_multiline(label: str, hint: str = "") -> str:
    """Prompt for text that may be long; show hint and accept a single line."""
    if hint:
        click.echo(f"  [dim]{hint}[/dim]" if False else f"  ({hint})")
    return click.prompt(label).strip()


def run_wizard(output_path: Path) -> None:
    """Run the interactive team-creation wizard and write YAML to *output_path*."""
    click.echo("\n🧙  team new — interactive wizard\n")
    click.echo("Answer each prompt to build your team YAML.\n")

    # ── Team basics ──────────────────────────────────────────────────────── #
    name = click.prompt(
        "Team name (lowercase letters, digits, hyphens, underscores)"
    ).strip().lower().replace(" ", "-")

    goal = click.prompt(
        "Team goal (one sentence; you can edit the file afterwards for longer text)"
    ).strip()

    workspace = click.prompt(
        "Workspace directory",
        default=f"./runs/{name}",
    ).strip()

    # ── Members ──────────────────────────────────────────────────────────── #
    num_members = click.prompt(
        "How many members?",
        default=2,
        type=click.IntRange(1, 10),
    )

    members: list[dict] = []
    click.echo(f"\nSuggested models: {', '.join(_SUGGESTED_MODELS[:5])}")
    click.echo("(you can type any model available in your Ollama instance)\n")

    for i in range(num_members):
        click.echo(f"── Member {i + 1} of {num_members} ──────────────")
        m_name = click.prompt(f"  Name").strip().lower().replace(" ", "_")
        m_role = click.prompt(f"  Role (e.g. 'Software Engineer')").strip()
        m_model = click.prompt(
            f"  Model",
            default=_SUGGESTED_MODELS[i % len(_SUGGESTED_MODELS)],
        ).strip()
        m_persona = click.prompt(
            f"  Persona (brief description of their expertise and style)"
        ).strip()
        members.append({
            "name": m_name,
            "role": m_role,
            "model": m_model,
            "persona": m_persona,
        })
        click.echo()

    # ── Workflow ─────────────────────────────────────────────────────────── #
    click.echo("── Workflow ──────────────────────────────────────────")
    for wf_type, desc in _WORKFLOW_DESCRIPTIONS.items():
        click.echo(f"  {wf_type:<20} {desc}")
    click.echo()

    wf_type = click.prompt(
        "Workflow type",
        type=click.Choice(list(_WORKFLOW_DESCRIPTIONS)),
        default="round_robin",
    )
    max_rounds = click.prompt("Max rounds", default=4, type=click.IntRange(1, 100))

    workflow: dict = {"type": wf_type, "max_rounds": max_rounds}
    member_names = [m["name"] for m in members]

    if wf_type == "manager":
        mgr = click.prompt(
            "Manager member name",
            type=click.Choice(member_names),
        )
        workflow["manager"] = mgr

    elif wf_type == "review_loop":
        producer = click.prompt("Producer member name", type=click.Choice(member_names))
        reviewer = click.prompt(
            "Reviewer member name",
            type=click.Choice([n for n in member_names if n != producer]),
        )
        workflow["producer"] = producer
        workflow["reviewer"] = reviewer

    elif wf_type == "debate":
        pro = click.prompt("PRO member name (argues in favour)", type=click.Choice(member_names))
        remaining = [n for n in member_names if n != pro]
        con = click.prompt("CON member name (argues against)", type=click.Choice(remaining))
        remaining2 = [n for n in remaining if n != con]
        if not remaining2:
            click.echo(
                "⚠  A debate needs at least 3 members (pro, con, judge). "
                "Judge will default to the last member."
            )
            judge = member_names[-1]
        else:
            judge = click.prompt("JUDGE member name", type=click.Choice(remaining2))
        workflow["pro"] = pro
        workflow["con"] = con
        workflow["judge"] = judge

    # ── Defaults ─────────────────────────────────────────────────────────── #
    click.echo("\n── Defaults ──────────────────────────────────────────")
    gpus = click.prompt(
        "GPU access",
        type=click.Choice(["none", "all"]),
        default="none",
    )
    context_window = click.prompt(
        "Context window size (tokens)",
        default=8192,
        type=click.IntRange(1024, 131072),
    )
    temperature = click.prompt(
        "Temperature (0.0 = deterministic, 1.0 = creative)",
        default=0.4,
        type=click.FloatRange(0.0, 2.0),
    )

    # ── Assemble YAML ────────────────────────────────────────────────────── #
    doc: dict = {
        "name": name,
        "goal": goal,
        "workspace": workspace,
        "workflow": workflow,
        "defaults": {
            "context_window": context_window,
            "temperature": temperature,
            "gpus": gpus,
        },
        "members": members,
    }

    # Validate before writing
    try:
        from team.config import load_team
        import tempfile, os
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as tmp:
            yaml.dump(doc, tmp, allow_unicode=True, default_flow_style=False, sort_keys=False)
            tmp_path = tmp.name
        load_team(tmp_path)
        os.unlink(tmp_path)
    except Exception as exc:
        click.echo(f"\n⚠  Validation warning: {exc}")
        click.echo("   The file will still be written — fix it manually.\n")

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        yaml.dump(doc, fh, allow_unicode=True, default_flow_style=False, sort_keys=False)

    click.echo(f"\n✅  Written to {output_path}")
    click.echo(f"    Next: team validate {output_path}")
    click.echo(f"          team run {output_path}")
