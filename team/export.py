"""Export a team run to a human-readable report.

Supports three output formats:

* ``markdown`` — a single ``.md`` file with the transcript, token usage
  summary, and artifact contents embedded as fenced code blocks.
* ``html`` — a self-contained ``.html`` file rendered from a Jinja2
  template; no external CSS or JS dependencies; includes dark-mode support.
* ``json`` — a structured JSON dump of the full run suitable for
  programmatic consumption.

Usage::

    from team.export import export_run
    text = export_run(cfg, fmt="html")
    Path("report.html").write_text(text)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from jinja2 import Environment, PackageLoader, select_autoescape

from team._version import __version__
from team.config import TeamConfig


ExportFormat = Literal["markdown", "html", "json"]


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def export_run(
    cfg: TeamConfig,
    fmt: ExportFormat = "markdown",
    *,
    no_artifacts: bool = False,
) -> str:
    """Return the full text of the export document for *cfg*'s most recent run.

    Parameters
    ----------
    cfg:
        Loaded team configuration (provides member list, models, workspace path).
    fmt:
        Output format: ``"markdown"``, ``"html"``, or ``"json"``.
    no_artifacts:
        When ``True``, workspace files under ``<workspace>/shared/`` are
        omitted from the export.  Useful for large codebases where embedding
        every file would make the report unwieldy.
    """
    turns = _load_transcript(cfg.workspace / "transcript.jsonl")
    shared_files = {} if no_artifacts else _load_shared_files(cfg.workspace / "shared")
    token_summary = _compute_token_summary(cfg, turns)
    generated_at = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    if fmt == "markdown":
        return _render_markdown(cfg, turns, shared_files, token_summary, generated_at)
    if fmt == "html":
        return _render_html(cfg, turns, shared_files, token_summary, generated_at)
    if fmt == "json":
        return _render_json(cfg, turns, shared_files, token_summary, generated_at)
    raise ValueError(f"unknown format: {fmt!r}")


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #


def _load_transcript(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    turns: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            turns.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return turns


def _load_shared_files(shared_dir: Path) -> dict[str, str]:
    """Return ``{rel_path: content}`` for every file in the shared workspace."""
    files: dict[str, str] = {}
    if not shared_dir.is_dir():
        return files
    for p in sorted(shared_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = str(p.relative_to(shared_dir))
        try:
            # errors="replace" ensures binary or non-UTF-8 files still appear
            # in the report (garbled but present) rather than raising an exception.
            files[rel] = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
    return files


# --------------------------------------------------------------------------- #
# Token + cost summary
# --------------------------------------------------------------------------- #


def _compute_token_summary(cfg: TeamConfig, turns: list[dict]) -> dict:
    """Aggregate token counts and estimated costs per member from *turns*.

    Returns a dict::

        {
            "by_member": {
                "<name>": {
                    "prompt": int,
                    "completion": int,
                    "total": int,
                    "model": str,
                    "cost": float | None,
                    "cost_fmt": str,
                }
            },
            "total_prompt": int,
            "total_completion": int,
            "total_tokens": int,
            "total_cost": float | None,
            "total_cost_fmt": str,
            "duration_seconds": float | None,
        }
    """
    from team.config import resolve_member_setting
    from team.pricing import estimate_cost, format_cost

    by_member: dict[str, dict] = {}
    first_ts = last_ts = None

    for t in turns:
        sp = t.get("speaker", "")
        if sp == "orchestrator":
            continue
        ts = t.get("timestamp")
        if ts:
            if first_ts is None or ts < first_ts:
                first_ts = ts
            if last_ts is None or ts > last_ts:
                last_ts = ts
        p = t.get("prompt_tokens", 0)
        c = t.get("completion_tokens", 0)
        if sp not in by_member:
            try:
                mc = cfg.member(sp)
                model = mc.model
                backend = resolve_member_setting(mc, cfg.defaults, "backend") or "ollama"
                is_local = backend == "ollama"
            except KeyError:
                model = "unknown"
                is_local = True  # safe default
            by_member[sp] = {
                "prompt": 0, "completion": 0, "total": 0,
                "model": model, "is_local": is_local,
            }
        by_member[sp]["prompt"] += p
        by_member[sp]["completion"] += c
        by_member[sp]["total"] += p + c

    total_prompt = total_completion = 0
    total_cost: float | None = 0.0
    for name, data in by_member.items():
        total_prompt += data["prompt"]
        total_completion += data["completion"]
        cost = estimate_cost(data["model"], data["prompt"], data["completion"],
                             is_local=data["is_local"])
        data["cost"] = cost
        data["cost_fmt"] = format_cost(cost)
        if cost is None:
            total_cost = None
        elif total_cost is not None:
            total_cost += cost

    duration = (last_ts - first_ts) if (first_ts and last_ts and last_ts > first_ts) else None

    return {
        "by_member": by_member,
        "total_prompt": total_prompt,
        "total_completion": total_completion,
        "total_tokens": total_prompt + total_completion,
        "total_cost": total_cost,
        "total_cost_fmt": format_cost(total_cost),
        "duration_seconds": duration,
    }


# --------------------------------------------------------------------------- #
# Markdown renderer
# --------------------------------------------------------------------------- #


def _fmt_timestamp(ts: float | None) -> str:
    if not ts:
        return ""
    try:
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    except (OSError, ValueError, OverflowError):
        return ""


def _render_markdown(
    cfg: TeamConfig,
    turns: list[dict],
    shared_files: dict[str, str],
    token_summary: dict,
    generated_at: str,
) -> str:
    lines: list[str] = []

    lines += [f"# Team run: {cfg.name}", ""]

    lines += ["## Goal", "", cfg.goal.strip(), ""]

    lines += ["## Members", ""]
    lines += ["| Member | Role | Model |", "| --- | --- | --- |"]
    for m in cfg.members:
        lines.append(f"| @{m.name} | {m.role} | `{m.model}` |")
    lines.append("")

    wf = cfg.workflow
    dur = token_summary.get("duration_seconds")
    dur_str = f"{dur:.1f}s" if dur is not None else "—"
    lines += [
        "## Workflow",
        "",
        f"**Type:** {wf.type}  ",
        f"**Max rounds:** {wf.max_rounds}  ",
        f"**Duration:** {dur_str}",
        "",
    ]

    # Token + cost summary table
    by_member = token_summary.get("by_member", {})
    if by_member:
        lines += ["## Token usage & estimated cost", ""]
        lines += ["| Member | Model | Prompt | Completion | Total | Est. cost |"]
        lines += ["| --- | --- | ---: | ---: | ---: | ---: |"]
        for name, d in by_member.items():
            lines.append(
                f"| @{name} | `{d['model']}` | {d['prompt']:,} | {d['completion']:,} "
                f"| {d['total']:,} | {d['cost_fmt']} |"
            )
        lines += ["| **total** | | "
                  f"{token_summary['total_prompt']:,} | "
                  f"{token_summary['total_completion']:,} | "
                  f"{token_summary['total_tokens']:,} | "
                  f"**{token_summary['total_cost_fmt']}** |", ""]

    member_turns = [t for t in turns if t.get("speaker") != "orchestrator"]
    if member_turns:
        lines += ["## Transcript", ""]
        for t in member_turns:
            ts = _fmt_timestamp(t.get("timestamp"))
            hdr = f"### Turn {t['index']} — @{t['speaker']} ({t['role']})"
            if ts:
                hdr += f"  _{ts}_"
            lines += [hdr, "", t["content"].strip(), ""]
            if t.get("files_written"):
                wrote = ", ".join(f"`{f}`" for f in t["files_written"])
                lines += [f"*Files written: {wrote}*", ""]

    if shared_files:
        lines += ["## Produced artifacts", ""]
        for rel, content in shared_files.items():
            ext = Path(rel).suffix.lstrip(".")
            lines += [f"### `{rel}`", "", f"```{ext}", content.rstrip(), "```", ""]

    lines += [
        "---",
        "",
        f"*Generated by **team-core** v{__version__} on {generated_at}*",
        "",
    ]

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# HTML renderer (Jinja2 template)
# --------------------------------------------------------------------------- #


def _render_html(
    cfg: TeamConfig,
    turns: list[dict],
    shared_files: dict[str, str],
    token_summary: dict,
    generated_at: str,
) -> str:
    env = Environment(
        loader=PackageLoader("team", "templates"),
        # autoescape is enabled for HTML so that member-generated content
        # (which may contain < > & " etc.) is safely escaped and cannot
        # inject unexpected markup into the report.
        autoescape=select_autoescape(["html", "j2"]),
    )
    env.filters["fmt_ts"] = _fmt_timestamp

    template = env.get_template("report.html.j2")
    return template.render(
        team=cfg,
        turns=turns,
        shared_files=shared_files,
        token_summary=token_summary,
        generated_at=generated_at,
        version=__version__,
        workspace=str(cfg.workspace),
    )


# --------------------------------------------------------------------------- #
# JSON renderer
# --------------------------------------------------------------------------- #


def _render_json(
    cfg: TeamConfig,
    turns: list[dict],
    shared_files: dict[str, str],
    token_summary: dict,
    generated_at: str,
) -> str:
    """Return a structured JSON dump of the run."""
    doc = {
        "format_version": 1,
        "generated_at": generated_at,
        "team_core_version": __version__,
        "team": {
            "name": cfg.name,
            "goal": cfg.goal,
            "workflow": {
                "type": cfg.workflow.type,
                "max_rounds": cfg.workflow.max_rounds,
            },
            "members": [
                {"name": m.name, "role": m.role, "model": m.model}
                for m in cfg.members
            ],
        },
        "stats": {
            "total_turns": sum(
                1 for t in turns if t.get("speaker") != "orchestrator"
            ),
            "duration_seconds": token_summary.get("duration_seconds"),
            "total_prompt_tokens": token_summary["total_prompt"],
            "total_completion_tokens": token_summary["total_completion"],
            "total_tokens": token_summary["total_tokens"],
            "estimated_cost_usd": token_summary["total_cost"],
        },
        "token_usage": {
            name: {
                "prompt_tokens": d["prompt"],
                "completion_tokens": d["completion"],
                "total_tokens": d["total"],
                "model": d["model"],
                "estimated_cost_usd": d["cost"],
            }
            for name, d in token_summary.get("by_member", {}).items()
        },
        "turns": [t for t in turns if t.get("speaker") != "orchestrator"],
        "artifacts": shared_files,
    }
    return json.dumps(doc, indent=2, ensure_ascii=False)
