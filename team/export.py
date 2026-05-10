"""Export a team run to a human-readable report.

Supports two output formats:

* ``markdown`` — a single ``.md`` file with the transcript and artifact
  contents embedded as fenced code blocks.
* ``html`` — a self-contained ``.html`` file rendered from a Jinja2
  template; no external CSS or JS dependencies.

Usage::

    from team.export import export_run
    text = export_run(cfg, fmt="html")
    Path("report.html").write_text(text)
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Literal

from jinja2 import Environment, PackageLoader, select_autoescape

from team.config import TeamConfig


ExportFormat = Literal["markdown", "html"]


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def export_run(cfg: TeamConfig, fmt: ExportFormat = "markdown") -> str:
    """Return the full text of the export document for *cfg*'s most recent run.

    Reads ``<workspace>/transcript.jsonl`` and everything under
    ``<workspace>/shared/``.
    """
    turns = _load_transcript(cfg.workspace / "transcript.jsonl")
    shared_files = _load_shared_files(cfg.workspace / "shared")
    if fmt == "markdown":
        return _render_markdown(cfg, turns, shared_files)
    if fmt == "html":
        return _render_html(cfg, turns, shared_files)
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
            files[rel] = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
    return files


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


def _render_markdown(cfg: TeamConfig, turns: list[dict], shared_files: dict[str, str]) -> str:
    lines: list[str] = []

    lines += [f"# Team run: {cfg.name}", ""]

    lines += ["## Goal", "", cfg.goal.strip(), ""]

    lines += ["## Members", ""]
    lines += ["| Member | Role | Model |", "| --- | --- | --- |"]
    for m in cfg.members:
        lines.append(f"| @{m.name} | {m.role} | `{m.model}` |")
    lines.append("")

    wf = cfg.workflow
    lines += [
        "## Workflow",
        "",
        f"**Type:** {wf.type}  ",
        f"**Max rounds:** {wf.max_rounds}",
        "",
    ]

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

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# HTML renderer (Jinja2 template)
# --------------------------------------------------------------------------- #


def _render_html(cfg: TeamConfig, turns: list[dict], shared_files: dict[str, str]) -> str:
    env = Environment(
        loader=PackageLoader("team", "templates"),
        autoescape=select_autoescape(["html", "j2"]),
    )
    env.filters["fmt_ts"] = _fmt_timestamp

    template = env.get_template("report.html.j2")
    return template.render(
        team=cfg,
        turns=turns,
        shared_files=shared_files,
        workspace=str(cfg.workspace),
    )
