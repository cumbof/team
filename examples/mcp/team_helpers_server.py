#!/usr/bin/env python3
"""Example stdio MCP server bundling team-process helper tools.

This is the MCP-native replacement for the old bundled ``skills/*.py`` helpers.
It exposes coordination utilities layered on top of the shared workspace:

* ``task_add`` / ``task_done`` / ``task_list`` — a shared task board (TASKS.md)
* ``search_transcript``                        — keyword search over transcript.jsonl
* ``progress_snapshot``                        — write/read PROGRESS.md
* ``request_critique`` / ``pick_critique`` / ``list_critiques`` — peer-review queue

Workspace access
----------------
An MCP server is a separate process and cannot see a member's injected context,
so it receives the shared workspace path via the ``TEAM_WORKSPACE`` environment
variable (set automatically by the orchestrator for every stdio server).  The
run transcript lives one level up at ``TEAM_WORKSPACE/../transcript.jsonl``.

Run standalone for debugging::

    TEAM_WORKSPACE=/path/to/run/shared python examples/mcp/team_helpers_server.py

Wire it into a team YAML::

    mcp_servers:
      helpers:
        transport: stdio
        command: python
        args: ["examples/mcp/team_helpers_server.py"]

    members:
      - name: coordinator
        tools: [helpers/*]
"""

from __future__ import annotations

import datetime
import json
import os
import re
import time
from pathlib import Path
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import Field

mcp = FastMCP("helpers")


def _workspace() -> Path:
    return Path(os.environ.get("TEAM_WORKSPACE", ".")).resolve()


# --------------------------------------------------------------------------- #
# Task board (TASKS.md)
# --------------------------------------------------------------------------- #

_BOARD_FILE = "TASKS.md"
_PENDING_HEADER = "## Pending"
_DONE_HEADER = "## Done"


def _read_board(ws: Path) -> str:
    p = ws / _BOARD_FILE
    if p.is_file():
        return p.read_text(encoding="utf-8")
    return f"# Team Task Board\n\n{_PENDING_HEADER}\n\n{_DONE_HEADER}\n"


def _parse_sections(content: str) -> tuple[list[str], list[str]]:
    pending: list[str] = []
    done: list[str] = []
    current: list[str] | None = None
    for line in content.splitlines():
        if line.strip() == _PENDING_HEADER:
            current = pending
        elif line.strip() == _DONE_HEADER:
            current = done
        elif current is not None and line.strip().startswith("- "):
            current.append(re.sub(r"^- \[.?\] ?", "", line.strip()))
    return pending, done


def _render_board(pending: list[str], done: list[str]) -> str:
    lines = ["# Team Task Board", "", _PENDING_HEADER]
    lines += [f"- [ ] {t}" for t in pending]
    lines += ["", _DONE_HEADER]
    lines += [f"- [x] {t}" for t in done]
    lines.append("")
    return "\n".join(lines)


@mcp.tool()
def task_add(task: Annotated[str, Field(description="The task description.")]) -> str:
    """Add a task to the shared task board (TASKS.md)."""
    ws = _workspace()
    t = task.strip()
    if not t:
        return "ERROR: provide a task description"
    pending, done = _parse_sections(_read_board(ws))
    if t in pending:
        return f"Task already exists in Pending: {t}"
    pending.append(t)
    (ws / _BOARD_FILE).write_text(_render_board(pending, done), encoding="utf-8")
    return f"Added to Pending: {t}"


@mcp.tool()
def task_done(
    match: Annotated[str, Field(description="A substring uniquely identifying the task.")],
) -> str:
    """Mark a pending task as done (matched by substring)."""
    ws = _workspace()
    needle = match.strip().lower()
    if not needle:
        return "ERROR: provide a substring to match the task"
    pending, done = _parse_sections(_read_board(ws))
    matches = [t for t in pending if needle in t.lower()]
    if not matches:
        return f"No pending task matches {needle!r}. Pending: {pending}"
    if len(matches) > 1:
        return f"Ambiguous: {len(matches)} tasks match {needle!r}: {matches}"
    task = matches[0]
    pending.remove(task)
    done.append(task)
    (ws / _BOARD_FILE).write_text(_render_board(pending, done), encoding="utf-8")
    return f"Marked done: {task}"


@mcp.tool()
def task_list() -> str:
    """List all tasks on the shared board with their status."""
    pending, done = _parse_sections(_read_board(_workspace()))
    if not (pending or done):
        return "Task board is empty."
    lines = [f"Pending ({len(pending)}):"]
    lines += [f"  [ ] {t}" for t in pending]
    lines.append(f"Done ({len(done)}):")
    lines += [f"  [x] {t}" for t in done]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# search_transcript
# --------------------------------------------------------------------------- #

_EXCERPT_LENGTH = 300


@mcp.tool()
def search_transcript(
    keyword: Annotated[str, Field(description="Search terms.")],
    speaker: Annotated[str, Field(description="Filter by speaker name (optional).")] = "",
    limit: Annotated[int, Field(description="Max results (default 5).")] = 5,
) -> str:
    """Search the run transcript for turns containing a keyword."""
    transcript_path = _workspace().parent / "transcript.jsonl"
    if not transcript_path.is_file():
        return f"ERROR: transcript not found at {transcript_path}"
    kw = keyword.strip()
    if not kw:
        return "ERROR: provide keyword: <search terms>"
    speaker_filter = speaker.strip().lower()
    pattern = re.compile(re.escape(kw), re.IGNORECASE)
    results: list[str] = []
    for raw_line in transcript_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        try:
            turn = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        spk = turn.get("speaker", "")
        content = turn.get("content", "")
        if speaker_filter and spk.lower() != speaker_filter:
            continue
        if not pattern.search(content):
            continue
        excerpt = content[:_EXCERPT_LENGTH].replace("\n", " ")
        if len(content) > _EXCERPT_LENGTH:
            excerpt += "…"
        results.append(f"[turn {turn.get('index', '?')}] @{spk}: {excerpt}")
        if len(results) >= limit:
            break
    if not results:
        desc = f"keyword={kw!r}" + (f", speaker={speaker_filter!r}" if speaker_filter else "")
        return f"No matches found for {desc}."
    return f"{len(results)} match(es) for {kw!r}:\n\n" + "\n\n".join(results)


# --------------------------------------------------------------------------- #
# progress_snapshot
# --------------------------------------------------------------------------- #

_SNAPSHOT_FILE = "PROGRESS.md"


@mcp.tool()
def progress_snapshot(
    body: Annotated[
        str, Field(description="Markdown describing done/in-progress/blocked. Empty reads current.")
    ] = "",
) -> str:
    """Write (or, with an empty body, read) a PROGRESS.md snapshot."""
    snapshot_path = _workspace() / _SNAPSHOT_FILE
    if not body.strip():
        if snapshot_path.is_file():
            return snapshot_path.read_text(encoding="utf-8")
        return "No progress snapshot exists yet."
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    content = f"# Progress Snapshot\n\n_Last updated: {ts}_\n\n{body.strip()}\n"
    snapshot_path.write_text(content, encoding="utf-8")
    return f"Progress snapshot written to {_SNAPSHOT_FILE} ({len(content)} chars)."


# --------------------------------------------------------------------------- #
# Critique queue
# --------------------------------------------------------------------------- #

_QUEUE_FILE = "CRITIQUE_QUEUE.md"
_SEP = "---"


def _read_queue(ws: Path) -> list[dict]:
    p = ws / _QUEUE_FILE
    if not p.is_file():
        return []
    entries: list[dict] = []
    current: dict = {}
    _fields = {
        "**ID:**": "id", "**From:**": "from", "**File:**": "file",
        "**Question:**": "question", "**Status:**": "status",
    }
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip() == _SEP:
            if current:
                entries.append(current)
                current = {}
            continue
        for marker, key in _fields.items():
            if line.startswith(marker):
                current[key] = line.split(marker)[1].strip()
                break
    if current:
        entries.append(current)
    return entries


def _write_queue(ws: Path, entries: list[dict]) -> None:
    lines = ["# Critique Queue", ""]
    for e in entries:
        lines += [
            f"**ID:** {e.get('id', '?')}",
            f"**From:** {e.get('from', '?')}",
            f"**File:** {e.get('file', '?')}",
            f"**Question:** {e.get('question', '?')}",
            f"**Status:** {e.get('status', 'pending')}",
            _SEP,
            "",
        ]
    (ws / _QUEUE_FILE).write_text("\n".join(lines), encoding="utf-8")


@mcp.tool()
def request_critique(
    question: Annotated[str, Field(description="What you want reviewed.")],
    from_member: Annotated[str, Field(description="Your member name.")] = "unknown",
    file: Annotated[str, Field(description="File under review, or 'general'.")] = "general",
) -> str:
    """Post a structured peer-review request to the shared queue."""
    ws = _workspace()
    q = question.strip()
    if not q:
        return "ERROR: provide question: <what you want reviewed>"
    entries = _read_queue(ws)
    req_id = f"CR-{int(time.time())}-{len(entries) + 1}"
    entries.append(
        {"id": req_id, "from": from_member, "file": file, "question": q, "status": "pending"}
    )
    _write_queue(ws, entries)
    return f"Critique request {req_id} posted.\n  From: {from_member}\n  File: {file}\n  Question: {q}"


@mcp.tool()
def pick_critique() -> str:
    """Claim the oldest pending critique request and return it."""
    ws = _workspace()
    entries = _read_queue(ws)
    pending = [e for e in entries if e.get("status") == "pending"]
    if not pending:
        return "No pending critique requests."
    picked = pending[0]
    picked["status"] = "claimed"
    _write_queue(ws, entries)
    return (
        f"Claimed critique request {picked['id']}.\n"
        f"  From: {picked['from']}\n  File: {picked['file']}\n"
        f"  Question: {picked['question']}"
    )


@mcp.tool()
def list_critiques() -> str:
    """List all pending critique requests without claiming any."""
    entries = _read_queue(_workspace())
    pending = [e for e in entries if e.get("status") == "pending"]
    if not pending:
        return "No pending critique requests."
    lines = [f"{len(pending)} pending critique request(s):"]
    for e in pending:
        lines.append(f"  [{e['id']}] from @{e['from']} — {e['file']}: {e['question']}")
    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run()
