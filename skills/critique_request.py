"""critique_request — structured asynchronous peer-review queue.

Decouples "I need a review" from the workflow's turn ordering.  Any member
can post a critique request to the queue; any other member can pick one up
on its next turn and address it.

Tools
-----
request_critique    — post a review request to the queue.
pick_critique       — read and remove the next pending request from the queue.
list_critiques      — list all pending critique requests without claiming one.

Body syntax
-----------
request_critique:
    from: <your member name>
    file: <relative path in workspace, or "general">
    question: <what you want reviewed or answered>

pick_critique:
    (body is ignored — always picks the oldest pending request)

list_critiques:
    (body is ignored)

Queue file
----------
Requests are stored in ``critique_queue.md`` in the shared workspace so
they survive context resets and are visible to human observers.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

_QUEUE_FILE = "critique_queue.md"
_SEP = "---"


def _parse_kv(body: str, key: str) -> str | None:
    for line in body.splitlines():
        line = line.strip()
        if line.lower().startswith(f"{key}:"):
            return line[len(key) + 1:].strip()
    return None


def _read_queue(workspace_path: Path) -> list[dict]:
    p = workspace_path / _QUEUE_FILE
    if not p.is_file():
        return []
    entries: list[dict] = []
    current: dict = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip() == _SEP:
            if current:
                entries.append(current)
                current = {}
        elif line.startswith("**ID:**"):
            current["id"] = line.split("**ID:**")[1].strip()
        elif line.startswith("**From:**"):
            current["from"] = line.split("**From:**")[1].strip()
        elif line.startswith("**File:**"):
            current["file"] = line.split("**File:**")[1].strip()
        elif line.startswith("**Question:**"):
            current["question"] = line.split("**Question:**")[1].strip()
        elif line.startswith("**Status:**"):
            current["status"] = line.split("**Status:**")[1].strip()
    if current:
        entries.append(current)
    return entries


def _write_queue(workspace_path: Path, entries: list[dict]) -> None:
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
    (workspace_path / _QUEUE_FILE).write_text("\n".join(lines), encoding="utf-8")


def _request_critique(
    body: str, *, workspace_path: Path | None = None, **_: Any
) -> str:
    if workspace_path is None:
        return "ERROR: workspace_path not available"

    from_member = _parse_kv(body, "from") or "unknown"
    file_ref = _parse_kv(body, "file") or "general"
    question = _parse_kv(body, "question")
    if not question:
        return "ERROR: provide 'question: <what you want reviewed>'"

    entries = _read_queue(workspace_path)
    req_id = f"CR-{int(time.time())}-{len(entries) + 1}"
    entries.append({
        "id": req_id,
        "from": from_member,
        "file": file_ref,
        "question": question,
        "status": "pending",
    })
    _write_queue(workspace_path, entries)
    return (
        f"Critique request {req_id} posted.\n"
        f"  From: {from_member}\n"
        f"  File: {file_ref}\n"
        f"  Question: {question}"
    )


def _pick_critique(
    body: str, *, workspace_path: Path | None = None, **_: Any
) -> str:
    if workspace_path is None:
        return "ERROR: workspace_path not available"

    entries = _read_queue(workspace_path)
    pending = [e for e in entries if e.get("status") == "pending"]
    if not pending:
        return "No pending critique requests."

    picked = pending[0]
    picked["status"] = "claimed"
    _write_queue(workspace_path, entries)
    return (
        f"Claimed critique request {picked['id']}.\n"
        f"  From: {picked['from']}\n"
        f"  File: {picked['file']}\n"
        f"  Question: {picked['question']}\n\n"
        "Address this in your reply, then call task_done or log_decision as appropriate."
    )


def _list_critiques(
    body: str, *, workspace_path: Path | None = None, **_: Any
) -> str:
    if workspace_path is None:
        return "ERROR: workspace_path not available"

    entries = _read_queue(workspace_path)
    pending = [e for e in entries if e.get("status") == "pending"]
    if not pending:
        return "No pending critique requests."
    lines = [f"{len(pending)} pending critique request(s):"]
    for e in pending:
        lines.append(
            f"  [{e['id']}] from @{e['from']} — {e['file']}: {e['question']}"
        )
    return "\n".join(lines)


TOOLS = {
    "request_critique": _request_critique,
    "pick_critique": _pick_critique,
    "list_critiques": _list_critiques,
}

TOOL_DESCRIPTIONS = {
    "request_critique": (
        "Post a structured peer-review request to the shared queue. "
        "Body: 'from: <name>\\nfile: <path or general>\\nquestion: <what to review>'."
    ),
    "pick_critique": (
        "Claim the oldest pending critique request from the queue and return it. "
        "Address it in your reply."
    ),
    "list_critiques": (
        "List all pending critique requests without claiming any."
    ),
}
