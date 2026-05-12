"""task_board — shared task tracker written to TASKS.md in the workspace.

Tools
-----
task_add    — add a task to the board.
task_done   — mark a task as done (matched by substring).
task_list   — list all tasks with their current status.

Body syntax
-----------
task_add:
    <task description>

task_done:
    <substring that uniquely identifies the task>

task_list:
    (body is ignored)

TASKS.md format
---------------
The board is stored as a Markdown file so it is human-readable and can be
written to the shared workspace like any other artifact.

    # Team Task Board

    ## Pending
    - [ ] Design the database schema
    - [ ] Write unit tests

    ## Done
    - [x] Define project structure
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

_BOARD_FILE = "TASKS.md"
_PENDING_HEADER = "## Pending"
_DONE_HEADER = "## Done"


def _read_board(workspace_path: Path) -> str:
    p = workspace_path / _BOARD_FILE
    if p.is_file():
        return p.read_text(encoding="utf-8")
    return f"# Team Task Board\n\n{_PENDING_HEADER}\n\n{_DONE_HEADER}\n"


def _write_board(workspace_path: Path, content: str) -> None:
    (workspace_path / _BOARD_FILE).write_text(content, encoding="utf-8")


def _parse_sections(content: str) -> tuple[list[str], list[str]]:
    """Return (pending_tasks, done_tasks) as plain-text lists."""
    pending: list[str] = []
    done: list[str] = []
    current: list[str] | None = None
    for line in content.splitlines():
        if line.strip() == _PENDING_HEADER:
            current = pending
        elif line.strip() == _DONE_HEADER:
            current = done
        elif current is not None and line.strip().startswith("- "):
            task = re.sub(r"^- \[.?\] ?", "", line.strip())
            current.append(task)
    return pending, done


def _render_board(pending: list[str], done: list[str]) -> str:
    lines = ["# Team Task Board", "", _PENDING_HEADER]
    for t in pending:
        lines.append(f"- [ ] {t}")
    lines.append("")
    lines.append(_DONE_HEADER)
    for t in done:
        lines.append(f"- [x] {t}")
    lines.append("")
    return "\n".join(lines)


def _task_add(body: str, *, workspace_path: Path | None = None, **_: Any) -> str:
    if workspace_path is None:
        return "ERROR: workspace_path not available"
    task = body.strip()
    if not task:
        return "ERROR: provide a task description in the body"
    content = _read_board(workspace_path)
    pending, done = _parse_sections(content)
    if task in pending:
        return f"Task already exists in Pending: {task}"
    pending.append(task)
    _write_board(workspace_path, _render_board(pending, done))
    return f"Added to Pending: {task}"


def _task_done(body: str, *, workspace_path: Path | None = None, **_: Any) -> str:
    if workspace_path is None:
        return "ERROR: workspace_path not available"
    needle = body.strip().lower()
    if not needle:
        return "ERROR: provide a substring to match the task"
    content = _read_board(workspace_path)
    pending, done = _parse_sections(content)
    matches = [t for t in pending if needle in t.lower()]
    if not matches:
        return f"No pending task matches {needle!r}. Pending: {pending}"
    if len(matches) > 1:
        return f"Ambiguous: {len(matches)} tasks match {needle!r}: {matches}"
    task = matches[0]
    pending.remove(task)
    done.append(task)
    _write_board(workspace_path, _render_board(pending, done))
    return f"Marked done: {task}"


def _task_list(body: str, *, workspace_path: Path | None = None, **_: Any) -> str:
    if workspace_path is None:
        return "ERROR: workspace_path not available"
    content = _read_board(workspace_path)
    pending, done = _parse_sections(content)
    lines = []
    lines.append(f"Pending ({len(pending)}):")
    for t in pending:
        lines.append(f"  [ ] {t}")
    lines.append(f"Done ({len(done)}):")
    for t in done:
        lines.append(f"  [x] {t}")
    return "\n".join(lines) if (pending or done) else "Task board is empty."


TOOLS = {
    "task_add": _task_add,
    "task_done": _task_done,
    "task_list": _task_list,
}

TOOL_DESCRIPTIONS = {
    "task_add": (
        "Add a task to the shared task board (TASKS.md). "
        "Body: the task description."
    ),
    "task_done": (
        "Mark a pending task as done. "
        "Body: a substring that uniquely identifies the task."
    ),
    "task_list": (
        "List all tasks on the shared board with their status (pending / done)."
    ),
}
