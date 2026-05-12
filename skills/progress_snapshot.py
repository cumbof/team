"""progress_snapshot — write a human-readable PROGRESS.md to the workspace.

Any member can call this tool to record the current state of the run: what
has been completed, what is in progress, and what is blocked.  The file is
overwritten each time so it always reflects the most recent snapshot.

Tool
----
progress_snapshot

Body syntax
-----------
Free-form Markdown.  Structure the body however is most useful, but the
following sections are recommended for consistency:

    ## Done
    - Implemented the data model
    - Wrote the API endpoints

    ## In progress
    - Writing unit tests (alpha)

    ## Blocked
    - Need sign-off on authentication approach (@manager)

    ## Next steps
    - Complete test coverage
    - Review with QA

If the body is empty the tool returns the current snapshot (if any).

Output file
-----------
``PROGRESS.md`` in the shared workspace.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

_SNAPSHOT_FILE = "PROGRESS.md"


def execute(
    body: str,
    *,
    workspace_path: Path | None = None,
    **_: Any,
) -> str:
    if workspace_path is None:
        return "ERROR: workspace_path not available"

    snapshot_path = workspace_path / _SNAPSHOT_FILE

    # Empty body → read and return the current snapshot.
    if not body.strip():
        if snapshot_path.is_file():
            return snapshot_path.read_text(encoding="utf-8")
        return "No progress snapshot exists yet."

    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%d %H:%M UTC"
    )
    content = f"# Progress Snapshot\n\n_Last updated: {timestamp}_\n\n{body.strip()}\n"
    snapshot_path.write_text(content, encoding="utf-8")
    return f"Progress snapshot written to {_SNAPSHOT_FILE} ({len(content)} chars)."


TOOL_NAME = "progress_snapshot"
TOOL_DESCRIPTION = (
    "Write (or read) a PROGRESS.md snapshot in the shared workspace. "
    "Body: Markdown describing what is done, in-progress, and blocked. "
    "Empty body reads the current snapshot."
)
