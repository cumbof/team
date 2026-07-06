"""``decisions`` server — log_decision, read_decisions."""

from __future__ import annotations

import datetime
import logging
from typing import Annotated

from pydantic import Field

from team.mcp.builtin import BuiltinContext
from team.mcp.builtin._util import offload, truncate

log = logging.getLogger(__name__)

_DECISIONS_FILE = "decisions.md"


def build(ctx: BuiltinContext) -> "object":
    from mcp.server.fastmcp import FastMCP

    srv = FastMCP("decisions")
    workspace_path = ctx.workspace_path
    member_name = ctx.member_name

    @srv.tool()
    @offload
    def log_decision(
        title: Annotated[str, Field(description="Short decision title.")],
        rationale: Annotated[str, Field(description="Why this decision was made (optional).")] = "",
        alternatives: Annotated[str, Field(description="Alternatives considered (optional).")] = "",
    ) -> str:
        """Append a timestamped decision entry to decisions.md in the shared workspace."""
        if workspace_path is None:
            return "ERROR: no workspace available"
        t = title.strip()
        if not t:
            return "ERROR: provide title: <short decision title>"
        now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        lines = [f"## Decision: {t}", f"**Date:** {now}  ", f"**By:** @{member_name}  "]
        if rationale:
            lines += ["", f"**Rationale:** {rationale}"]
        if alternatives:
            lines += ["", f"**Alternatives considered:** {alternatives}"]
        lines += ["", "---", ""]
        entry = "\n".join(lines)
        target = (workspace_path / _DECISIONS_FILE).resolve()
        try:
            target.relative_to(workspace_path.resolve())
        except ValueError:
            return "ERROR: path traversal guard triggered for decisions.md"
        try:
            with target.open("a", encoding="utf-8") as fh:
                fh.write(entry)
            return f"Decision logged: {t!r} (appended to {_DECISIONS_FILE})"
        except Exception as exc:  # noqa: BLE001
            return f"ERROR writing decisions.md: {exc}"

    @srv.tool()
    @offload
    def read_decisions() -> str:
        """Read the full decision log (decisions.md) from the shared workspace."""
        if workspace_path is None:
            return "ERROR: no workspace available"
        target = workspace_path / _DECISIONS_FILE
        if not target.is_file():
            return "No decisions have been logged yet (decisions.md does not exist)."
        try:
            return truncate(target.read_text(encoding="utf-8", errors="replace"))
        except Exception as exc:  # noqa: BLE001
            return f"ERROR reading decisions.md: {exc}"

    return srv
