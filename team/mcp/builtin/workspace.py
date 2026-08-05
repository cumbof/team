"""``workspace`` server — read_file, write_file, append_file, list_files."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

from pydantic import Field

from team.mcp.builtin import BuiltinContext
from team.mcp.builtin._util import offload, truncate

log = logging.getLogger(__name__)


def _resolve_in_workspace(workspace_path: Path, rel: str) -> tuple[Path | None, str | None]:
    """Resolve *rel* under the workspace with a traversal guard.

    Returns ``(target, None)`` or ``(None, error_message)``.
    """
    try:
        target = (workspace_path / rel).resolve()
        target.relative_to(workspace_path.resolve())
    except ValueError:
        return None, f"ERROR: path {rel!r} escapes the workspace"
    return target, None


def build(ctx: BuiltinContext) -> "object":
    from mcp.server.fastmcp import FastMCP

    srv = FastMCP("workspace")
    workspace_path = ctx.workspace_path

    @srv.tool()
    @offload
    def read_file(
        path: Annotated[str, Field(description="Relative path inside the shared workspace.")],
    ) -> str:
        """Read a file from the shared workspace by relative path and return its contents."""
        rel = path.strip()
        if not rel:
            return "ERROR: provide path: <relative path>"
        if workspace_path is None:
            return "ERROR: no workspace available"
        target, err = _resolve_in_workspace(workspace_path, rel)
        if err:
            return err
        if not target.is_file():
            return f"ERROR: {rel!r} not found in shared workspace"
        try:
            return truncate(target.read_text(encoding="utf-8", errors="replace"))
        except Exception as exc:  # noqa: BLE001
            return f"ERROR reading file: {exc}"

    @srv.tool()
    @offload
    def write_file(
        path: Annotated[str, Field(description="Relative path inside the shared workspace.")],
        content: Annotated[str, Field(description="File content to write.")],
    ) -> str:
        """Write (create or overwrite) a file in the shared workspace."""
        if workspace_path is None:
            return "ERROR: no workspace available"
        rel = path.strip()
        if not rel:
            return "ERROR: provide a relative path"
        target, err = _resolve_in_workspace(workspace_path, rel)
        if err:
            return err
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return f"wrote {len(content)} chars to {rel}"
        except Exception as exc:  # noqa: BLE001
            return f"ERROR writing file: {exc}"

    @srv.tool()
    @offload
    def append_file(
        path: Annotated[str, Field(description="Relative path inside the shared workspace.")],
        content: Annotated[str, Field(description="Text to append.")],
    ) -> str:
        """Append text to a file in the shared workspace (creates the file if it does not exist)."""
        if workspace_path is None:
            return "ERROR: no workspace available"
        rel = path.strip()
        if not rel:
            return "ERROR: provide a relative path"
        target, err = _resolve_in_workspace(workspace_path, rel)
        if err:
            return err
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("a", encoding="utf-8") as fh:
                fh.write(content)
            return f"appended {len(content)} chars to {rel}"
        except Exception as exc:  # noqa: BLE001
            return f"ERROR appending to file: {exc}"

    @srv.tool()
    @offload
    def list_files(
        pattern: Annotated[
            str, Field(description="Glob pattern, e.g. **/*.py. Omit to list all.")
        ] = "",
    ) -> str:
        """List files in the shared workspace, optionally filtered by a glob pattern."""
        if workspace_path is None:
            return "ERROR: no workspace available"
        pat = (pattern or "").strip()
        if not workspace_path.is_dir():
            return "(workspace is empty)"
        try:
            all_files = sorted(
                str(p.relative_to(workspace_path))
                for p in workspace_path.rglob("*")
                if p.is_file()
            )
        except Exception as exc:  # noqa: BLE001
            return f"ERROR listing workspace: {exc}"
        if not all_files:
            return "(workspace is empty)"
        if pat:
            matched = [f for f in all_files if Path(f).match(pat)]
            if not matched:
                return f"(no files match pattern {pat!r})"
        else:
            matched = all_files
        return truncate("\n".join(matched))

    return srv
