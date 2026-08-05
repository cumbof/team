"""``memory`` server — remember, recall, forget, list_memories."""

from __future__ import annotations

import logging
from typing import Annotated

from pydantic import Field

from team.mcp.builtin import BuiltinContext
from team.mcp.builtin._util import offload, truncate

log = logging.getLogger(__name__)

_DISABLED = "ERROR: memory is not enabled for this member (set memory.enabled: true)"


def build(ctx: BuiltinContext) -> "object":
    from mcp.server.fastmcp import FastMCP

    srv = FastMCP("memory")
    memory = ctx.memory

    @srv.tool()
    @offload
    def remember(
        key: Annotated[str, Field(description="Unique memory key.")],
        value: Annotated[str, Field(description="Value to store (multi-line text allowed).")],
        tags: Annotated[str, Field(description="Comma-separated tags (optional).")] = "",
        importance: Annotated[float, Field(description="Importance score 0–1 (default 1.0).")] = 1.0,
    ) -> str:
        """Store a named memory in your persistent cross-session memory store."""
        if memory is None:
            return _DISABLED
        if not key.strip():
            return "ERROR: provide key: <memory key>"
        tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
        if not value.strip():
            return "ERROR: memory value must not be empty"
        return memory.remember(key.strip(), value.strip(), tags=tag_list, importance=importance)

    @srv.tool()
    @offload
    def recall(
        query: Annotated[str, Field(description="Keywords to search for.")],
        limit: Annotated[int, Field(description="Maximum results to return (default 5).")] = 5,
    ) -> str:
        """Search your persistent memory by keyword."""
        if memory is None:
            return _DISABLED
        q = query.strip()
        if not q:
            return "ERROR: provide query: <search terms>"
        results = memory.recall(q, limit=limit)
        if not results:
            return f"No memories found matching {q!r}."
        lines = [f"Found {len(results)} memory(ies) matching {q!r}:"]
        for m in results:
            tags_part = f" [{m['tags']}]" if m["tags"] else ""
            lines.append(f"- **{m['key']}** (imp: {m['importance']:.1f}){tags_part}: {m['value']}")
        return truncate("\n".join(lines))

    @srv.tool()
    @offload
    def forget(
        key: Annotated[str, Field(description="Key of the memory to delete.")],
    ) -> str:
        """Delete a memory from your persistent store by key."""
        if memory is None:
            return _DISABLED
        k = key.strip()
        if not k:
            return "ERROR: provide key: <memory key>"
        deleted = memory.forget(k)
        return f"Deleted memory: {k!r}" if deleted else f"No memory found with key: {k!r}"

    @srv.tool()
    @offload
    def list_memories(
        tag: Annotated[str, Field(description="Filter by tag (optional).")] = "",
        limit: Annotated[int, Field(description="Maximum entries to return (default 20).")] = 20,
    ) -> str:
        """List memories in your persistent store, optionally filtered by tag."""
        if memory is None:
            return _DISABLED
        tag_val = tag.strip() or None
        entries = memory.list_memories(tag=tag_val, limit=limit)
        if not entries:
            return (
                f"No memories found with tag {tag_val!r}." if tag_val else "No memories stored yet."
            )
        lines = [f"{len(entries)} memory(ies)" + (f" tagged {tag_val!r}" if tag_val else "") + ":"]
        for m in entries:
            tags_part = f" [{m['tags']}]" if m["tags"] else ""
            lines.append(
                f"- [{m['id']}] **{m['key']}** (imp: {m['importance']:.1f}){tags_part}: {m['value']}"
            )
        return truncate("\n".join(lines))

    return srv
