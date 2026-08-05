"""Shared helpers for the built-in MCP servers."""

from __future__ import annotations

import functools

import anyio


def offload(fn):
    """Decorate a blocking sync tool body to run in a worker thread.

    Built-in tools do blocking I/O (subprocess, disk, network — federation waits
    up to 600s on a remote team). FastMCP invokes sync tool functions *inline on
    the event loop*, so without this a slow tool would freeze the single bus loop
    and stall every other member's concurrent tool calls. Offloading to a thread
    keeps the loop responsive.

    ``functools.wraps`` preserves the wrapped function's signature and docstring,
    so FastMCP still generates the correct input schema from the typed params.
    """

    @functools.wraps(fn)
    async def wrapper(**kwargs):
        return await anyio.to_thread.run_sync(functools.partial(fn, **kwargs))

    return wrapper


#: Maximum characters returned by any single tool (mirrors the old tools._MAX_OUTPUT).
_MAX_OUTPUT = 8192
#: Tighter cap for search results.
_MAX_SEARCH_OUTPUT = 4096


def truncate(text: str, limit: int = _MAX_OUTPUT) -> str:
    """Truncate *text* to *limit* chars with a visible marker."""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n[… truncated at {limit} chars]"
