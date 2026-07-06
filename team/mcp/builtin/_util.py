"""Shared helpers for the built-in MCP servers."""

from __future__ import annotations

#: Maximum characters returned by any single tool (mirrors the old tools._MAX_OUTPUT).
_MAX_OUTPUT = 8192
#: Tighter cap for search results.
_MAX_SEARCH_OUTPUT = 4096


def truncate(text: str, limit: int = _MAX_OUTPUT) -> str:
    """Truncate *text* to *limit* chars with a visible marker."""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n[… truncated at {limit} chars]"
