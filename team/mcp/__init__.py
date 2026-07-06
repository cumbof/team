"""MCP-native tool layer for team.

All tool calls — built-in and external — flow through a single
:class:`~team.mcp.bus.MCPToolBus`.  Built-in tools are exposed as in-process
FastMCP servers (see :mod:`team.mcp.builtin`); external tools are ordinary MCP
servers connected over stdio or Streamable HTTP.  Each :class:`Member` sees the
bus through a :class:`~team.mcp.bus.MemberToolset` that filters to the tools it
has enabled and applies output truncation.

The ``mcp`` SDK is only imported from within this package so the rest of the
codebase stays SDK-agnostic.
"""

from __future__ import annotations

from team.mcp.bus import MCPToolBus, MemberToolset, ToolInfo

__all__ = ["MCPToolBus", "MemberToolset", "ToolInfo"]
