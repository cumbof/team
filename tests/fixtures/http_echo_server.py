#!/usr/bin/env python3
"""A tiny streamable-HTTP FastMCP server used by test_mcp_external_http.

Usage: http_echo_server.py <port>   (serves MCP at http://127.0.0.1:<port>/mcp)
"""

from __future__ import annotations

import sys

from mcp.server.fastmcp import FastMCP


def build(port: int) -> FastMCP:
    mcp = FastMCP("echo", host="127.0.0.1", port=port)

    @mcp.tool()
    def echo(text: str) -> str:
        """Return the text unchanged."""
        return text

    @mcp.tool()
    def shout(text: str) -> str:
        """Return the text uppercased."""
        return text.upper()

    return mcp


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    build(port).run(transport="streamable-http")
