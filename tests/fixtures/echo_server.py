#!/usr/bin/env python3
"""A tiny stdio FastMCP server used by test_mcp_external_stdio."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("echo")


@mcp.tool()
def echo(text: str) -> str:
    """Return the text unchanged."""
    return text


@mcp.tool()
def shout(text: str) -> str:
    """Return the text uppercased."""
    return text.upper()


if __name__ == "__main__":
    mcp.run()
