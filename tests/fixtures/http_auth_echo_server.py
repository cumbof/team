#!/usr/bin/env python3
"""A bearer-auth-gated streamable-HTTP MCP server (behaves like a remote one).

Every HTTP request must carry `Authorization: Bearer <AUTH_TOKEN>` or receives
401.  Uses a RAW ASGI guard rather than BaseHTTPMiddleware (which buffers and
breaks SSE), so authorized requests stream normally and `lifespan` passes
through untouched.

Usage: AUTH_TOKEN=secret http_auth_echo_server.py <port>
"""

from __future__ import annotations

import os
import sys

import uvicorn
from mcp.server.fastmcp import FastMCP

TOKEN = os.environ.get("AUTH_TOKEN", "")


class RequireBearer:
    """Raw-ASGI wrapper: 401 unless Authorization matches; forwards the rest."""

    def __init__(self, app, token: str) -> None:
        self.app = app
        self.expected = f"Bearer {token}"

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            headers = {k.lower(): v for k, v in scope.get("headers", [])}
            if headers.get(b"authorization", b"").decode() != self.expected:
                await send({
                    "type": "http.response.start",
                    "status": 401,
                    "headers": [(b"content-type", b"application/json")],
                })
                await send({"type": "http.response.body", "body": b'{"error":"unauthorized"}'})
                return
        await self.app(scope, receive, send)


if __name__ == "__main__":
    port = int(sys.argv[1])
    mcp = FastMCP("secure", host="127.0.0.1", port=port)

    @mcp.tool()
    def echo(text: str) -> str:
        """Return the text unchanged."""
        return text

    app = RequireBearer(mcp.streamable_http_app(), TOKEN)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
