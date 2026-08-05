"""Shared fixtures for MCP-native tests."""

from __future__ import annotations

import pytest

from team.mcp.bus import MCPToolBus, MemberToolset
from team.mcp.builtin import BUILTIN_SERVERS, BuiltinContext


@pytest.fixture()
def tool_bus():
    """A started MCPToolBus, stopped at teardown."""
    bus = MCPToolBus()
    bus.start()
    try:
        yield bus
    finally:
        bus.stop()


@pytest.fixture()
def mcp_toolset(tool_bus):
    """Factory: register the built-in servers a member references and return its toolset.

    Usage::

        ts = mcp_toolset("agent", ["code/*"], workspace_path=tmp_path)
    """

    def _make(
        member_name: str,
        patterns: list[str],
        *,
        workspace_path,
        memory=None,
        beliefs=None,
        peers=None,
        registry_url=None,
        bridge_secret=None,
        tool_timeout: int = 30,
    ) -> MemberToolset:
        ctx = BuiltinContext(
            workspace_path=workspace_path,
            member_name=member_name,
            memory=memory,
            beliefs=beliefs,
            peers=peers or {},
            registry_url=registry_url,
            bridge_secret=bridge_secret,
            tool_timeout=tool_timeout,
        )
        referenced = {p.split("/", 1)[0] for p in patterns}
        for name in referenced:
            if name in BUILTIN_SERVERS:
                tool_bus.add_inprocess_server(name, BUILTIN_SERVERS[name](ctx), owner=member_name)
        return MemberToolset(tool_bus, member_name, patterns)

    return _make
