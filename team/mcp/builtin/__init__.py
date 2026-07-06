"""Built-in tools, grouped into 8 in-process MCP servers.

Each server is created **per member** by a factory that closes over a
:class:`BuiltinContext` carrying that member's identity and injected state
(workspace path, its own :class:`~team.memory.AgentMemory`, the shared
:class:`~team.beliefs.BeliefBoard`, bridge/registry settings, sandbox mode).
Baking identity into the server instance means the tools stay ordinary MCP
tools with no notion of a caller, and member attribution cannot be spoofed.

The 8 servers and their reserved names:

======================  ================================================
server                  tools
======================  ================================================
``code``                run_python, run_bash
``web``                 web_search, read_url
``workspace``           read_file, write_file, append_file, list_files
``memory``              remember, recall, forget, list_memories
``beliefs``             assert_belief, contest_belief, accept_belief,
                        list_beliefs
``decisions``           log_decision, read_decisions
``federation``          delegate_task, broadcast_task, list_peers,
                        cancel_remote_task, sync_beliefs, query_registry
``expert``              delegate_to_expert
======================  ================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from team.beliefs import BeliefBoard
    from team.memory import AgentMemory

#: The 8 reserved server names.  Config rejects mcp_servers colliding with these.
RESERVED_SERVER_NAMES: frozenset[str] = frozenset(
    {"code", "web", "workspace", "memory", "beliefs", "decisions", "federation", "expert"}
)


@dataclass
class BuiltinContext:
    """Per-member state injected into that member's built-in servers."""

    workspace_path: Path
    member_name: str
    memory: "AgentMemory | None" = None
    beliefs: "BeliefBoard | None" = None
    bridge_secret: str | None = None
    peers: dict[str, str] = field(default_factory=dict)
    registry_url: str | None = None
    sandbox: str = "none"
    tool_timeout: int = 300


def _server_factories() -> dict[str, Callable[[BuiltinContext], "FastMCP"]]:
    """Import and assemble the 8 factories lazily (keeps mcp import local)."""
    from team.mcp.builtin.beliefs import build as build_beliefs
    from team.mcp.builtin.code import build as build_code
    from team.mcp.builtin.decisions import build as build_decisions
    from team.mcp.builtin.expert import build as build_expert
    from team.mcp.builtin.federation import build as build_federation
    from team.mcp.builtin.memory import build as build_memory
    from team.mcp.builtin.web import build as build_web
    from team.mcp.builtin.workspace import build as build_workspace

    return {
        "code": build_code,
        "web": build_web,
        "workspace": build_workspace,
        "memory": build_memory,
        "beliefs": build_beliefs,
        "decisions": build_decisions,
        "federation": build_federation,
        "expert": build_expert,
    }


class _LazyRegistry:
    """Mapping of server name → factory, built on first access."""

    def __init__(self) -> None:
        self._cache: dict[str, Callable[[BuiltinContext], "FastMCP"]] | None = None

    def _get(self) -> dict[str, Callable[[BuiltinContext], "FastMCP"]]:
        if self._cache is None:
            self._cache = _server_factories()
        return self._cache

    def __getitem__(self, key: str) -> Callable[[BuiltinContext], "FastMCP"]:
        return self._get()[key]

    def __contains__(self, key: str) -> bool:
        return key in self._get()

    def __iter__(self):
        return iter(self._get())

    def keys(self):
        return self._get().keys()

    def items(self):
        return self._get().items()


#: name → factory(ctx) -> FastMCP.  Lazy so importing this package is cheap.
BUILTIN_SERVERS = _LazyRegistry()

__all__ = ["BuiltinContext", "BUILTIN_SERVERS", "RESERVED_SERVER_NAMES"]
