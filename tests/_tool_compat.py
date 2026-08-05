"""Test shim: run a federation built-in tool from the old KV-body call style.

The federation tests were written against the pre-MCP ``execute_tool(name, body,
**ctx)`` API where *body* was a ``key: value`` string.  Rather than rewrite every
call site, this shim parses that same KV body into an arguments dict and drives
the real ``federation`` MCP server through a one-shot bus — so the tests exercise
the new code path unchanged (mocked BridgeClient / RegistryClient still apply,
since the tools import them at call time).
"""

from __future__ import annotations

from pathlib import Path

from team.mcp.bus import MCPToolBus
from team.mcp.builtin import BUILTIN_SERVERS, BuiltinContext

# Tools in this shim all live on the federation server.
_FEDERATION_TOOLS = {
    "list_peers",
    "delegate_task",
    "broadcast_task",
    "cancel_remote_task",
    "query_registry",
    "sync_beliefs",
}
_INT_KEYS = {"limit", "task_timeout"}


def _kv_to_args(name: str, body: str) -> dict:
    """Parse an old KV-style tool body into a new-style arguments dict."""
    args: dict = {}
    tags: list[str] = []
    for line in (body or "").splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key, value = key.strip(), value.strip()
        if key == "tag":  # query_registry: repeatable
            tags.append(value)
            continue
        args[key] = value
    if tags:
        args["tags"] = tags
    # Renames from the old body keys to the new parameter names.
    if name in ("delegate_task", "broadcast_task") and "timeout" in args:
        args["task_timeout"] = args.pop("timeout")
    if name == "broadcast_task" and "peers" in args:
        args["peers_list"] = args.pop("peers")
    for k in list(args):
        if k in _INT_KEYS:
            try:
                args[k] = int(args[k])
            except (TypeError, ValueError):
                pass
    return args


def execute_tool(
    name: str,
    body: str = "",
    *,
    workspace_path=None,
    peers=None,
    registry_url=None,
    bridge_secret=None,
    beliefs=None,
    member_name: str = "tester",
    timeout: int = 30,
    **_ignored,
) -> str:
    """Run a federation tool from a KV body and return its string output."""
    if name not in _FEDERATION_TOOLS:
        raise ValueError(f"_tool_compat only supports federation tools, not {name!r}")
    args = _kv_to_args(name, body)
    bus = MCPToolBus()
    bus.start()
    try:
        ctx = BuiltinContext(
            workspace_path=Path(workspace_path) if workspace_path else Path("."),
            member_name=member_name,
            peers=peers or {},
            registry_url=registry_url,
            bridge_secret=bridge_secret,
            beliefs=beliefs,
            tool_timeout=timeout,
        )
        bus.add_inprocess_server("federation", BUILTIN_SERVERS["federation"](ctx), owner=member_name)
        return bus.call_tool(f"federation__{name}", args, owner=member_name, timeout=timeout)
    finally:
        bus.stop()
