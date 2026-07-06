"""The MCP tool bus: one uniform bus for every tool call in team.

Architecture
------------
The rest of team is synchronous (member turns run on plain threads, including
the :class:`~concurrent.futures.ThreadPoolExecutor` used by parallel
workflows).  The ``mcp`` SDK is asyncio-native.  We bridge the two with **one**
background daemon thread running a dedicated event loop:

* :meth:`MCPToolBus.start` spawns the thread and enters an
  :class:`~contextlib.AsyncExitStack` that owns every open session/transport.
* Every sync method (``add_*``, :meth:`list_tools`, :meth:`call_tool`,
  :meth:`stop`) hops onto that loop via
  :func:`asyncio.run_coroutine_threadsafe` and blocks on the returned future.
  This is safe to call from any thread, including the parallel-member workers.

Server identity
---------------
Built-in servers are **per-member** (each member gets its own FastMCP instance
closing over its identity — see :mod:`team.mcp.builtin`), so connections are
keyed by ``(owner, server_name)`` where *owner* is the member name.  External
servers are **shared** for the whole team (``owner=None``); MCP multiplexes
concurrent requests by request-id, so parallel member threads share one session
safely.  Tool resolution prefers a member-owned server, then falls back to a
shared one.

Naming
------
The model-visible ("wire") name of a tool is ``server__tool``.  Server names
match ``^[a-z][a-z0-9-]*$`` (no underscores), so ``wire.split("__", 1)`` always
recovers ``(server, tool)`` even though tool names contain underscores.
"""

from __future__ import annotations

import asyncio
import fnmatch
import logging
import threading
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from datetime import timedelta
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mcp import ClientSession
    from mcp.server.fastmcp import FastMCP

log = logging.getLogger(__name__)

#: Maximum characters returned by any tool call (mirrors the old tools._MAX_OUTPUT).
_MAX_OUTPUT = 8192


def wire_name(server: str, tool: str) -> str:
    """Build the model-visible tool name from *server* and *tool*."""
    return f"{server}__{tool}"


def split_wire_name(name: str) -> tuple[str, str]:
    """Split a ``server__tool`` wire name into ``(server, tool)``.

    Falls back to ``("", name)`` when the name is not qualified.
    """
    if "__" in name:
        server, tool = name.split("__", 1)
        return server, tool
    return "", name


@dataclass(frozen=True)
class ToolInfo:
    """Static description of a single tool exposed on the bus."""

    server: str
    name: str
    wire_name: str
    description: str
    input_schema: dict


@dataclass
class _Conn:
    """A live (or dead) connection to one MCP server."""

    name: str
    owner: str | None
    session: "ClientSession | None" = None
    tools: list[ToolInfo] = field(default_factory=list)
    dead: bool = False
    cause: str = ""


class MCPToolBus:
    """Owns the background event loop and every MCP session for a team."""

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        # keyed by (owner, server_name); owner=None for shared external servers.
        self._conns: dict[tuple[str | None, str], _Conn] = {}
        self._started = False
        # The manager task owns the AsyncExitStack.  All context managers from
        # the mcp SDK use anyio cancel scopes that must be entered and exited in
        # the *same* task, so every enter (add_*) and the final aclose (stop)
        # run inside this one long-lived task, driven by a command queue.
        self._cmd_queue: "asyncio.Queue | None" = None
        self._manager_future: Any = None
        self._queue_ready = threading.Event()

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """Start the background loop thread + manager task.  Idempotent."""
        if self._started:
            return
        loop = asyncio.new_event_loop()
        ready = threading.Event()

        def _run() -> None:
            asyncio.set_event_loop(loop)
            ready.set()
            loop.run_forever()

        thread = threading.Thread(target=_run, name="mcp-bus", daemon=True)
        thread.start()
        ready.wait()
        self._loop = loop
        self._thread = thread
        self._queue_ready.clear()
        self._manager_future = asyncio.run_coroutine_threadsafe(self._manager(), loop)
        self._queue_ready.wait()  # block until the command queue exists
        self._started = True
        log.debug("MCPToolBus started")

    async def _manager(self) -> None:
        """Own the AsyncExitStack and serialise all context enter/exit here."""
        self._cmd_queue = asyncio.Queue()
        self._queue_ready.set()
        async with AsyncExitStack() as stack:
            while True:
                cmd = await self._cmd_queue.get()
                if cmd is None:  # shutdown sentinel
                    break
                fn, fut = cmd
                try:
                    fut.set_result(await fn(stack))
                except Exception as exc:  # noqa: BLE001
                    fut.set_exception(exc)
            # aclose happens on __aexit__ of `async with`, in this same task.

    def stop(self, timeout: float = 10.0) -> None:
        """Close all sessions/transports and stop the loop.  No-op if never started."""
        if not self._started or self._loop is None:
            return
        loop = self._loop

        async def _send_stop() -> None:
            assert self._cmd_queue is not None
            await self._cmd_queue.put(None)

        try:
            asyncio.run_coroutine_threadsafe(_send_stop(), loop).result(timeout=timeout)
            # Wait for the manager task to unwind the stack (reaps stdio children).
            if self._manager_future is not None:
                self._manager_future.result(timeout=timeout)
        except Exception as exc:  # noqa: BLE001 — a wedged child must not hang shutdown
            log.warning("MCPToolBus.stop: error closing sessions (%s); forcing loop stop", exc)
        finally:
            loop.call_soon_threadsafe(loop.stop)
            if self._thread is not None:
                self._thread.join(timeout=timeout)
            self._loop = None
            self._thread = None
            self._cmd_queue = None
            self._manager_future = None
            self._conns.clear()
            self._started = False
            log.debug("MCPToolBus stopped")

    def _run(self, coro: Any, timeout: float | None = None) -> Any:
        """Run *coro* on the bus loop from a sync caller and return its result."""
        if not self._started or self._loop is None:
            raise RuntimeError("MCPToolBus is not started")
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result(timeout=timeout)

    def _submit(self, opener: Any, timeout: float | None = None) -> Any:
        """Run *opener(stack)* inside the manager task and return its result."""
        async def _enqueue() -> Any:
            assert self._loop is not None and self._cmd_queue is not None
            fut = self._loop.create_future()
            await self._cmd_queue.put((opener, fut))
            return await fut

        return self._run(_enqueue(), timeout=timeout)

    # ------------------------------------------------------------------ #
    # Server registration
    # ------------------------------------------------------------------ #

    def add_inprocess_server(
        self, name: str, server: "FastMCP", *, owner: str | None = None
    ) -> None:
        """Connect an in-process FastMCP *server* via the SDK in-memory transport."""
        from mcp.shared.memory import create_connected_server_and_client_session as _conn

        async def _open(stack: AsyncExitStack) -> _Conn:
            session = await stack.enter_async_context(_conn(server))
            await session.initialize()
            return await self._build_conn(name, owner, session)

        self._register(name, owner, _open)

    def add_stdio_server(
        self,
        name: str,
        command: str,
        args: list[str],
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> None:
        """Connect an external MCP server launched as a stdio subprocess (shared)."""
        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client

        params = StdioServerParameters(
            command=command, args=list(args), env=env or None, cwd=cwd
        )

        async def _open(stack: AsyncExitStack) -> _Conn:
            read, write = await stack.enter_async_context(stdio_client(params))
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            return await self._build_conn(name, None, session)

        self._register(name, None, _open)

    def add_http_server(
        self, name: str, url: str, headers: dict[str, str] | None = None
    ) -> None:
        """Connect an external MCP server over Streamable HTTP (shared)."""
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        async def _open(stack: AsyncExitStack) -> _Conn:
            read, write, _ = await stack.enter_async_context(
                streamablehttp_client(url, headers=headers or None)
            )
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            return await self._build_conn(name, None, session)

        self._register(name, None, _open)

    def _register(self, name: str, owner: str | None, opener: Any) -> None:
        """Open a connection (via *opener*) and store it; failures mark it dead."""
        key = (owner, name)
        try:
            conn = self._submit(opener)
        except Exception as exc:  # noqa: BLE001 — a bad server must not abort startup
            log.error("MCP server %r failed to connect: %s", name, exc)
            self._conns[key] = _Conn(name=name, owner=owner, dead=True, cause=str(exc))
            return
        self._conns[key] = conn
        log.info(
            "MCP server %r connected (%d tool(s)%s)",
            name, len(conn.tools), f", owner={owner}" if owner else "",
        )

    async def _build_conn(
        self, name: str, owner: str | None, session: "ClientSession"
    ) -> _Conn:
        """List a freshly connected server's tools and wrap them in a :class:`_Conn`."""
        listed = await session.list_tools()
        tools = [
            ToolInfo(
                server=name,
                name=t.name,
                wire_name=wire_name(name, t.name),
                description=t.description or "",
                input_schema=t.inputSchema or {"type": "object", "properties": {}},
            )
            for t in listed.tools
        ]
        return _Conn(name=name, owner=owner, session=session, tools=tools)

    # ------------------------------------------------------------------ #
    # Discovery + dispatch
    # ------------------------------------------------------------------ #

    def _resolve(self, server: str, owner: str | None) -> _Conn | None:
        """Find the connection for *server*, preferring member-owned over shared."""
        if owner is not None and (owner, server) in self._conns:
            return self._conns[(owner, server)]
        return self._conns.get((None, server))

    def list_tools(self, *, owner: str | None = None) -> list[ToolInfo]:
        """Return every tool visible to *owner* (its own servers + shared servers)."""
        out: list[ToolInfo] = []
        seen: set[str] = set()
        for (conn_owner, _name), conn in self._conns.items():
            if conn_owner not in (None, owner):
                continue
            for ti in conn.tools:
                if ti.wire_name not in seen:
                    seen.add(ti.wire_name)
                    out.append(ti)
        return out

    def call_tool(
        self,
        name: str,
        arguments: dict,
        *,
        owner: str | None = None,
        timeout: float | None = None,
    ) -> str:
        """Call a tool by wire name and return its output as a plain string.

        Never raises into the caller: unknown tools, dead servers, timeouts, and
        transport errors all come back as ``ERROR: ...`` strings so a single bad
        call cannot abort a member's turn.
        """
        server, _tool = split_wire_name(name)
        conn = self._resolve(server, owner)
        if conn is None:
            return f"ERROR: unknown tool {name!r}. Enabled tools: {self._enabled_names(owner)}"
        if conn.dead or conn.session is None:
            return f"ERROR: MCP server {server!r} is not running ({conn.cause or 'unavailable'})"
        if not any(ti.name == _tool for ti in conn.tools):
            return f"ERROR: unknown tool {name!r}. Enabled tools: {self._enabled_names(owner)}"

        read_timeout = timedelta(seconds=timeout) if timeout else None

        async def _call() -> Any:
            return await conn.session.call_tool(
                _tool, arguments, read_timeout_seconds=read_timeout
            )

        # Guard against a hung transport that ignores the protocol-level timeout.
        wall = (timeout + 5) if timeout else None
        try:
            result = self._run(_call(), timeout=wall)
        except TimeoutError:
            return f"ERROR: tool {name!r} timed out after {timeout}s"
        except Exception as exc:  # noqa: BLE001
            if timeout and "timed out" in str(exc).lower():
                return f"ERROR: tool {name!r} timed out after {timeout}s"
            return f"ERROR: tool {name!r} failed: {exc}"
        return _result_to_str(result)

    def _enabled_names(self, owner: str | None) -> str:
        names = sorted(ti.wire_name for ti in self.list_tools(owner=owner))
        return ", ".join(names) if names else "(none)"


def _result_to_str(result: Any) -> str:
    """Flatten an MCP ``CallToolResult`` into a single string with ERROR prefixing."""
    parts: list[str] = []
    for block in result.content or []:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
        else:
            parts.append(f"[{getattr(block, 'type', 'unknown')} content omitted]")
    text = "\n".join(parts) if parts else "(no output)"
    if getattr(result, "isError", False) and not text.lstrip().startswith("ERROR"):
        text = f"ERROR: {text}"
    return text


class MemberToolset:
    """A single member's view of the bus: enablement filtering + truncation.

    *patterns* are the member's configured ``tools:`` entries in ``server/tool``
    or ``server/*`` form (e.g. ``["code/*", "workspace/read_file"]``).
    """

    def __init__(self, bus: MCPToolBus, member_name: str, patterns: list[str]) -> None:
        self._bus = bus
        self._member = member_name
        self._patterns = list(patterns or [])

    def _matches(self, ti: ToolInfo) -> bool:
        for pat in self._patterns:
            if "/" not in pat:
                continue
            p_server, p_tool = pat.split("/", 1)
            if p_server != ti.server:
                continue
            if p_tool == "*" or fnmatch.fnmatch(ti.name, p_tool):
                return True
        return False

    def tools(self) -> list[ToolInfo]:
        """Return the enabled :class:`ToolInfo` objects for this member."""
        return [ti for ti in self._bus.list_tools(owner=self._member) if self._matches(ti)]

    def is_enabled(self, name: str) -> bool:
        """Return whether *name* (a wire name) is enabled for this member."""
        return any(ti.wire_name == name for ti in self.tools())

    def call(self, name: str, arguments: dict, timeout: float) -> str:
        """Call an enabled tool; enforce enablement then truncate the output."""
        if not self.is_enabled(name):
            enabled = ", ".join(sorted(ti.wire_name for ti in self.tools())) or "(none)"
            return f"ERROR: tool {name!r} is not enabled for this member. Enabled: {enabled}"
        out = self._bus.call_tool(
            name, arguments, owner=self._member, timeout=timeout
        )
        if len(out) > _MAX_OUTPUT:
            out = out[:_MAX_OUTPUT] + f"\n[… truncated at {_MAX_OUTPUT} chars]"
        return out
