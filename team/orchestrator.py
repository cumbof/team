"""High-level orchestrator: ties containers, members, transcript, workspace,
and the chosen workflow together.
"""

from __future__ import annotations

import concurrent.futures
import logging
from pathlib import Path
from typing import Callable

from team.bus import Transcript
from team.config import TeamConfig, expand_env, resolve_member_setting
from team.container import ContainerManager
from team.member import DONE_TOKEN, Member, TurnResult
from team.mcp import MCPToolBus, MemberToolset
from team.mcp.builtin import BUILTIN_SERVERS, BuiltinContext
from team.workflows import get_workflow
from team.workspace import CheckpointManager, SharedWorkspace

log = logging.getLogger(__name__)


def _resolve_entry_point(ref: str):
    """Resolve an ``entry_point`` reference to a FastMCP server instance.

    *ref* is either a ``module:attr`` path or a name registered under the
    ``team.mcp_servers`` entry-point group.  The referenced object is a zero-arg
    callable returning a ``FastMCP`` (or a ``FastMCP`` directly).
    """
    factory = None
    if ":" in ref:
        import importlib

        module_path, attr = ref.split(":", 1)
        factory = getattr(importlib.import_module(module_path), attr)
    else:
        from importlib.metadata import entry_points

        for ep in entry_points(group="team.mcp_servers"):
            if ep.name == ref:
                factory = ep.load()
                break
        if factory is None:
            raise ValueError(
                f"entry_point {ref!r} is not registered under the 'team.mcp_servers' group"
            )
    return factory() if callable(factory) else factory


class TurnTimeoutError(RuntimeError):
    """Raised when a member's turn exceeds its configured ``turn_timeout``."""


class TokenBudgetError(RuntimeError):
    """Raised when a member has exhausted its configured ``token_budget``."""


class Orchestrator:
    def __init__(self, team: TeamConfig, container_manager: ContainerManager | None = None, resume: bool = False):
        self.team = team
        self.containers = container_manager or ContainerManager(team)
        self.workspace = SharedWorkspace(team.workspace)
        self.checkpoints = CheckpointManager(team.workspace)
        self.transcript = Transcript(
            persist_path=team.workspace / "transcript.jsonl",
            resume=resume,
        )
        self.members: dict[str, Member] = {}
        self._replay_queue: list = [
            t for t in self.transcript.turns if t.speaker not in ("orchestrator", "human")
        ]
        self._on_turn_start: Callable[[str], None] | None = None
        self._on_token: Callable[[str], None] | None = None
        self._on_turn_end: Callable[[str], None] | None = None
        self._on_round_end: Callable[[int], None] | None = None
        # F4: agent tool-use hooks for console display.
        self._on_tool_call: Callable[[str, str, str], None] | None = None
        self._on_tool_result: Callable[[str, str, str], None] | None = None
        # F14: parallel round hooks.
        self._on_parallel_round_start: "Callable[[list[str]], None] | None" = None
        self._on_parallel_round_end: "Callable[[list[tuple[str, TurnResult]]], None] | None" = None
        self.inject_path: Path = team.workspace / "inject.txt"
        # F6: token usage accumulated across live turns (not replayed ones).
        self._token_totals: dict[str, dict[str, int]] = {}
        # Shared belief board (None when beliefs.enabled is False).
        self.beliefs = None
        # MCP tool bus (started in up(), stopped in down()).
        self.tool_bus = MCPToolBus()

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def up(self, prepare_deadline_seconds: int = 300) -> None:
        """Start containers and ensure all models are pulled."""
        # Optionally create the shared belief board.
        if self.team.beliefs.enabled:
            from team.beliefs import BeliefBoard
            beliefs_path = self.team.workspace / "beliefs.json"
            self.beliefs = BeliefBoard(
                path=beliefs_path,
                member_names=self.team.member_names(),
                consensus_threshold=self.team.beliefs.consensus_threshold,
            )
            log.info("beliefs: board enabled at %s", beliefs_path)

        # Start the MCP tool bus and connect any external servers (shared).
        self.tool_bus.start()
        self._connect_external_servers()

        runtimes = self.containers.start_all()
        for rt in runtimes:
            # Optionally create a per-member persistent memory store.
            member_memory = None
            if self.team.memory.enabled:
                from team.memory import AgentMemory
                mem_dir = (
                    Path(self.team.memory.store).expanduser()
                    if self.team.memory.store
                    else self.team.workspace / "memory"
                )
                mem_path = mem_dir / f"{rt.member.name}.db"
                member_memory = AgentMemory(mem_path)
                log.info("memory: enabled for @%s at %s", rt.member.name, mem_path)

            toolset = self._build_member_toolset(rt.member, member_memory)

            self.members[rt.member.name] = Member(
                self.team, rt.member, rt,
                memory=member_memory,
                beliefs=self.beliefs,
                toolset=toolset,
            )

        for m in self.members.values():
            m.prepare(deadline_seconds=prepare_deadline_seconds)
        self._kickoff()

    # ------------------------------------------------------------------ #
    # Tool bus wiring
    # ------------------------------------------------------------------ #

    def _connect_external_servers(self) -> None:
        """Connect the team's declared ``mcp_servers`` as shared bus servers.

        A connection failure is logged and the server is marked dead (its tools
        return an error string) but never aborts startup.
        """
        base = (
            self.team.source_path.parent
            if self.team.source_path is not None
            else self.team.workspace
        )
        for sc in self.team.mcp_servers:
            try:
                if sc.transport == "stdio":
                    env = {k: expand_env(v, default="") for k, v in sc.env.items()}
                    env["TEAM_WORKSPACE"] = str(self.team.workspace)
                    self.tool_bus.add_stdio_server(
                        sc.name, sc.command, sc.args, env=env, cwd=str(base)
                    )
                elif sc.transport == "http":
                    headers = {k: expand_env(v, default="") for k, v in sc.headers.items()}
                    self.tool_bus.add_http_server(sc.name, sc.url, headers=headers)
                elif sc.transport == "entry_point":
                    server = _resolve_entry_point(sc.entry_point)
                    self.tool_bus.add_inprocess_server(sc.name, server)
            except Exception as exc:  # noqa: BLE001 — a bad server must not abort startup
                log.error("mcp server %r failed to connect: %s", sc.name, exc)

    def _build_member_toolset(self, member, member_memory) -> MemberToolset:
        """Register the built-in servers a member references and return its toolset."""
        patterns = (
            member.tools if member.tools is not None else self.team.defaults.tools
        ) or []
        referenced = {p.split("/", 1)[0] for p in patterns}

        # Resolve + validate the effective sandbox mode.
        from team.mcp.builtin.code import VALID_TOOL_SANDBOXES

        sandbox = resolve_member_setting(member, self.team.defaults, "tool_sandbox") or "none"
        if sandbox not in VALID_TOOL_SANDBOXES:
            log.warning(
                "@%s: unknown tool_sandbox value %r; falling back to 'none'. Valid: %s",
                member.name, sandbox, ", ".join(sorted(VALID_TOOL_SANDBOXES)),
            )
            sandbox = "none"

        # Security advisory: code execution on the host with no sandbox.
        using_host_ollama = bool(member.ollama_url or self.team.defaults.ollama_url)
        if "code" in referenced and using_host_ollama and sandbox == "none":
            log.warning(
                "SECURITY: @%s has code tools enabled in host-Ollama mode (no Docker "
                "isolation). Code will execute on the host with the privileges of this "
                "process. Set tool_sandbox: firejail or tool_sandbox: bubblewrap in your "
                "team YAML to limit exposure.",
                member.name,
            )

        tool_timeout = resolve_member_setting(member, self.team.defaults, "tool_timeout") or 300
        ctx = BuiltinContext(
            workspace_path=self.team.workspace,
            member_name=member.name,
            memory=member_memory,
            beliefs=self.beliefs,
            bridge_secret=self.team.bridge.secret,
            peers=dict(self.team.bridge.peers or {}),
            registry_url=self.team.registry.url,
            sandbox=sandbox,
            tool_timeout=tool_timeout,
        )
        for name in referenced:
            if name in BUILTIN_SERVERS:
                self.tool_bus.add_inprocess_server(
                    name, BUILTIN_SERVERS[name](ctx), owner=member.name
                )
        return MemberToolset(self.tool_bus, member.name, patterns)

    def _kickoff(self) -> None:
        if self.transcript.turns:
            return  # resuming
        self.transcript.append(
            speaker="orchestrator",
            role="system",
            content=(
                f"# Team `{self.team.name}` convened\n"
                f"## Goal\n{self.team.goal.strip()}\n\n"
                f"## Members\n"
                + "\n".join(
                    f"- @{m.name} — {m.role} (model: {m.model})"
                    for m in self.team.members
                )
                + f"\n\n## Workflow\n{self.team.workflow.type} "
                f"(max_rounds={self.team.workflow.max_rounds})"
            ),
        )

    def down(self, *, remove_volumes: bool = False) -> None:
        try:
            self._down_body(remove_volumes=remove_volumes)
        finally:
            self.tool_bus.stop()

    def _down_body(self, *, remove_volumes: bool = False) -> None:
        # For local-Ollama members (no Docker container), optionally evict the
        # loaded model from memory so it doesn't linger after the run.
        for member in self.members.values():
            should_unload = resolve_member_setting(
                member.config, self.team.defaults, "unload_on_exit"
            )
            if not should_unload:
                continue
            # Only applies to local Ollama — Docker containers are torn down
            # below anyway, and openai_compat has no model to evict.
            from team.ollama_client import OllamaClient  # avoid circular at module level
            if member.runtime.container is not None or not isinstance(member.client, OllamaClient):
                continue
            try:
                member.client.unload_model(member.config.model)
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "could not unload model %s for @%s: %s",
                    member.config.model, member.name, exc,
                )
        self.containers.stop_all(remove_volumes=remove_volumes)

    # ------------------------------------------------------------------ #
    # Human-in-the-loop injection
    # ------------------------------------------------------------------ #

    def inject_directive(self, content: str) -> None:
        """Inject a human directive into the transcript immediately.

        The directive is appended as a ``speaker="human"`` turn so every
        member sees it in the next turn's conversation context.
        """
        text = content.strip()
        if not text:
            return
        self.transcript.append(speaker="human", role="director", content=text)
        log.info("inject: human directive added to transcript (%d chars)", len(text))

    def _check_inject(self) -> None:
        """Read and consume the inject file if it exists, adding its contents
        to the transcript as a human directive."""
        try:
            if self.inject_path.is_file():
                content = self.inject_path.read_text(encoding="utf-8").strip()
                self.inject_path.unlink()
                if content:
                    self.inject_directive(content)
        except OSError as exc:
            log.warning("inject: failed to read/delete inject.txt: %s", exc)

    # ------------------------------------------------------------------ #
    # Turn execution (used by workflows)
    # ------------------------------------------------------------------ #

    def run_turn(self, member_name: str, prompt: str | None = None) -> TurnResult:
        # If a cached turn for this member is next in the replay queue, replay it
        # without calling the LLM (the result is already persisted on disk).
        # We match on speaker name so that a workflow path change (e.g. a different
        # manager decision) naturally falls through to the live path instead of
        # replaying the wrong cached turn.
        if self._replay_queue and self._replay_queue[0].speaker == member_name:
            cached = self._replay_queue.pop(0)
            log.info(
                "resume: replaying turn %d for @%s (skipping LLM call)",
                cached.index, member_name,
            )
            # Restore the "recently changed" tracking so the next live turn's
            # context section lists the replayed files correctly.
            for path in cached.files_written:
                self.workspace.touch(path)
            return TurnResult(
                content=cached.content,
                declared_done=DONE_TOKEN in cached.content,
                files_written=cached.files_written,
            )

        # Check for a human directive dropped into inject.txt before this turn.
        # inject.txt is only consumed on the live path — not during replay —
        # so human directives always land in the correct position in the
        # transcript relative to the new live turns.
        self._check_inject()

        # F15: token budget check — only on the live path (replay never consumes
        # real tokens and must not be blocked by a spent budget).
        member = self.members[member_name]
        token_budget = resolve_member_setting(member.config, self.team.defaults, "token_budget")
        if token_budget:
            totals = self._token_totals.get(member_name, {"prompt": 0, "completion": 0})
            used = totals["prompt"] + totals["completion"]
            if used >= token_budget:
                raise TokenBudgetError(
                    f"@{member_name} has exhausted its token budget "
                    f"({used:,} of {token_budget:,} tokens used)"
                )

        # Snapshot the shared workspace before this member writes anything so
        # users can restore the project to this state if the turn produces
        # undesirable changes.
        self.checkpoints.create(len(self.transcript.turns), member_name)

        log.info("turn: @%s", member_name)
        if self._on_turn_start:
            self._on_turn_start(member_name)

        # F12: per-turn timeout — wrap the member turn in a thread so we can
        # apply a wall-clock deadline without blocking the event loop.
        turn_timeout = resolve_member_setting(member.config, self.team.defaults, "turn_timeout")
        _take_turn_kwargs = dict(
            transcript=self.transcript,
            workspace=self.workspace,
            prompt=prompt,
            token_callback=self._on_token,
            on_tool_call=self._on_tool_call,
            on_tool_result=self._on_tool_result,
        )
        if turn_timeout:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as _executor:
                _future = _executor.submit(member.take_turn, **_take_turn_kwargs)
                try:
                    result = _future.result(timeout=turn_timeout)
                except concurrent.futures.TimeoutError:
                    raise TurnTimeoutError(
                        f"@{member_name} turn timed out after {turn_timeout}s"
                    )
        else:
            result = member.take_turn(**_take_turn_kwargs)
        if self._on_turn_end:
            self._on_turn_end(member_name)

        # Warn when the member's context window is near-full.  A saturated
        # context causes Ollama to silently truncate early turns, which makes
        # small models lose track of the conversation and produce garbage output.
        eff_ctx = resolve_member_setting(member.config, self.team.defaults, "context_window")
        if eff_ctx and result.prompt_tokens >= int(eff_ctx * 0.9):
            log.warning(
                "@%s context window near-full: %d / %d prompt tokens used (%.0f%%). "
                "Consider raising context_window or setting "
                "context_strategy: sliding_window in the team config.",
                member_name,
                result.prompt_tokens,
                eff_ctx,
                100.0 * result.prompt_tokens / eff_ctx,
            )

        self.transcript.append(
            speaker=member.name,
            role=member.config.role,
            content=result.content,
            files_written=result.files_written,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
        )
        # Accumulate token usage for the live turn (F6).
        totals = self._token_totals.setdefault(member_name, {"prompt": 0, "completion": 0})
        totals["prompt"] += result.prompt_tokens
        totals["completion"] += result.completion_tokens
        return result

    def run_parallel_round(
        self,
        member_names: list[str],
        prompts: "dict[str, str | None] | None" = None,
    ) -> "list[tuple[str, TurnResult]]":
        """Run *member_names* concurrently and commit their turns in declaration order.

        All members receive the **same transcript snapshot** — the state at the
        start of the round — so no member sees another's reply from this batch.
        After all threads complete, turns are appended in the order given by
        *member_names*, keeping the transcript deterministic across resume runs.

        Per-member ``turn_timeout`` settings are honoured.  If a member exceeds
        its deadline a :exc:`TurnTimeoutError` is raised (for the first offender
        in declaration order) after the round completes.

        **Thread-safety note**: ``member.take_turn()`` reads the transcript
        (read-only during the parallel window) and writes to the shared workspace.
        Concurrent writes to the *same* file path are a race condition and should
        be avoided by ensuring parallel members work on disjoint paths.

        Parameters
        ----------
        member_names:
            Ordered list of member names to run.  Turns are committed in this
            order regardless of which thread finishes first.
        prompts:
            Optional per-member prompt overrides (keyed by member name).
        """
        prompts = prompts or {}

        # Separate cached (replay) members from live ones.
        replay: list[tuple[str, TurnResult]] = []
        to_run: list[str] = []
        for name in member_names:
            if self._replay_queue and self._replay_queue[0].speaker == name:
                cached = self._replay_queue.pop(0)
                log.info(
                    "resume: replaying parallel turn %d for @%s",
                    cached.index, name,
                )
                for path in cached.files_written:
                    self.workspace.touch(path)
                replay.append((name, TurnResult(
                    content=cached.content,
                    declared_done=DONE_TOKEN in cached.content,
                    files_written=cached.files_written,
                )))
            else:
                to_run.append(name)

        # Build result maps.
        replay_map: dict[str, TurnResult] = {n: r for n, r in replay}
        live_results: dict[str, TurnResult] = {}
        live_errors: dict[str, Exception] = {}

        # F15: filter out budget-exhausted members before dispatching threads.
        # Replay turns are exempt (they don't consume real tokens).
        non_exhausted: list[str] = []
        for name in to_run:
            member_obj = self.members[name]
            token_budget = resolve_member_setting(
                member_obj.config, self.team.defaults, "token_budget"
            )
            if token_budget:
                totals = self._token_totals.get(name, {"prompt": 0, "completion": 0})
                used = totals["prompt"] + totals["completion"]
                if used >= token_budget:
                    live_errors[name] = TokenBudgetError(
                        f"@{name} has exhausted its token budget "
                        f"({used:,} of {token_budget:,} tokens used)"
                    )
                    continue
            non_exhausted.append(name)
        to_run = non_exhausted

        if to_run:
            # Check for a human directive before launching threads.
            self._check_inject()

            # Create a checkpoint for each live member.
            turn_idx = len(self.transcript.turns)
            for name in to_run:
                self.checkpoints.create(turn_idx, name)

            if self._on_parallel_round_start:
                self._on_parallel_round_start(to_run)

            def _run_one(name: str) -> None:
                member = self.members[name]
                turn_timeout = resolve_member_setting(
                    member.config, self.team.defaults, "turn_timeout"
                )
                kwargs: dict = dict(
                    transcript=self.transcript,
                    workspace=self.workspace,
                    prompt=prompts.get(name),
                )
                try:
                    if turn_timeout:
                        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as _inner:
                            _fut = _inner.submit(member.take_turn, **kwargs)
                            try:
                                live_results[name] = _fut.result(timeout=turn_timeout)
                            except concurrent.futures.TimeoutError:
                                raise TurnTimeoutError(
                                    f"@{name} turn timed out after {turn_timeout}s"
                                )
                    else:
                        live_results[name] = member.take_turn(**kwargs)
                except Exception as exc:  # noqa: BLE001
                    live_errors[name] = exc

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=len(to_run), thread_name_prefix="parallel_member"
            ) as pool:
                futures = [pool.submit(_run_one, name) for name in to_run]
                concurrent.futures.wait(futures)

        # Commit results in declaration order; raise first error encountered.
        ordered: list[tuple[str, TurnResult]] = []
        first_error: Exception | None = None
        for name in member_names:
            if name in live_errors:
                if first_error is None:
                    first_error = live_errors[name]
                continue
            result = replay_map.get(name) or live_results.get(name)
            if result is None:
                continue
            if name in live_results:
                self.transcript.append(
                    speaker=name,
                    role=self.members[name].config.role,
                    content=result.content,
                    files_written=result.files_written,
                    prompt_tokens=result.prompt_tokens,
                    completion_tokens=result.completion_tokens,
                )
                totals = self._token_totals.setdefault(name, {"prompt": 0, "completion": 0})
                totals["prompt"] += result.prompt_tokens
                totals["completion"] += result.completion_tokens
            ordered.append((name, result))

        if self._on_parallel_round_end:
            self._on_parallel_round_end(ordered)

        if first_error is not None:
            raise first_error
        return ordered

    # ------------------------------------------------------------------ #
    # Drive a workflow
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        wf = get_workflow(self.team.workflow.type)
        wf(self)
        self._check_run_outcome()

    def _check_run_outcome(self) -> None:
        """Warn when the workflow finishes without clear evidence of success.

        Two conditions trigger a warning:

        * The workflow exhausted all rounds without any member emitting
          ``[[TEAM_DONE]]``.
        * Additionally, no workspace files were written during the run — the
          strongest signal that the team produced no tangible deliverable.
        """
        live_turns = [
            t for t in self.transcript.turns
            if t.speaker not in ("orchestrator", "human")
        ]
        if not live_turns:
            return

        declared_done = any(DONE_TOKEN in t.content for t in live_turns)
        files_produced = any(t.files_written for t in live_turns)

        if not declared_done and not files_produced:
            log.warning(
                "Run ended without [[TEAM_DONE]] and without any workspace "
                "files being written — the team goal may not have been achieved. "
                "Check the transcript and consider adjusting personas, "
                "model size, or context_window.",
            )
        elif not declared_done:
            log.warning(
                "Run ended without [[TEAM_DONE]]: the workflow exhausted all "
                "rounds before any member signalled completion. "
                "Consider increasing max_rounds or tightening member instructions.",
            )

    # ------------------------------------------------------------------ #
    # Inspection helpers
    # ------------------------------------------------------------------ #

    def status(self) -> list[dict]:
        return self.containers.status()

    def transcript_path(self) -> Path:
        return self.team.workspace / "transcript.jsonl"
