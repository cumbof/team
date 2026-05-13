"""High-level orchestrator: ties containers, members, transcript, workspace,
and the chosen workflow together.
"""

from __future__ import annotations

import concurrent.futures
import logging
from pathlib import Path
from typing import Callable

from team.bus import Transcript
from team.config import TeamConfig, resolve_member_setting
from team.container import ContainerManager
from team.member import DONE_TOKEN, Member, TurnResult
from team.workflows import get_workflow
from team.workspace import CheckpointManager, SharedWorkspace

log = logging.getLogger(__name__)


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

            self.members[rt.member.name] = Member(
                self.team, rt.member, rt,
                memory=member_memory,
                beliefs=self.beliefs,
            )

        for m in self.members.values():
            m.prepare(deadline_seconds=prepare_deadline_seconds)
        self._kickoff()

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
