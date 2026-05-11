"""High-level orchestrator: ties containers, members, transcript, workspace,
and the chosen workflow together.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from team.bus import Transcript
from team.config import TeamConfig
from team.container import ContainerManager
from team.member import DONE_TOKEN, Member, TurnResult
from team.workflows import get_workflow
from team.workspace import CheckpointManager, SharedWorkspace

log = logging.getLogger(__name__)


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

        # Snapshot the shared workspace before this member writes anything so
        # users can restore the project to this state if the turn produces
        # undesirable changes.
        self.checkpoints.create(len(self.transcript.turns), member_name)

        member = self.members[member_name]
        log.info("turn: @%s", member_name)
        if self._on_turn_start:
            self._on_turn_start(member_name)
        result = member.take_turn(
            self.transcript, self.workspace,
            prompt=prompt,
            token_callback=self._on_token,
            on_tool_call=self._on_tool_call,
            on_tool_result=self._on_tool_result,
        )
        if self._on_turn_end:
            self._on_turn_end(member_name)
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

    # ------------------------------------------------------------------ #
    # Drive a workflow
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        wf = get_workflow(self.team.workflow.type)
        wf(self)

    # ------------------------------------------------------------------ #
    # Inspection helpers
    # ------------------------------------------------------------------ #

    def status(self) -> list[dict]:
        return self.containers.status()

    def transcript_path(self) -> Path:
        return self.team.workspace / "transcript.jsonl"
