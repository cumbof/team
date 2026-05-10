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
from team.workspace import SharedWorkspace

log = logging.getLogger(__name__)


class Orchestrator:
    def __init__(self, team: TeamConfig, container_manager: ContainerManager | None = None, resume: bool = False):
        self.team = team
        self.containers = container_manager or ContainerManager(team)
        self.workspace = SharedWorkspace(team.workspace)
        self.transcript = Transcript(
            persist_path=team.workspace / "transcript.jsonl",
            resume=resume,
        )
        self.members: dict[str, Member] = {}
        # Member turns already persisted — used to fast-forward past them on resume.
        self._replay_queue: list = [
            t for t in self.transcript.turns if t.speaker != "orchestrator"
        ]
        # Optional streaming hooks — set by the CLI or other callers.
        # _on_turn_start(member_name) is called before a live LLM turn begins.
        # _on_token(token)            is called for each streamed content chunk.
        # _on_turn_end(member_name)   is called after the full reply is received.
        self._on_turn_start: Callable[[str], None] | None = None
        self._on_token: Callable[[str], None] | None = None
        self._on_turn_end: Callable[[str], None] | None = None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def up(self, prepare_deadline_seconds: int = 300) -> None:
        """Start containers and ensure all models are pulled."""
        runtimes = self.containers.start_all()
        for rt in runtimes:
            self.members[rt.member.name] = Member(self.team, rt.member, rt)
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
    # Turn execution (used by workflows)
    # ------------------------------------------------------------------ #

    def run_turn(self, member_name: str, prompt: str | None = None) -> TurnResult:
        # If a cached turn for this member is next in the replay queue, replay it
        # without calling the LLM (the result is already persisted on disk).
        if self._replay_queue and self._replay_queue[0].speaker == member_name:
            cached = self._replay_queue.pop(0)
            log.info(
                "resume: replaying turn %d for @%s (skipping LLM call)",
                cached.index, member_name,
            )
            for path in cached.files_written:
                self.workspace.touch(path)
            return TurnResult(
                content=cached.content,
                declared_done=DONE_TOKEN in cached.content,
                files_written=cached.files_written,
            )

        member = self.members[member_name]
        log.info("turn: @%s", member_name)
        if self._on_turn_start:
            self._on_turn_start(member_name)
        result = member.take_turn(
            self.transcript, self.workspace,
            prompt=prompt,
            token_callback=self._on_token,
        )
        if self._on_turn_end:
            self._on_turn_end(member_name)
        self.transcript.append(
            speaker=member.name,
            role=member.config.role,
            content=result.content,
            files_written=result.files_written,
        )
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
