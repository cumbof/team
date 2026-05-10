"""High-level orchestrator: ties containers, members, transcript, workspace,
and the chosen workflow together.
"""

from __future__ import annotations

import logging
from pathlib import Path

from team.bus import Transcript
from team.config import TeamConfig
from team.container import ContainerManager
from team.member import Member
from team.workflows import get_workflow
from team.workspace import SharedWorkspace

log = logging.getLogger(__name__)


class Orchestrator:
    def __init__(self, team: TeamConfig, container_manager: ContainerManager | None = None):
        self.team = team
        self.containers = container_manager or ContainerManager(team)
        self.workspace = SharedWorkspace(team.workspace)
        self.transcript = Transcript(persist_path=team.workspace / "transcript.jsonl")
        self.members: dict[str, Member] = {}

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

    def run_turn(self, member_name: str, prompt: str | None = None):
        member = self.members[member_name]
        log.info("turn: @%s", member_name)
        result = member.take_turn(self.transcript, self.workspace, prompt=prompt)
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
