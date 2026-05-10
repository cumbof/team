"""A live :class:`Member` couples a config with its container runtime and
Ollama client, and knows how to take a turn given the current transcript.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

from team.bus import Transcript
from team.config import MemberConfig, TeamConfig, resolve_member_setting
from team.container import MemberRuntime
from team.ollama_client import ChatMessage, OllamaClient, OllamaError
from team.personas import render_system_prompt
from team.workspace import SharedWorkspace, list_dir_files

log = logging.getLogger(__name__)

DONE_TOKEN = "[[TEAM_DONE]]"


@dataclass
class TurnResult:
    content: str
    declared_done: bool
    files_written: list[str]


class Member:
    def __init__(
        self,
        team: TeamConfig,
        config: MemberConfig,
        runtime: MemberRuntime,
    ):
        self.team = team
        self.config = config
        self.runtime = runtime
        self.client = OllamaClient(
            base_url=runtime.base_url,
            timeout=team.defaults.request_timeout,
            max_retries=team.defaults.max_retries,
            retry_backoff=team.defaults.retry_backoff,
        )
        self._system_prompt = render_system_prompt(team, config)
        self._ready = False

    @property
    def name(self) -> str:
        return self.config.name

    # ----- lifecycle ---------------------------------------------------- #

    def prepare(self, deadline_seconds: int = 180) -> None:
        log.info("waiting for %s ollama at %s", self.name, self.runtime.base_url)
        self.client.wait_ready(deadline_seconds=deadline_seconds)
        log.info("pulling model %s for %s", self.config.model, self.name)
        self.client.ensure_model(
            self.config.model, timeout=self.team.defaults.pull_timeout
        )
        self._ready = True

    # ----- conversation ------------------------------------------------- #

    def _build_messages(
        self,
        transcript: Transcript,
        workspace: SharedWorkspace,
        prompt: str | None,
    ) -> list[ChatMessage]:
        ctx_lines: list[str] = []

        # Shared workspace -------------------------------------------------- #
        files = workspace.list_files()
        if files:
            ctx_lines.append("## Files currently in the shared workspace")
            ctx_lines.extend(f"- {f}" for f in files[:50])
            ctx_lines.append("")
        recent = workspace.recent_changes(limit=10)
        if recent:
            ctx_lines.append("## Most recently changed files")
            ctx_lines.extend(f"- {f}" for f in recent)
            ctx_lines.append("")

        # Private workspace ------------------------------------------------- #
        private_root = self.team.workspace / "members" / self.config.name
        private_files = list_dir_files(private_root, limit=30)
        if private_files:
            ctx_lines.append("## Files in your private workspace (/private)")
            ctx_lines.extend(f"- {f}" for f in private_files)
            ctx_lines.append("")

        # Transcript -------------------------------------------------------- #
        ctx_lines.append("## Conversation so far")
        ctx_lines.append(transcript.render(viewer=self.name) or "(no turns yet)")
        if prompt:
            ctx_lines.append("")
            ctx_lines.append("## Your turn")
            ctx_lines.append(prompt)
        else:
            ctx_lines.append("")
            ctx_lines.append("## Your turn")
            ctx_lines.append(
                "Take the next action that best advances the team goal, "
                "respecting your role and the protocol."
            )
        user_content = "\n".join(ctx_lines)
        return [
            ChatMessage(role="system", content=self._system_prompt),
            ChatMessage(role="user", content=user_content),
        ]

    def take_turn(
        self,
        transcript: Transcript,
        workspace: SharedWorkspace,
        prompt: str | None = None,
        token_callback: Callable[[str], None] | None = None,
    ) -> TurnResult:
        if not self._ready:
            raise RuntimeError(f"member {self.name!r} not prepared")
        defaults = self.team.defaults
        messages = self._build_messages(transcript, workspace, prompt)
        kwargs = dict(
            model=self.config.model,
            messages=messages,
            temperature=resolve_member_setting(self.config, defaults, "temperature"),
            top_p=resolve_member_setting(self.config, defaults, "top_p"),
            num_ctx=resolve_member_setting(self.config, defaults, "context_window"),
        )
        if token_callback is not None:
            chunks: list[str] = []
            for token in self.client.stream_chat(**kwargs):
                token_callback(token)
                chunks.append(token)
            content = "".join(chunks)
            if not content:
                raise OllamaError(f"stream returned no content for @{self.name}")
        else:
            content = self.client.chat(**kwargs)
        declared_done = DONE_TOKEN in content
        writes: list[str] = []
        if self.config.can_write_files:
            writes = [w.path for w in workspace.apply_reply(content)]
        return TurnResult(
            content=content.strip(),
            declared_done=declared_done,
            files_written=writes,
        )
