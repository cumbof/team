"""A live :class:`Member` couples a config with its container runtime and
Ollama client, and knows how to take a turn given the current transcript.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from team.bus import Transcript
from team.config import MemberConfig, TeamConfig, resolve_member_setting
from team.container import MemberRuntime
from team.ollama_client import ChatMessage, OllamaClient, OllamaError, OpenAICompatClient
from team.personas import render_system_prompt
from team.tools import TOOLS, execute_tool, parse_tool_blocks
from team.workspace import SharedWorkspace, list_dir_files

log = logging.getLogger(__name__)

DONE_TOKEN = "[[TEAM_DONE]]"

_VALID_CONTEXT_STRATEGIES = {"none", "sliding_window", "truncate", "summarize"}


@dataclass
class TurnResult:
    content: str
    declared_done: bool
    files_written: list[str]
    prompt_tokens: int = 0
    completion_tokens: int = 0


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

        # Pick the right LLM client based on the effective backend setting.
        backend = resolve_member_setting(config, team.defaults, "backend") or "ollama"
        if backend == "openai_compat":
            api_key = resolve_member_setting(config, team.defaults, "api_key")
            self.client: OllamaClient | OpenAICompatClient = OpenAICompatClient(
                base_url=runtime.base_url,
                api_key=api_key,
                timeout=team.defaults.request_timeout,
                max_retries=team.defaults.max_retries,
                retry_backoff=team.defaults.retry_backoff,
            )
        else:
            self.client = OllamaClient(
                base_url=runtime.base_url,
                timeout=team.defaults.request_timeout,
                max_retries=team.defaults.max_retries,
                retry_backoff=team.defaults.retry_backoff,
            )

        # Resolve the effective tool list (member override → defaults).
        raw_tools = config.tools if config.tools is not None else team.defaults.tools
        self._enabled_tools: list[str] = [t for t in (raw_tools or []) if t in TOOLS]
        unknown = [t for t in (raw_tools or []) if t not in TOOLS]
        if unknown:
            log.warning("member %s: unknown tools ignored: %s", config.name, unknown)

        self._system_prompt = render_system_prompt(team, config, enabled_tools=self._enabled_tools)
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

    # ----- context management ------------------------------------------- #

    def _apply_context_strategy(self, transcript: Transcript) -> str:
        """Render the transcript, applying the configured context strategy."""
        strategy = (
            resolve_member_setting(self.config, self.team.defaults, "context_strategy")
            or "none"
        )
        budget = (
            resolve_member_setting(self.config, self.team.defaults, "context_budget")
            or 0
        )

        if strategy == "sliding_window" and budget > 0:
            return transcript.render(viewer=self.name, max_turns=budget)

        if strategy in ("truncate", "summarize") and budget > 0:
            # Estimate tokens at ~4 chars/token; binary-search for the max
            # number of turns that fits within the budget.
            full = transcript.render(viewer=self.name)
            if len(full) // 4 <= budget:
                return full
            # Walk down from the full count until we fit.
            n = len(transcript.turns)
            while n > 1 and len(transcript.render(viewer=self.name, max_turns=n)) // 4 > budget:
                n = max(1, n - max(1, n // 8))
            trimmed = transcript.render(viewer=self.name, max_turns=n)
            omitted = len(transcript.turns) - n
            prefix = (
                f"[Context note: {omitted} earlier turn(s) were omitted to stay within "
                f"the ~{budget}-token context budget. Key decisions from omitted turns "
                f"may have been lost — use your best judgement.]\n\n"
            )
            return prefix + trimmed

        return transcript.render(viewer=self.name)

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
        ctx_lines.append(self._apply_context_strategy(transcript) or "(no turns yet)")
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

    def _call_llm(
        self,
        messages: list[ChatMessage],
        token_callback: Callable[[str], None] | None,
    ) -> str:
        """Issue a single LLM call and return the response text."""
        kwargs = dict(
            model=self.config.model,
            messages=messages,
            temperature=resolve_member_setting(self.config, self.team.defaults, "temperature"),
            top_p=resolve_member_setting(self.config, self.team.defaults, "top_p"),
            num_ctx=resolve_member_setting(self.config, self.team.defaults, "context_window"),
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
        return content

    def _run_agentic_turn(
        self,
        messages: list[ChatMessage],
        workspace: SharedWorkspace,
        token_callback: Callable[[str], None] | None,
        on_tool_call: Callable[[str, str, str], None] | None,
        on_tool_result: Callable[[str, str, str], None] | None,
    ) -> tuple[str, int, int]:
        """Run the agentic loop: LLM → tools → LLM → ... until no tool blocks.

        Returns ``(final_content, total_prompt_tokens, total_completion_tokens)``.
        The loop runs at most ``max_tool_rounds`` iterations.
        """
        max_rounds = (
            self.config.max_tool_rounds
            if self.config.max_tool_rounds is not None
            else self.team.defaults.max_tool_rounds
        )
        tool_timeout = (
            self.config.tool_timeout
            if self.config.tool_timeout is not None
            else self.team.defaults.tool_timeout
        )
        workspace_path = workspace.shared_dir

        total_prompt = 0
        total_completion = 0
        running_messages = list(messages)

        for round_num in range(max_rounds):
            content = self._call_llm(running_messages, token_callback if round_num == 0 else None)
            usage = self.client.last_usage
            total_prompt += usage.prompt_tokens if usage else 0
            total_completion += usage.completion_tokens if usage else 0

            # Check for tool invocations in the reply.
            tool_blocks = parse_tool_blocks(content)
            enabled = set(self._enabled_tools)
            active_blocks = [(n, b) for n, b in tool_blocks if n in enabled]

            if not active_blocks:
                # No tool calls — this is the final reply.
                return content, total_prompt, total_completion

            # Execute each tool and collect results.
            result_parts: list[str] = []
            for tool_name, tool_body in active_blocks:
                log.info("@%s round %d: invoking tool %s", self.name, round_num, tool_name)
                if on_tool_call:
                    on_tool_call(self.name, tool_name, tool_body)
                result = execute_tool(
                    tool_name,
                    tool_body,
                    workspace_path=workspace_path,
                    timeout=tool_timeout,
                )
                if on_tool_result:
                    on_tool_result(self.name, tool_name, result)
                result_parts.append(
                    f"Tool `{tool_name}` result:\n```\n{result}\n```"
                )

            # Inject the assistant's reply and the tool results back as messages.
            running_messages.append(ChatMessage(role="assistant", content=content))
            running_messages.append(
                ChatMessage(role="user", content="\n\n".join(result_parts))
            )

        # Exhausted tool rounds — do one final LLM call with no streaming.
        log.warning("@%s exhausted %d tool rounds; requesting final reply", self.name, max_rounds)
        content = self._call_llm(running_messages, None)
        usage = self.client.last_usage
        total_prompt += usage.prompt_tokens if usage else 0
        total_completion += usage.completion_tokens if usage else 0
        return content, total_prompt, total_completion

    def take_turn(
        self,
        transcript: Transcript,
        workspace: SharedWorkspace,
        prompt: str | None = None,
        token_callback: Callable[[str], None] | None = None,
        on_tool_call: Callable[[str, str, str], None] | None = None,
        on_tool_result: Callable[[str, str, str], None] | None = None,
    ) -> TurnResult:
        if not self._ready:
            raise RuntimeError(f"member {self.name!r} not prepared")

        messages = self._build_messages(transcript, workspace, prompt)

        if self._enabled_tools:
            content, prompt_tokens, completion_tokens = self._run_agentic_turn(
                messages,
                workspace,
                token_callback,
                on_tool_call,
                on_tool_result,
            )
        else:
            content = self._call_llm(messages, token_callback)
            usage = self.client.last_usage
            prompt_tokens = usage.prompt_tokens if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0

        declared_done = DONE_TOKEN in content

        # F7: route file:private/... blocks to the member's private directory.
        writes: list[str] = []
        if self.config.can_write_files:
            private_root = self.team.workspace / "members" / self.config.name
            writes = [w.path for w in workspace.apply_reply(content, private_root=private_root)]

        return TurnResult(
            content=content.strip(),
            declared_done=declared_done,
            files_written=writes,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

