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
from team.ollama_client import ChatMessage, OllamaClient, OllamaError, OpenAICompatClient, ToolCall
from team.personas import render_system_prompt
from team.mcp.textmode import parse_tool_blocks, parse_tool_body
from team.workspace import SharedWorkspace, list_dir_files

# Optional feature imports — guarded so the modules are only required when enabled.
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from team.memory import AgentMemory
    from team.beliefs import BeliefBoard
    from team.mcp.bus import MemberToolset

log = logging.getLogger(__name__)

DONE_TOKEN = "[[TEAM_DONE]]"

_VALID_CONTEXT_STRATEGIES = {"none", "sliding_window", "truncate", "summarize"}


def _extract_json(text: str) -> "tuple[dict | list | None, str | None]":
    """Try to extract a JSON value from an LLM reply.

    Attempts (in order):
    1. Parse the whole string as JSON.
    2. Extract the first ``json`` fenced code block and parse that.
    3. Extract the first bare fenced code block and parse that.

    Returns ``(parsed, None)`` on success or ``(None, error_message)`` on failure.
    """
    import json
    import re

    stripped = text.strip()
    try:
        return json.loads(stripped), None
    except json.JSONDecodeError:
        pass

    for pattern in (r"```json\s*([\s\S]+?)\s*```", r"```\s*([\s\S]+?)\s*```"):
        m = re.search(pattern, stripped)
        if m:
            try:
                return json.loads(m.group(1).strip()), None
            except json.JSONDecodeError:
                pass

    return None, "no valid JSON found in reply"


def _validate_json_schema(data: "dict | list", schema: dict) -> "str | None":
    """Validate *data* against *schema* using jsonschema (if installed).

    Returns ``None`` if valid or jsonschema is not installed, or an error
    message string if validation fails.
    """
    try:
        import jsonschema  # type: ignore[import]
        jsonschema.validate(data, schema)
        return None
    except ImportError:
        log.debug("jsonschema not installed; skipping schema validation")
        return None
    except Exception as exc:  # jsonschema.ValidationError
        return str(getattr(exc, "message", exc))


@dataclass
class TurnResult:
    content: str
    declared_done: bool
    files_written: list[str]
    prompt_tokens: int = 0
    completion_tokens: int = 0
    json_output: "dict | list | None" = None


class Member:
    def __init__(
        self,
        team: TeamConfig,
        config: MemberConfig,
        runtime: MemberRuntime,
        memory: "AgentMemory | None" = None,
        beliefs: "BeliefBoard | None" = None,
        toolset: "MemberToolset | None" = None,
    ):
        self.team = team
        self.config = config
        self.runtime = runtime
        self.memory = memory
        self.beliefs = beliefs
        self.toolset = toolset

        # Pick the right LLM client based on the effective backend setting.
        backend = resolve_member_setting(config, team.defaults, "backend") or "ollama"
        eff_max_retries: int = resolve_member_setting(config, team.defaults, "max_retries") or team.defaults.max_retries
        eff_retry_backoff: float = resolve_member_setting(config, team.defaults, "retry_backoff") or team.defaults.retry_backoff
        if backend == "openai_compat":
            api_key = resolve_member_setting(config, team.defaults, "api_key")
            self.client: OllamaClient | OpenAICompatClient = OpenAICompatClient(
                base_url=runtime.base_url,
                api_key=api_key,
                timeout=team.defaults.request_timeout,
                max_retries=eff_max_retries,
                retry_backoff=eff_retry_backoff,
            )
        else:
            self.client = OllamaClient(
                base_url=runtime.base_url,
                timeout=team.defaults.request_timeout,
                max_retries=eff_max_retries,
                retry_backoff=eff_retry_backoff,
            )

        # Tools come from the MemberToolset (built-in + external MCP servers),
        # already filtered to this member's enabled patterns by the orchestrator.
        enabled_tools = toolset.tools() if toolset is not None else []

        eff_tool_mode = resolve_member_setting(config, team.defaults, "tool_mode") or "text"
        self._system_prompt = render_system_prompt(
            team,
            config,
            tools=enabled_tools,
            injected_context=self._load_extra_context() or None,
            tool_mode=eff_tool_mode,
        )

        self._ready = False

    def _load_extra_context(self) -> list[str]:
        """Read the member's ``extra_context:`` files, relative to the team YAML dir.

        Member setting overrides defaults.  Each file is truncated at 8192 chars
        (matching context.md handling) and injected into the system prompt.

        Each entry is tried as a relative path first; if no such file exists,
        it falls back to a name registered via the ``team.skills`` entry-point
        group (see :mod:`team.skills`).
        """
        sources = (
            self.config.extra_context
            if self.config.extra_context is not None
            else self.team.defaults.extra_context
        )
        if not sources:
            return []
        base = (
            self.team.source_path.parent
            if self.team.source_path is not None
            else self.team.workspace
        )
        out: list[str] = []
        limit = 8192
        for rel in sources:
            path = (base / rel).expanduser()
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                from team.skills import SkillLoadError, resolve_skill
                try:
                    path = Path(resolve_skill(rel))
                    text = path.read_text(encoding="utf-8", errors="replace")
                except (SkillLoadError, OSError) as exc:
                    log.warning("@%s: could not resolve extra_context %r: %s", self.config.name, rel, exc)
                    continue
            if len(text) > limit:
                text = text[:limit] + f"\n[… {rel} truncated at {limit} chars]"
            if text.strip():
                out.append(text.strip())
        return out

    @property
    def name(self) -> str:
        return self.config.name

    # ----- lifecycle ---------------------------------------------------- #

    def prepare(
        self,
        deadline_seconds: int = 180,
        on_progress: Callable[[dict], None] | None = None,
    ) -> None:
        log.info("waiting for %s ollama at %s", self.name, self.runtime.base_url)
        self.client.wait_ready(deadline_seconds=deadline_seconds)
        log.info("pulling model %s for %s", self.config.model, self.name)

        def _tag_member(event: dict) -> None:
            if on_progress is not None:
                on_progress({**event, "phase": "model", "member": self.name})

        self.client.ensure_model(
            self.config.model,
            timeout=self.team.defaults.pull_timeout,
            on_progress=_tag_member if on_progress is not None else None,
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

        if strategy == "truncate" and budget > 0:
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

        if strategy == "summarize" and budget > 0:
            full = transcript.render(viewer=self.name)
            if len(full) // 4 <= budget:
                return full
            # Reserve 20 % of the budget for the summary; keep 80 % for recent turns.
            recent_budget = int(budget * 0.80)
            n_total = len(transcript.turns)
            n_recent = n_total
            while n_recent > 1 and (
                len(transcript.render(viewer=self.name, max_turns=n_recent)) // 4 > recent_budget
            ):
                n_recent = max(1, n_recent - max(1, n_recent // 8))
            n_old = n_total - n_recent
            recent_text = transcript.render(viewer=self.name, max_turns=n_recent)
            if n_old > 0:
                old_text = transcript.render(viewer=self.name, first_n=n_old)
                summary = self._summarize_transcript(old_text, n_old)
                return (
                    f"[Summary of {n_old} earlier turn(s)]\n{summary}\n\n"
                    f"[Recent conversation ({n_recent} turn(s))]\n{recent_text}"
                )
            return recent_text

        return transcript.render(viewer=self.name)

    def _summarize_transcript(self, old_text: str, turn_count: int) -> str:
        """Ask the LLM to compress *old_text* into a concise bullet-point digest.

        Uses the member's own model at low temperature for factual accuracy.
        Falls back to a plain omission notice if the call fails.
        """
        system = (
            "You are a conversation summarizer. "
            "Produce a concise bullet-point digest of the key decisions, conclusions, "
            "facts, and open questions established in the conversation excerpt below. "
            "Be brief but capture every important detail. Output only the digest."
        )
        user = f"Summarize the following conversation turns:\n\n{old_text}"
        try:
            summary = self.client.chat(
                model=self.config.model,
                messages=[
                    ChatMessage(role="system", content=system),
                    ChatMessage(role="user", content=user),
                ],
                temperature=0.2,
                top_p=None,
                num_ctx=resolve_member_setting(self.config, self.team.defaults, "context_window"),
                keep_alive=resolve_member_setting(self.config, self.team.defaults, "keep_alive"),
            )
            return summary.strip()
        except Exception as exc:
            log.warning(
                "@%s: context summarization failed (%s); omitting %d earlier turn(s)",
                self.name,
                exc,
                turn_count,
            )
            return (
                f"[Summarization failed — {turn_count} earlier turn(s) omitted. Error: {exc}]"
            )

    # ----- conversation ------------------------------------------------- #

    def _load_context_file(self) -> str | None:
        """Read ``context.md`` from the workspace root, if present.

        The file is placed by the user (not members) and injects team-wide
        institutional knowledge into every member's system context on every turn.
        Returns the file content, or ``None`` when the file is absent.
        """
        ctx_path = self.team.workspace / "context.md"
        if not ctx_path.is_file():
            return None
        try:
            text = ctx_path.read_text(encoding="utf-8", errors="replace")
            # Prevent a very large file from overwhelming the context window.
            limit = 8192
            if len(text) > limit:
                text = text[:limit] + f"\n[… context.md truncated at {limit} chars]"
            return text.strip()
        except OSError:
            return None

    def _build_messages(
        self,
        transcript: Transcript,
        workspace: SharedWorkspace,
        prompt: str | None,
    ) -> list[ChatMessage]:
        ctx_lines: list[str] = []

        # Team institutional context (context.md at workspace root) ------------ #
        ctx_file_content = self._load_context_file()
        if ctx_file_content:
            ctx_lines.append("## Team institutional context")
            ctx_lines.append(ctx_file_content)
            ctx_lines.append("")

        # Persistent memory (injected first so the agent has it in mind) ------- #
        if self.memory is not None:
            mem_summary = self.memory.summary_for_prompt(
                n=self.team.memory.inject_recent
            )
            if mem_summary:
                ctx_lines.append(mem_summary)
                ctx_lines.append("")

        # Shared belief board -------------------------------------------------- #
        if self.beliefs is not None:
            belief_summary = self.beliefs.summary_for_prompt(
                limit=self.team.beliefs.inject_limit
            )
            if belief_summary:
                ctx_lines.append(belief_summary)
                ctx_lines.append("")

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

        # Persona reinforcement — re-state role + persona before every reply to
        # prevent smaller models from drifting away from their assigned identity
        # over long conversations.
        if resolve_member_setting(self.config, self.team.defaults, "persona_reinforcement"):
            ctx_lines.append("")
            ctx_lines.append("## Persona reminder")
            ctx_lines.append(
                f"You are @{self.config.name} — {self.config.role}. "
                f"Stay in character:\n{self.config.persona.strip()}"
            )

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
            keep_alive=resolve_member_setting(self.config, self.team.defaults, "keep_alive"),
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
            if not tool_blocks:
                # No tool calls — this is the final reply.
                return content, total_prompt, total_completion

            tools_by_wire = {t.wire_name: t for t in self.toolset.tools()}
            enabled_list = ", ".join(sorted(tools_by_wire)) or "(none)"

            # Execute each tool and collect results.
            result_parts: list[str] = []
            for wire, tool_body in tool_blocks:
                log.info("@%s round %d: invoking tool %s", self.name, round_num, wire)
                if on_tool_call:
                    on_tool_call(self.name, wire, tool_body)
                tool = tools_by_wire.get(wire)
                if tool is None:
                    result = f"ERROR: unknown tool {wire!r}. Enabled tools: {enabled_list}"
                else:
                    args, err = parse_tool_body(tool_body, tool)
                    if err is not None:
                        result = err
                    else:
                        result = self.toolset.call(wire, args, tool_timeout)
                if on_tool_result:
                    on_tool_result(self.name, wire, result)
                result_parts.append(
                    f"Tool `{wire}` result:\n```\n{result}\n```"
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

    def _run_native_agentic_turn(
        self,
        messages: list[ChatMessage],
        workspace: SharedWorkspace,
        on_tool_call: Callable[[str, str, str], None] | None,
        on_tool_result: Callable[[str, str, str], None] | None,
    ) -> tuple[str, int, int]:
        """Agentic loop using native LLM function-calling (Ollama/OpenAI ``tools`` API).

        The model receives JSON Schema tool definitions and responds with
        structured :class:`~team.ollama_client.ToolCall` objects instead of
        fenced text blocks.  Results are passed back via ``tool`` role messages.

        Returns ``(final_content, total_prompt_tokens, total_completion_tokens)``.
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
        # Build tools-API schemas directly from the member's enabled ToolInfo.
        tool_schemas = [
            {
                "type": "function",
                "function": {
                    "name": t.wire_name,
                    "description": t.description,
                    "parameters": t.input_schema,
                },
            }
            for t in self.toolset.tools()
        ]

        total_prompt = 0
        total_completion = 0
        running_messages = list(messages)

        for round_num in range(max_rounds):
            content, tool_calls = self.client.chat_native(
                model=self.config.model,
                messages=running_messages,
                tools=tool_schemas,
                temperature=resolve_member_setting(self.config, self.team.defaults, "temperature"),
                top_p=resolve_member_setting(self.config, self.team.defaults, "top_p"),
                num_ctx=resolve_member_setting(self.config, self.team.defaults, "context_window"),
                keep_alive=resolve_member_setting(self.config, self.team.defaults, "keep_alive"),
            )
            usage = self.client.last_usage
            total_prompt += usage.prompt_tokens if usage else 0
            total_completion += usage.completion_tokens if usage else 0

            if not tool_calls:
                # No tool calls — this is the final reply.
                return content, total_prompt, total_completion

            # Build the assistant message with structured tool_calls.
            asst_msg = ChatMessage(
                role="assistant",
                content=content,
                tool_calls=tool_calls,
            )
            running_messages.append(asst_msg)

            # Execute each tool and collect results as individual tool messages.
            for tc in tool_calls:
                wire = tc.name
                log.info(
                    "@%s round %d: native tool call %s(%s)",
                    self.name, round_num, wire, str(tc.arguments)[:80],
                )
                if on_tool_call:
                    on_tool_call(self.name, wire, str(tc.arguments))
                result = self.toolset.call(wire, tc.arguments or {}, tool_timeout)
                if on_tool_result:
                    on_tool_result(self.name, wire, result)
                # Ollama/OpenAI expect a "tool" role message for each call result.
                running_messages.append(ChatMessage(role="tool", content=result))

        # Exhausted rounds — do one final call with no tools to get a text reply.
        log.warning(
            "@%s exhausted %d native tool rounds; requesting final reply",
            self.name, max_rounds,
        )
        content, _ = self.client.chat_native(
            model=self.config.model,
            messages=running_messages,
            tools=[],
            temperature=resolve_member_setting(self.config, self.team.defaults, "temperature"),
            top_p=resolve_member_setting(self.config, self.team.defaults, "top_p"),
            num_ctx=resolve_member_setting(self.config, self.team.defaults, "context_window"),
            keep_alive=resolve_member_setting(self.config, self.team.defaults, "keep_alive"),
        )
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

        tool_mode = resolve_member_setting(self.config, self.team.defaults, "tool_mode") or "text"
        has_tools = bool(self.toolset is not None and self.toolset.tools())

        if has_tools and tool_mode == "native":
            content, prompt_tokens, completion_tokens = self._run_native_agentic_turn(
                messages,
                workspace,
                on_tool_call,
                on_tool_result,
            )
        elif has_tools:
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

        # F11: structured JSON output — validate and optionally retry.
        json_output = None
        output_format = self.config.output_format
        if output_format == "json":
            output_schema = self.config.output_schema
            retry_messages = list(messages)
            _max_json_retries = 3
            for _attempt in range(_max_json_retries):
                parsed, error = _extract_json(content)
                if parsed is not None and output_schema:
                    schema_error = _validate_json_schema(parsed, output_schema)
                    if schema_error:
                        error = f"schema validation failed: {schema_error}"
                        parsed = None
                if parsed is not None:
                    json_output = parsed
                    break
                log.warning(
                    "@%s JSON attempt %d/%d failed: %s",
                    self.name, _attempt + 1, _max_json_retries, error,
                )
                if _attempt < _max_json_retries - 1:
                    retry_messages = retry_messages + [
                        ChatMessage(role="assistant", content=content),
                        ChatMessage(
                            role="user",
                            content=(
                                f"Your reply was not valid JSON. Error: {error}. "
                                "Please respond with ONLY a valid JSON value — "
                                "no prose, no markdown fences, no explanation."
                            ),
                        ),
                    ]
                    content = self._call_llm(retry_messages, None)
                    extra_usage = self.client.last_usage
                    if extra_usage:
                        prompt_tokens += extra_usage.prompt_tokens
                        completion_tokens += extra_usage.completion_tokens
            else:
                log.warning(
                    "@%s: could not produce valid JSON after %d attempts",
                    self.name, _max_json_retries,
                )

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
            json_output=json_output,
        )

