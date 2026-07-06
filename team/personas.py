"""System-prompt rendering for team members.

Each member receives a system prompt that:

* States its identity (name + role) and persona.
* Describes the team goal and the other members it can address.
* Documents the small "protocol" the orchestrator understands so the
  member can address peers, write files, and signal completion.
* When the member has tools enabled, appends a tool-use protocol section.

Keeping the protocol tiny and explicit makes models far more reliable at
following it (especially smaller models).  See :func:`render_system_prompt`.
"""

from __future__ import annotations

from textwrap import dedent
from typing import TYPE_CHECKING

from team.config import MemberConfig, TeamConfig

if TYPE_CHECKING:
    from team.mcp.bus import ToolInfo

PROTOCOL = dedent(
    """\
    ## Collaboration protocol

    You are part of a TEAM.  At every turn, ONE member speaks; the others read
    the transcript and may speak on a later turn.  Always respect the protocol
    below — the orchestrator parses your reply mechanically.

    1. Begin your reply with one short sentence summarising your contribution.
    2. To explicitly address another member, prefix the relevant section with
       a line of the form `@<member-name>:` (lowercase, exact).  You may omit
       this if you are addressing the whole team.
    3. To create or overwrite a file in the SHARED workspace, use a fenced
       block with an `file:` info-string, e.g.

           ```file:report/intro.md
           # Introduction
           ...
           ```

       The orchestrator will atomically write the contents to that path under
       the shared workspace.  Paths must be relative and may not escape the
       workspace.  Any number of file blocks per reply is allowed.
    4. You also have a PRIVATE workspace (`/private` inside your container)
       for personal scratch files — drafts, notes, intermediate results — that
       are not shared with the team.  Files you create there persist between
       turns and are listed at the top of each of your turn prompts.
    5. To read an existing file, ask the team in plain text and a member with
       file access (or the orchestrator-provided context) will surface it.
       Recently written/changed files are listed at the top of every turn.
    6. When YOU believe the team's goal is fully achieved, end your reply
       with a line containing exactly: `[[TEAM_DONE]]`.
       Use this sparingly — only when the deliverables are complete and you
       have verified them.

    ## House rules
    * Be concrete.  Prefer artifacts (files, tables, code) over prose.
    * Disagree explicitly when warranted; reference earlier turns by member
      name.  Avoid sycophancy.
    * Stay strictly within your role; defer outside-scope work to the
      appropriate teammate.
    """
)

def _render_tool_section(tools: "list[ToolInfo]") -> str:
    from team.mcp.textmode import render_tool_protocol

    code_enabled = any(t.server == "code" for t in tools)
    return render_tool_protocol(tools, code_enabled=code_enabled)


def render_system_prompt(
    team: TeamConfig,
    member: MemberConfig,
    tools: "list[ToolInfo] | None" = None,
    injected_context: list[str] | None = None,
) -> str:
    """Render the complete system prompt for *member*.

    Parameters
    ----------
    team:
        The full team configuration.
    member:
        The member whose system prompt is being rendered.
    tools:
        The :class:`~team.mcp.bus.ToolInfo` objects this member has enabled.
        When non-empty a text-mode tool-use protocol section is appended.
        Pass ``None`` or ``[]`` to omit the section.
    injected_context:
        Strings loaded from ``extra_context:`` files, appended verbatim after
        the collaboration protocol so the member has this background knowledge
        in every turn without needing to call a tool.
    """
    # Build the teammate list excluding the member itself.
    teammates = [
        f"- @{m.name} — {m.role}"
        for m in team.members
        if m.name != member.name
    ]
    teammate_block = "\n".join(teammates) if teammates else "- (you are working alone)"

    parts = [
        f"# You are @{member.name} — {member.role}",
        "",
        "## Persona",
        member.persona.strip(),
        "",
        "## Team goal",
        team.goal.strip(),
        "",
        "## Your teammates",
        teammate_block,
        "",
        PROTOCOL.strip(),
    ]
    # Inject skill context (Markdown skills or INJECT_INTO_CONTEXT) after the
    # collaboration protocol so the LLM has background knowledge in every turn.
    if injected_context:
        for ctx in injected_context:
            if ctx.strip():
                parts.extend(["", ctx.strip()])
    # Append extra instructions last so they can override or extend the
    # boilerplate sections above without modifying the shared PROTOCOL constant.
    if member.extra_system:
        parts.extend(["", "## Additional instructions", member.extra_system.strip()])
    if getattr(member, "output_format", None) == "json":
        schema_hint = ""
        if getattr(member, "output_schema", None):
            import json as _json
            schema_hint = (
                "\n\nRequired JSON Schema:\n"
                f"```json\n{_json.dumps(member.output_schema, indent=2)}\n```"
            )
        parts.extend([
            "",
            "## Output format",
            (
                "**You must respond with valid JSON only.**  "
                "Do not include any prose, markdown formatting, or explanation "
                "outside the JSON structure.  Do not wrap the JSON in a code "
                "fence.  Your entire reply must be parseable by `json.loads()`."
                + schema_hint
            ),
        ])
    if tools:
        parts.extend(["", _render_tool_section(tools)])
    return "\n".join(parts)
