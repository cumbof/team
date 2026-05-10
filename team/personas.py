"""System-prompt rendering for team members.

Each member receives a system prompt that:

* States its identity (name + role) and persona.
* Describes the team goal and the other members it can address.
* Documents the small "protocol" the orchestrator understands so the
  member can address peers, write files, and signal completion.

Keeping the protocol tiny and explicit makes models far more reliable at
following it (especially smaller models).  See :func:`render_system_prompt`.
"""

from __future__ import annotations

from textwrap import dedent

from team.config import MemberConfig, TeamConfig

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


def render_system_prompt(team: TeamConfig, member: MemberConfig) -> str:
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
    # Append extra instructions last so they can override or extend the
    # boilerplate sections above without modifying the shared PROTOCOL constant.
    if member.extra_system:
        parts.extend(["", "## Additional instructions", member.extra_system.strip()])
    return "\n".join(parts)
