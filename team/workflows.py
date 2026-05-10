"""Workflow strategies — pluggable turn schedulers.

Each workflow is a callable
``run(orchestrator) -> None`` that drives the conversation until the workflow
decides to stop (typically when ``max_rounds`` is exhausted or a member
declares ``[[TEAM_DONE]]``).

We expose three reference workflows:

* :func:`round_robin` — every member speaks in declaration order, repeated
  ``max_rounds`` times.
* :func:`manager_driven` — a designated *manager* member is asked, before
  every turn, to nominate the next speaker (or to declare done).
* :func:`review_loop` — a *producer* / *reviewer* pair iterate; the loop
  ends when the reviewer outputs ``APPROVED`` (or ``[[TEAM_DONE]]``) or
  ``max_rounds`` revisions are reached.

Workflows interact with the orchestrator through a small surface
(:meth:`Orchestrator.run_turn`, :attr:`Orchestrator.members`,
:attr:`Orchestrator.transcript`) so they remain easy to add.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from team.orchestrator import Orchestrator

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Round-robin
# --------------------------------------------------------------------------- #


def round_robin(orch: "Orchestrator") -> None:
    members = list(orch.members.values())
    max_rounds = orch.team.workflow.max_rounds
    for round_idx in range(max_rounds):
        log.info("round %d/%d", round_idx + 1, max_rounds)
        for m in members:
            res = orch.run_turn(m.name)
            if res.declared_done:
                log.info("member %s declared TEAM_DONE", m.name)
                return
        if orch._on_round_end:
            orch._on_round_end(round_idx)


# --------------------------------------------------------------------------- #
# Manager-driven
# --------------------------------------------------------------------------- #


# Matches "NEXT: @alice" or "NEXT: alice" (case-insensitive).
# The manager uses this marker to nominate the next speaker.
_NEXT_RE = re.compile(r"NEXT:\s*@?([a-z0-9_-]+)", re.IGNORECASE)


def manager_driven(orch: "Orchestrator") -> None:
    manager_name = orch.team.workflow.options.get("manager")
    manager = orch.members[manager_name]
    max_rounds = orch.team.workflow.max_rounds

    bootstrap_prompt = (
        "You are the manager.  Open the work: clearly restate the goal in "
        "your own words, lay out a concrete plan, then on the LAST line of "
        "your reply write `NEXT: @<member>` to nominate who speaks next."
    )
    res = orch.run_turn(manager.name, prompt=bootstrap_prompt)
    if res.declared_done:
        return
    next_speaker = _parse_next(res.content, orch) or _next_default(orch, manager.name)

    for turn in range(max_rounds):
        res = orch.run_turn(next_speaker)
        if res.declared_done:
            return
        # Manager always decides who is next (unless it just spoke).
        if next_speaker == manager.name:
            chosen = _parse_next(res.content, orch)
        else:
            mres = orch.run_turn(
                manager.name,
                prompt=(
                    "Review the latest turn.  Briefly evaluate progress, "
                    "give pointed direction, and on the LAST line of your "
                    "reply write `NEXT: @<member>` (or `NEXT: @"
                    f"{manager.name}` to take the floor yourself, or "
                    "`[[TEAM_DONE]]` if the goal is fully achieved)."
                ),
            )
            if mres.declared_done:
                return
            chosen = _parse_next(mres.content, orch)
        next_speaker = chosen or _next_default(orch, next_speaker)
        if orch._on_round_end:
            orch._on_round_end(turn)


def _parse_next(content: str, orch: "Orchestrator") -> str | None:
    for m in _NEXT_RE.finditer(content):
        candidate = m.group(1).lower()
        if candidate in orch.members:
            return candidate
    return None


def _next_default(orch: "Orchestrator", current: str) -> str:
    # Circular round-robin fallback: used when the manager's reply contains no
    # parseable NEXT: marker so the conversation doesn't stall.
    names = list(orch.members.keys())
    i = names.index(current)
    return names[(i + 1) % len(names)]


# --------------------------------------------------------------------------- #
# Review loop
# --------------------------------------------------------------------------- #


def review_loop(orch: "Orchestrator") -> None:
    opts = orch.team.workflow.options
    producer = orch.members[opts["producer"]]
    reviewer = orch.members[opts["reviewer"]]
    max_rounds = orch.team.workflow.max_rounds
    approve_token = opts.get("approve_token", "APPROVED")

    # Initial production
    pres = orch.run_turn(
        producer.name,
        prompt=(
            "Produce the FIRST complete draft of the deliverable required "
            "by the team goal.  Use file blocks for any artifacts."
        ),
    )
    if pres.declared_done:
        return

    for revision in range(1, max_rounds + 1):
        rres = orch.run_turn(
            reviewer.name,
            prompt=(
                f"Review revision #{revision} from @{producer.name}.  "
                "Provide concrete, line-level feedback.  If — and only if — "
                "the deliverable fully meets the team goal with no further "
                f"changes needed, end your reply with the single token "
                f"`{approve_token}`."
            ),
        )
        if rres.declared_done or approve_token in rres.content:
            log.info("reviewer approved after %d revision(s)", revision)
            # Let producer ack and finalize.
            orch.run_turn(
                producer.name,
                prompt=(
                    "The reviewer approved.  Finalise the deliverable, write "
                    "any remaining files, and end with `[[TEAM_DONE]]`."
                ),
            )
            return

        pres = orch.run_turn(
            producer.name,
            prompt=(
                f"Address ALL of @{reviewer.name}'s feedback above.  Update "
                "the affected files in place using file blocks.  Be explicit "
                "about what you changed and why."
            ),
        )
        if pres.declared_done:
            return
        if orch._on_round_end:
            orch._on_round_end(revision - 1)

    log.info("review loop hit max_rounds=%d", max_rounds)


# --------------------------------------------------------------------------- #
# Sequential chain
# --------------------------------------------------------------------------- #

_DEFAULT_CHAIN_TEMPLATE = (
    "@{prev_speaker} just produced the following output.  "
    "Process it according to your role and the team goal:\n\n{prev_content}"
)


def sequential_chain(orch: "Orchestrator") -> None:
    """Pipeline workflow: each member's reply becomes the explicit prompt for
    the next member in declaration order.

    The chain wraps around ``max_rounds`` times.  A configurable
    ``prompt_template`` controls how the previous member's output is
    presented; it receives two named placeholders: ``{prev_speaker}`` and
    ``{prev_content}``.

    Example YAML::

        workflow:
          type: sequential_chain
          max_rounds: 2
          prompt_template: |
            @{prev_speaker} produced the draft below.  Improve it:

            {prev_content}
    """
    opts = orch.team.workflow.options
    members = list(orch.members.values())
    max_rounds = orch.team.workflow.max_rounds
    prompt_template: str = opts.get("prompt_template", _DEFAULT_CHAIN_TEMPLATE)

    # `prev_content` and `prev_speaker` are maintained *outside* the round loop
    # so the first member of round N+1 receives the last member of round N's
    # reply — creating a continuous pipeline that wraps across rounds.
    prev_content: str | None = None
    prev_speaker: str | None = None

    for round_idx in range(max_rounds):
        log.info("sequential chain round %d/%d", round_idx + 1, max_rounds)
        for m in members:
            if prev_content is not None and prev_speaker is not None:
                prompt = prompt_template.format(
                    prev_speaker=prev_speaker,
                    prev_content=prev_content,
                )
            else:
                prompt = None  # first member of the first round gets the default
            res = orch.run_turn(m.name, prompt=prompt)
            if res.declared_done:
                log.info("member %s declared TEAM_DONE", m.name)
                return
            prev_content = res.content
            prev_speaker = m.name
        if orch._on_round_end:
            orch._on_round_end(round_idx)

    log.info("sequential chain completed %d round(s)", max_rounds)


# --------------------------------------------------------------------------- #
# Debate
# --------------------------------------------------------------------------- #


def debate(orch: "Orchestrator") -> None:
    """Two members argue opposing sides; a judge delivers the final verdict.

    Required workflow options:

    * ``pro``   — name of the member arguing *in favour* of the proposition
    * ``con``   — name of the member arguing *against* the proposition
    * ``judge`` — name of the member who delivers the impartial verdict

    Optional:

    * ``topic`` — explicit proposition text (defaults to ``team.goal``)

    Example YAML::

        workflow:
          type: debate
          max_rounds: 3          # rebuttal rounds per side (default 6 → 3 per side)
          pro: advocate
          con: critic
          judge: arbiter
          topic: "Open-source LLMs will surpass proprietary ones within 5 years"
    """
    opts = orch.team.workflow.options
    pro_name = opts["pro"]
    con_name = opts["con"]
    judge_name = opts["judge"]
    max_rounds = orch.team.workflow.max_rounds
    topic = opts.get("topic") or orch.team.goal.strip()

    # Opening statements ---------------------------------------------------- #
    res = orch.run_turn(
        pro_name,
        prompt=(
            f"Deliver your OPENING ARGUMENT in favour of the proposition:\n\n"
            f"> {topic}\n\n"
            "Structure your case with distinct, well-reasoned points. "
            "Be clear, logical, and evidence-driven. Do not simply restate the proposition."
        ),
    )
    if res.declared_done:
        return

    res = orch.run_turn(
        con_name,
        prompt=(
            f"Deliver your OPENING ARGUMENT against the proposition:\n\n"
            f"> {topic}\n\n"
            f"Structure your case with distinct points. Directly rebut at least one "
            f"claim made by @{pro_name} in their opening."
        ),
    )
    if res.declared_done:
        return

    # Rebuttal rounds ------------------------------------------------------- #
    for round_idx in range(max_rounds):
        log.info("debate rebuttal round %d/%d", round_idx + 1, max_rounds)
        res = orch.run_turn(
            pro_name,
            prompt=(
                f"REBUTTAL round {round_idx + 1}/{max_rounds} (PRO).\n"
                f"Address @{con_name}'s most recent argument specifically. "
                "Reinforce your strongest points with new reasoning or evidence. "
                "Be concise and stay on topic."
            ),
        )
        if res.declared_done:
            return

        res = orch.run_turn(
            con_name,
            prompt=(
                f"REBUTTAL round {round_idx + 1}/{max_rounds} (CON).\n"
                f"Address @{pro_name}'s most recent argument specifically. "
                "Reinforce your strongest points with new reasoning or evidence. "
                "Be concise and stay on topic."
            ),
        )
        if res.declared_done:
            return

        if orch._on_round_end:
            orch._on_round_end(round_idx)

    # Closing statements ---------------------------------------------------- #
    res = orch.run_turn(
        pro_name,
        prompt=(
            "Deliver your CLOSING STATEMENT. Summarise your three strongest arguments, "
            "explain why they were not adequately rebutted, and state your conclusion clearly."
        ),
    )
    if res.declared_done:
        return

    res = orch.run_turn(
        con_name,
        prompt=(
            "Deliver your CLOSING STATEMENT. Summarise your three strongest arguments, "
            "explain why they were not adequately rebutted, and state your conclusion clearly."
        ),
    )
    if res.declared_done:
        return

    # Judge's verdict ------------------------------------------------------- #
    orch.run_turn(
        judge_name,
        prompt=(
            f"You are the impartial judge of this debate on the proposition:\n\n"
            f"> {topic}\n\n"
            "Review the full transcript. Evaluate both sides on: "
            "(1) logical coherence, (2) quality of evidence, "
            "(3) effectiveness of rebuttals, (4) clarity of communication.\n\n"
            "Name the stronger side and explain your reasoning in detail. "
            "End your verdict with `[[TEAM_DONE]]`."
        ),
    )


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #


WORKFLOWS = {
    "round_robin": round_robin,
    "manager": manager_driven,
    "review_loop": review_loop,
    "sequential_chain": sequential_chain,
    "debate": debate,
}


def get_workflow(name: str):
    if name not in WORKFLOWS:
        raise KeyError(f"unknown workflow {name!r}")
    return WORKFLOWS[name]
