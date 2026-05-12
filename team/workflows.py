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
# Parallel review
# --------------------------------------------------------------------------- #


def parallel_review(orch: "Orchestrator") -> None:
    """Fan-out review: N reviewers critique a deliverable simultaneously, then a
    synthesizer consolidates their feedback, and the producer revises.

    Unlike :func:`review_loop`, which uses a single serial reviewer, this
    workflow dispatches **all** reviewers in parallel (using a thread pool) so
    the total review time is bounded by the slowest reviewer rather than the
    sum of all reviewers.  Their findings are concatenated in a consistent
    order and handed to a *synthesizer* who produces a single actionable
    verdict before the producer revises.

    Required workflow options:

    * ``producer``    — member who creates and revises the deliverable.
    * ``reviewers``   — list of 2+ member names who review in parallel.
    * ``synthesizer`` — member who consolidates the parallel reviews (may be
      the same as ``producer``).

    Optional:

    * ``approve_token`` — token the synthesizer emits to signal approval
      (default: ``APPROVED``).

    Example YAML::

        workflow:
          type: parallel_review
          max_rounds: 4
          producer: writer
          reviewers: [critic_a, critic_b, critic_c]
          synthesizer: editor
          approve_token: APPROVED

    Thread-safety note
    ------------------
    Each reviewer calls :meth:`~team.member.Member.take_turn` concurrently.
    ``take_turn`` reads the shared transcript (read-only during the parallel
    window) and issues independent LLM calls, so there are no data races.
    Reviewer members should not use file-writing tools during parallel turns
    to avoid concurrent workspace writes.
    """
    import concurrent.futures

    opts = orch.team.workflow.options
    producer_name: str = opts["producer"]
    reviewer_names: list[str] = opts["reviewers"]
    synthesizer_name: str = opts.get("synthesizer", producer_name)
    approve_token: str = opts.get("approve_token", "APPROVED")
    max_rounds = orch.team.workflow.max_rounds

    if not reviewer_names or len(reviewer_names) < 2:
        raise ValueError("parallel_review requires at least 2 reviewers")

    # ---------- Phase 1: Producer creates the first deliverable ------------ #
    pres = orch.run_turn(
        producer_name,
        prompt=(
            "Produce the FIRST complete draft of the deliverable required "
            "by the team goal.  Use file blocks for any artifacts."
        ),
    )
    if pres.declared_done:
        return

    for revision in range(1, max_rounds + 1):
        log.info(
            "parallel_review revision %d/%d — dispatching %d reviewers",
            revision, max_rounds, len(reviewer_names),
        )

        # ---------- Phase 2: All reviewers run in parallel ---------------- #
        review_results: dict[str, "TurnResult"] = {}  # type: ignore[type-arg]
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(reviewer_names), thread_name_prefix="reviewer"
        ) as pool:
            futures = {
                pool.submit(
                    orch.members[r_name].take_turn,
                    orch.transcript,
                    orch.workspace,
                    (
                        f"PARALLEL REVIEW — revision #{revision} from @{producer_name}.\n"
                        "Provide specific, actionable feedback on the deliverable.  "
                        "Note what is good, what is problematic, and what concrete "
                        "changes would fix the problems.  Be concise and direct."
                    ),
                ): r_name
                for r_name in reviewer_names
            }
            for future in concurrent.futures.as_completed(futures):
                r_name = futures[future]
                try:
                    review_results[r_name] = future.result()
                except Exception as exc:  # noqa: BLE001
                    log.error("reviewer @%s raised an exception: %s", r_name, exc)
                    from team.member import TurnResult
                    review_results[r_name] = TurnResult(
                        content=f"[ERROR: reviewer @{r_name} failed: {exc}]",
                        declared_done=False,
                        files_written=[],
                    )

        # Append all review turns to the transcript in a deterministic order.
        for r_name in reviewer_names:
            result = review_results[r_name]
            member = orch.members[r_name]
            orch.transcript.append(
                speaker=r_name,
                role=member.config.role,
                content=result.content,
                files_written=result.files_written,
            )
            log.info("parallel_review: @%s review appended", r_name)

        # Check if any reviewer declared done (unusual but possible).
        if any(review_results[r].declared_done for r in reviewer_names):
            log.info("a reviewer declared TEAM_DONE — parallel_review ending")
            return

        # ---------- Phase 3: Synthesizer consolidates the reviews ---------- #
        reviewer_list = ", ".join(f"@{r}" for r in reviewer_names)
        sres = orch.run_turn(
            synthesizer_name,
            prompt=(
                f"SYNTHESIS — {len(reviewer_names)} parallel reviews from "
                f"{reviewer_list} have been appended to the transcript.\n\n"
                "Read all of them and produce ONE consolidated, prioritised "
                "list of required changes.  If — and only if — all reviewers "
                "agree the deliverable is ready with no further changes, end "
                f"your synthesis with the single token `{approve_token}`.  "
                "Otherwise list the must-fix items in priority order."
            ),
        )
        if sres.declared_done or approve_token in sres.content:
            log.info("synthesizer approved after revision %d", revision)
            orch.run_turn(
                producer_name,
                prompt=(
                    "All reviewers approved the deliverable.  Finalise any "
                    "remaining files and end with `[[TEAM_DONE]]`."
                ),
            )
            return

        # ---------- Phase 4: Producer revises ------------------------------ #
        pres = orch.run_turn(
            producer_name,
            prompt=(
                "Address ALL items raised in the synthesizer's verdict above.  "
                "Update the affected files using file blocks.  Describe what you "
                "changed and why."
            ),
        )
        if pres.declared_done:
            return

        if orch._on_round_end:
            orch._on_round_end(revision - 1)

    log.info("parallel_review hit max_rounds=%d", max_rounds)


# --------------------------------------------------------------------------- #
# Parallel rounds
# --------------------------------------------------------------------------- #


def parallel_rounds(orch: "Orchestrator") -> None:
    """All members speak simultaneously in every round.

    Each round dispatches one LLM call per member concurrently.  Members
    receive the same transcript snapshot (the state at the start of the round),
    so they cannot reference each other's output from the *current* round —
    only from previous rounds.  After all threads complete, turns are appended
    to the transcript in member-declaration order.

    This is ideal for independent-expert or brainstorming scenarios where you
    want parallel depth rather than sequential dialogue.

    The workflow ends when any member emits ``[[TEAM_DONE]]`` (checked after
    each round) or when ``max_rounds`` is exhausted.

    Example YAML::

        workflow:
          type: parallel
          max_rounds: 4

    Thread-safety note
    ------------------
    ``member.take_turn()`` is called concurrently from multiple threads.
    The shared transcript is read-only during each parallel window; writes
    happen sequentially after the window closes.  Concurrent writes to the
    *same workspace file path* are a race condition — ensure parallel members
    work on disjoint output paths.
    """
    member_names = list(orch.members.keys())
    max_rounds = orch.team.workflow.max_rounds

    for round_idx in range(max_rounds):
        log.info(
            "parallel round %d/%d — dispatching %d members",
            round_idx + 1, max_rounds, len(member_names),
        )
        results = orch.run_parallel_round(member_names)
        if any(result.declared_done for _, result in results):
            log.info("a member declared TEAM_DONE — parallel workflow ending")
            return
        if orch._on_round_end:
            orch._on_round_end(round_idx)

    log.info("parallel workflow completed %d round(s)", max_rounds)


WORKFLOWS = {
    "round_robin": round_robin,
    "manager": manager_driven,
    "review_loop": review_loop,
    "sequential_chain": sequential_chain,
    "debate": debate,
    "parallel_review": parallel_review,
    "parallel": parallel_rounds,
}


def get_workflow(name: str):
    if name not in WORKFLOWS:
        raise KeyError(f"unknown workflow {name!r}")
    return WORKFLOWS[name]
