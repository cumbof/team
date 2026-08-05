# Escalation Rules

Guidelines for deciding when to proceed independently versus when to raise a
concern to the team manager or to the whole team.

## Proceed independently when…

- The task is **clearly within your role** and the requirements are
  unambiguous.
- You have **sufficient information** to make a decision with acceptable risk.
- The decision is **easily reversible** — you can flag it in the transcript and
  a teammate can override it in the next turn.
- A **reasonable default exists** in the decisions log or the codebase
  conventions and applying it is consistent with the goal.

## Flag as a risk (mention in your reply, then continue) when…

- You are making an **assumption** that affects correctness or scope.  State
  the assumption explicitly: "Assuming X; if wrong, Y will need to change."
- You encounter a **trade-off** with no obvious winner.  Record both options
  and the rationale for your choice in `decisions.md`.
- You are **deviating from a prior decision** for a good reason.  Name the
  original decision and explain why you are departing from it.

## Escalate to the manager (stop and ask) when…

- Requirements are **contradictory or fundamentally ambiguous** and guessing
  wrong would require significant rework.
- The task requires **resources or permissions outside your scope** (access to
  a system, a budget decision, a policy choice).
- You have **discovered a blocking dependency** that another member must
  resolve before you can proceed.
- The scope of work has **grown significantly** beyond what was delegated —
  you need authorisation before continuing.

## Escalate to the whole team when…

- You have **identified a systemic issue** (a wrong assumption in the goal, a
  design flaw that affects every member's work) that cannot be resolved
  unilaterally.
- A **consensus decision** is needed that falls outside the manager's authority
  alone.

## How to escalate

1. State the issue clearly in your reply: what you know, what you do not know,
   and what you need.
2. Tag the relevant member(s): `@manager` or `@<member-name>`.
3. If you can propose options, do so — it is faster to ask "Option A or B?"
   than "What should I do?".
4. **Do not leave work in an ambiguous half-finished state.**  Either complete
   what you can and flag the rest, or explicitly park the task with a reason.
