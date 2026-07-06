"""``beliefs`` server — assert_belief, contest_belief, accept_belief, list_beliefs."""

from __future__ import annotations

import logging
from typing import Annotated

from pydantic import Field

from team.mcp.builtin import BuiltinContext
from team.mcp.builtin._util import offload, truncate

log = logging.getLogger(__name__)

_DISABLED = "ERROR: beliefs are not enabled (set beliefs.enabled: true)"
_ICONS = {"accepted": "✓", "contested": "⚡", "rejected": "✗", "pending": "?"}


def build(ctx: BuiltinContext) -> "object":
    from mcp.server.fastmcp import FastMCP

    srv = FastMCP("beliefs")
    beliefs = ctx.beliefs
    member_name = ctx.member_name

    @srv.tool()
    @offload
    def assert_belief(
        claim: Annotated[str, Field(description="The claim text.")],
        confidence: Annotated[float, Field(description="Confidence 0–1 (default 0.5).")] = 0.5,
        evidence: Annotated[str, Field(description="Supporting evidence (optional).")] = "",
    ) -> str:
        """Add a claim to the team's shared belief board."""
        if beliefs is None:
            return _DISABLED
        conf = max(0.0, min(1.0, confidence))
        text = claim.strip()
        if not text:
            return "ERROR: claim text must not be empty"
        b = beliefs.assert_belief(text, author=member_name, confidence=conf, evidence=evidence)
        return (
            f"Belief [{b.id}] asserted (status: {b.status}):\n"
            f"  Claim: {b.claim}\n"
            f"  Confidence: {b.confidence:.0%}"
        )

    @srv.tool()
    @offload
    def contest_belief(
        id: Annotated[str, Field(description="Belief ID to contest.")],
        reason: Annotated[
            str, Field(description="Reason for contesting (optional but recommended).")
        ] = "",
    ) -> str:
        """Contest an existing team belief, moving it to contested status."""
        if beliefs is None:
            return _DISABLED
        belief_id = id.strip()
        if not belief_id:
            return "ERROR: provide id: <belief id>"
        try:
            b = beliefs.contest_belief(belief_id, voter=member_name, reason=reason)
        except KeyError:
            return f"ERROR: no belief found with id {belief_id!r}"
        return (
            f"Belief [{b.id}] is now contested.\n"
            f"  Claim: {b.claim}\n"
            f"  Reason given: {reason or '(none)'}"
        )

    @srv.tool()
    @offload
    def accept_belief(
        id: Annotated[str, Field(description="Belief ID to accept.")],
    ) -> str:
        """Cast an accept vote for an existing team belief."""
        if beliefs is None:
            return _DISABLED
        belief_id = id.strip()
        if not belief_id:
            return "ERROR: provide id: <belief id>"
        try:
            b = beliefs.accept_belief(belief_id, voter=member_name)
        except KeyError:
            return f"ERROR: no belief found with id {belief_id!r}"
        return (
            f"Voted to accept belief [{b.id}] (status: {b.status}).\n"
            f"  Votes for: {len(b.votes_for)}, against: {len(b.votes_against)}"
        )

    @srv.tool()
    @offload
    def list_beliefs(
        status: Annotated[
            str,
            Field(description="Filter by status: pending|accepted|contested|rejected (optional)."),
        ] = "",
    ) -> str:
        """List the team's shared belief board, optionally filtered by status."""
        if beliefs is None:
            return _DISABLED
        status_val = status.strip() or None
        if status_val and status_val not in ("pending", "accepted", "contested", "rejected"):
            return f"ERROR: unknown status {status_val!r}; use pending|accepted|contested|rejected"
        items = beliefs.list_beliefs(status=status_val)
        if not items:
            return (
                f"No beliefs with status {status_val!r}."
                if status_val
                else "The belief board is empty."
            )
        lines = [
            f"Belief board — {len(items)} belief(s)"
            + (f" [status={status_val}]" if status_val else "")
            + ":"
        ]
        for b in items:
            icon = _ICONS.get(b.status, "?")
            lines.append(
                f"  [{icon}] `{b.id}` {b.claim}\n"
                f"       conf={b.confidence:.0%}, by @{b.author}, "
                f"for={len(b.votes_for)}, against={len(b.votes_against)}, status={b.status}"
            )
        return truncate("\n".join(lines))

    return srv
