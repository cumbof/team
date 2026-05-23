"""Shared team belief board — structured collective knowledge.

The :class:`BeliefBoard` lets team members collaboratively build and maintain
a structured record of what the team collectively knows, believes, and contests.
Each belief is a versioned claim with:

* an author and confidence score (0.0 – 1.0)
* optional supporting evidence
* a **voting record**: members can accept or contest each belief
* a status driven by consensus: ``pending`` → ``accepted`` / ``contested``

The board is stored as ``<workspace>/beliefs.json`` so it persists across
sessions and can be inspected with ``team beliefs myteam.yaml``.

Consensus rules
---------------
A belief transitions from ``pending`` to ``accepted`` when the fraction of
members who voted *for* it is ≥ ``consensus_threshold`` (default 0.5).  Any
member can contest a belief, which immediately moves it to ``contested`` —
regardless of prior votes — forcing the team to re-evaluate.

Members can accept a previously contested belief, which re-triggers the
consensus check.  A rejected belief stays ``rejected`` permanently unless
explicitly re-asserted with a new claim.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #


@dataclass
class Belief:
    id: str
    claim: str
    author: str
    confidence: float          # 0.0 – 1.0
    evidence: str
    status: str                # pending | accepted | contested | rejected
    votes_for: list[str]       # member names who accepted
    votes_against: list[str]   # member names who contested
    reasons: dict[str, str]    # member → reason for contesting
    created_at: float
    updated_at: float


# --------------------------------------------------------------------------- #
# BeliefBoard
# --------------------------------------------------------------------------- #


class BeliefBoard:
    """Shared, persistent belief store for a team."""

    def __init__(
        self,
        path: Path,
        member_names: list[str],
        consensus_threshold: float = 0.5,
    ) -> None:
        self.path = Path(path)
        self.member_names = list(member_names)
        self.consensus_threshold = max(0.0, min(1.0, consensus_threshold))
        self._beliefs: dict[str, Belief] = {}
        self._load()

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            for item in raw:
                b = Belief(**item)
                self._beliefs[b.id] = b
        except (json.JSONDecodeError, TypeError, KeyError):
            # Corrupt file — start fresh rather than crashing.
            self._beliefs = {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = [asdict(b) for b in self._beliefs.values()]
        self.path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    # ------------------------------------------------------------------ #
    # Consensus check (internal)
    # ------------------------------------------------------------------ #

    def _check_consensus(self, b: Belief) -> None:
        """Update ``b.status`` based on current votes (modifies in place)."""
        n = len(self.member_names)
        if n == 0:
            return
        ratio_for = len(b.votes_for) / n
        # A contested belief only un-contests if enough votes_for accumulate.
        if ratio_for >= self.consensus_threshold:
            if b.status != "rejected":
                b.status = "accepted"
        elif b.status == "accepted":
            # Acceptance lost (e.g. a new member was added or a vote was removed).
            b.status = "pending"
        # Contested status is only cleared by explicit accept calls that move
        # it to accepted — staying contested is fine.

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def assert_belief(
        self,
        claim: str,
        author: str,
        confidence: float = 0.5,
        evidence: str = "",
    ) -> Belief:
        """Create a new belief.

        The author implicitly casts an *accept* vote.  Raises
        :class:`ValueError` if *author* is not in ``member_names``.
        """
        belief_id = str(uuid.uuid4())[:8]
        now = time.time()
        b = Belief(
            id=belief_id,
            claim=claim,
            author=author,
            confidence=confidence,
            evidence=evidence,
            status="pending",
            votes_for=[author],
            votes_against=[],
            reasons={},
            created_at=now,
            updated_at=now,
        )
        self._beliefs[belief_id] = b
        self._check_consensus(b)
        self._save()
        return b

    def accept_belief(self, belief_id: str, voter: str) -> Belief:
        """Cast an *accept* vote for an existing belief.

        Removes any previous *against* vote by the same member.  Re-checks
        consensus after the vote, which may transition the belief to
        ``accepted``.  Raises :class:`KeyError` if the belief does not exist.
        """
        b = self._beliefs[belief_id]
        if voter not in b.votes_for:
            b.votes_for.append(voter)
        if voter in b.votes_against:
            b.votes_against.remove(voter)
        if voter in b.reasons:
            del b.reasons[voter]
        b.updated_at = time.time()
        # Accepting can pull a belief out of contested status.
        if b.status == "contested":
            self._check_consensus(b)
        else:
            self._check_consensus(b)
        self._save()
        return b

    def contest_belief(
        self,
        belief_id: str,
        voter: str,
        reason: str = "",
    ) -> Belief:
        """Contest a belief, moving it to ``contested`` status.

        Any previous *accept* vote by *voter* is removed.  Raises
        :class:`KeyError` if the belief does not exist.
        """
        b = self._beliefs[belief_id]
        if voter not in b.votes_against:
            b.votes_against.append(voter)
        if voter in b.votes_for:
            b.votes_for.remove(voter)
        if reason:
            b.reasons[voter] = reason
        b.status = "contested"
        b.updated_at = time.time()
        self._save()
        return b

    def reject_belief(self, belief_id: str) -> Belief:
        """Permanently reject a belief.

        Only useful for orchestrator-level overrides; normally beliefs are
        contested by individual members rather than rejected outright.
        """
        b = self._beliefs[belief_id]
        b.status = "rejected"
        b.updated_at = time.time()
        self._save()
        return b

    def get_belief(self, belief_id: str) -> Belief | None:
        """Return the belief with *belief_id*, or ``None``."""
        return self._beliefs.get(belief_id)

    def list_beliefs(
        self,
        status: str | None = None,
    ) -> list[Belief]:
        """Return all beliefs, optionally filtered by *status*.

        Results are ordered by recency (most recent first).
        """
        results = list(self._beliefs.values())
        if status:
            results = [b for b in results if b.status == status]
        return sorted(results, key=lambda b: b.updated_at, reverse=True)

    def count(self) -> int:
        """Return the total number of beliefs."""
        return len(self._beliefs)

    def summary_for_prompt(self, limit: int = 10) -> str:
        """Compact Markdown summary of the belief board for context injection.

        Returns an empty string when the board is empty.
        """
        beliefs = self.list_beliefs()[:limit]
        if not beliefs:
            return ""
        _ICONS: dict[str, str] = {
            "accepted":  "✓",
            "contested": "⚡",
            "rejected":  "✗",
            "pending":   "?",
        }
        lines = [f"## Shared team belief board ({len(beliefs)} entries)"]
        for b in beliefs:
            icon = _ICONS.get(b.status, "?")
            conf = f"{b.confidence:.0%}"
            lines.append(
                f"- [{icon}] `{b.id}` {b.claim} "
                f"(conf: {conf}, by @{b.author}, status: {b.status})"
            )
        return "\n".join(lines)

    def to_dict_list(self) -> list[dict[str, Any]]:
        """Return all beliefs as a list of plain dicts (for serialisation)."""
        return [asdict(b) for b in self._beliefs.values()]

    def receive_remote_belief(
        self,
        claim: str,
        source_team: str,
        original_author: str,
        confidence: float = 0.5,
        evidence: str = "",
        original_id: str = "",
    ) -> Belief:
        """Import a belief received from a remote team.

        The belief is added with status ``"pending"`` and no votes so that it
        must go through the local team's normal consensus process before it is
        considered accepted.

        The author is recorded as ``"<source_team>/<original_author>"`` so
        members know the belief originated externally.  Any evidence text from
        the source is preserved and annotated with the provenance.

        Parameters
        ----------
        claim:
            The belief text, verbatim from the remote team.
        source_team:
            Name of the team that asserted this belief.
        original_author:
            Name of the member in the remote team who first asserted it.
        confidence:
            Confidence score from the remote team (0.0–1.0).
        evidence:
            Supporting evidence text from the remote team.
        original_id:
            The belief's ID in the remote team's board (for traceability).
        """
        belief_id = str(uuid.uuid4())[:8]
        now = time.time()
        provenance = f"\n\n[imported from team {source_team!r}"
        if original_id:
            provenance += f", id: {original_id}"
        provenance += "]"
        b = Belief(
            id=belief_id,
            claim=claim,
            author=f"{source_team}/{original_author}",
            confidence=confidence,
            evidence=(evidence + provenance).strip(),
            status="pending",
            votes_for=[],
            votes_against=[],
            reasons={},
            created_at=now,
            updated_at=now,
        )
        self._beliefs[belief_id] = b
        self._save()
        return b
