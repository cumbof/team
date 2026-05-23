"""Federated belief synchronization — cross-team belief exchange.

Two independent ``team`` clusters can share their collective knowledge via the
bridge protocol.  Each team's bridge server exposes belief-board endpoints; this
module implements the serialization format and the import/export logic.

Protocol
--------
When Team A wants to share knowledge with Team B:

1. Team A's bridge server exposes ``GET /beliefs`` — returns a
   :class:`BeliefBundle` (JSON).
2. Team B calls :func:`pull_beliefs` (via :class:`~team.bridge_client.BridgeClient`)
   to download the bundle.
3. Team B passes the bundle to :func:`import_beliefs`, which adds each novel
   belief as ``pending`` so it goes through Team B's local consensus process.

Push direction (Team A → Team B):

1. Team A calls :func:`export_beliefs` to build a bundle from its board.
2. Team A sends it to Team B via ``POST /beliefs/import``.
3. Team B runs the same import logic.

Deduplication
-------------
Beliefs are deduplicated by a *claim hash* — a SHA-256 digest of the
normalized claim text (lowercased, punctuation stripped, whitespace collapsed).
If a belief with the same normalized claim already exists locally — regardless
of its current status — the incoming belief is silently skipped.

Author attribution
------------------
Imported beliefs carry the author tag ``"<source_team>/<original_author>"`` so
local members can distinguish their own assertions from imported ones in the
belief board.

Usage example
-------------
::

    from team.beliefs import BeliefBoard
    from team.bridge_client import BridgeClient

    board = BeliefBoard(workspace / "beliefs.json", member_names)
    client = BridgeClient("http://lab-b.example.com:7001")

    # Pull beliefs from Lab B and merge them into the local board
    bundle = client.pull_beliefs(status="accepted", limit=20)
    if bundle is not None:
        imported, skipped = import_beliefs(board, bundle)
        print(f"imported {imported}, skipped {skipped} duplicates")
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from team.beliefs import BeliefBoard


# --------------------------------------------------------------------------- #
# Claim normalisation and hashing (for dedup)
# --------------------------------------------------------------------------- #


def _normalize_claim(claim: str) -> str:
    """Return a stable normalized form of *claim* for deduplication."""
    text = claim.lower()
    text = re.sub(r"[^\w\s]", "", text)    # strip punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text


def claim_hash(claim: str) -> str:
    """Return a 16-char hex digest of the normalized claim (for dedup checks)."""
    return hashlib.sha256(_normalize_claim(claim).encode()).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# BeliefBundle — wire format for cross-team belief exchange
# --------------------------------------------------------------------------- #


@dataclass
class BeliefBundle:
    """A serializable batch of beliefs from one team.

    Parameters
    ----------
    source_team:
        Name of the team that produced this bundle.
    source_url:
        Bridge URL of the source team (for provenance).
    beliefs:
        List of :class:`~team.beliefs.Belief` dicts (via ``to_dict_list()``).
    exported_at:
        Unix timestamp of when the bundle was created.
    filter_status:
        The status filter applied when exporting (e.g. ``"accepted"``), or
        ``None`` when all beliefs were included.
    """

    source_team: str
    source_url: str
    beliefs: list[dict[str, Any]] = field(default_factory=list)
    exported_at: float = field(default_factory=time.time)
    filter_status: str | None = None

    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BeliefBundle":
        known = {f for f in ("source_team", "source_url", "beliefs",
                              "exported_at", "filter_status")}
        return cls(**{k: v for k, v in data.items() if k in known})


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #


def export_beliefs(
    board: "BeliefBoard",
    source_team: str,
    source_url: str,
    status: str | None = None,
    limit: int = 50,
) -> BeliefBundle:
    """Build a :class:`BeliefBundle` from *board*.

    Parameters
    ----------
    board:
        The local belief board to export from.
    source_team:
        Name of the local team (included in the bundle for provenance).
    source_url:
        Bridge URL of the local team's server.
    status:
        Optional status filter (e.g. ``"accepted"``).  ``None`` exports all
        beliefs.
    limit:
        Maximum number of beliefs to include (default: 50).
    """
    beliefs = board.list_beliefs(status=status)[:limit]
    return BeliefBundle(
        source_team=source_team,
        source_url=source_url,
        beliefs=[asdict(b) for b in beliefs],
        filter_status=status,
    )


# --------------------------------------------------------------------------- #
# Import
# --------------------------------------------------------------------------- #


def import_beliefs(
    board: "BeliefBoard",
    bundle: BeliefBundle,
) -> tuple[int, int]:
    """Import beliefs from *bundle* into *board*.

    Each incoming belief is checked against the local board for duplicates
    using the :func:`claim_hash` of the normalized claim text.  Duplicates
    are silently skipped.  New beliefs are added as ``"pending"`` so they must
    pass the local team's consensus process before being accepted.

    Parameters
    ----------
    board:
        The local board to import into.
    bundle:
        The belief bundle received from the remote team.

    Returns
    -------
    tuple[int, int]
        ``(imported, skipped)`` — counts of newly added and duplicate beliefs.
    """
    # Build a set of claim hashes already on the local board.
    existing_hashes = {
        claim_hash(b.claim) for b in board.list_beliefs()
    }

    imported = 0
    skipped = 0
    for raw in bundle.beliefs:
        remote_claim = raw.get("claim", "")
        if not remote_claim:
            skipped += 1
            continue
        if claim_hash(remote_claim) in existing_hashes:
            skipped += 1
            continue
        board.receive_remote_belief(
            claim=remote_claim,
            source_team=bundle.source_team,
            original_author=raw.get("author", "unknown"),
            confidence=float(raw.get("confidence", 0.5)),
            evidence=raw.get("evidence", ""),
            original_id=raw.get("id", ""),
        )
        # Add the new hash so we don't import duplicates within the same bundle.
        existing_hashes.add(claim_hash(remote_claim))
        imported += 1

    return imported, skipped
