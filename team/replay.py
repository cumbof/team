"""Interactive transcript replay — browse a saved run turn-by-turn.

:class:`ReplaySession` manages position and navigation state over a list of
:class:`~team.bus.Turn` objects.  Rendering and terminal I/O live in
:mod:`team.cli` so this module stays fully testable without a TTY.

Usage (programmatic)::

    from team.replay import load_transcript, ReplaySession

    turns = load_transcript(Path("runs/myteam/transcript.jsonl"))
    session = ReplaySession(turns)

    session.next()          # advance one turn
    session.jump(5)         # jump to turn 5
    session.find_next("alice")  # forward to alice's next turn
    session.stats()         # dict of totals/counts

CLI entry point::

    team replay myteam.yaml            # interactive, starts at turn 0
    team replay myteam.yaml --from 4   # start at turn 4
    team replay myteam.yaml --speaker alice  # jump to alice's first turn
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from team.bus import Turn


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def load_transcript(path: Path) -> list[Turn]:
    """Load all turns from a JSONL transcript file.

    Silently skips malformed lines (same behaviour as
    :class:`~team.bus.Transcript`).  Returns an empty list when the file
    does not exist or is empty.
    """
    if not path.is_file():
        return []
    turns: list[Turn] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            turns.append(Turn(**data))
        except (json.JSONDecodeError, TypeError):
            continue
    return turns


# --------------------------------------------------------------------------- #
# Session
# --------------------------------------------------------------------------- #


class ReplaySession:
    """Manages replay position and navigation over a list of turns.

    All navigation methods return a boolean indicating whether the position
    actually changed, which lets the caller decide whether to re-render.

    The session never raises on navigation out-of-bounds — it simply returns
    ``False`` and leaves the position unchanged.
    """

    def __init__(self, turns: list[Turn]) -> None:
        self.turns: list[Turn] = list(turns)
        self._idx: int = 0

    # ------------------------------------------------------------------ #
    # State properties
    # ------------------------------------------------------------------ #

    @property
    def index(self) -> int:
        """Current position (0-based turn index)."""
        return self._idx

    @property
    def total(self) -> int:
        """Total number of turns in the transcript."""
        return len(self.turns)

    @property
    def at_start(self) -> bool:
        """True when positioned at the very first turn."""
        return self._idx == 0

    @property
    def at_end(self) -> bool:
        """True when positioned at the very last turn (or transcript is empty)."""
        return not self.turns or self._idx >= len(self.turns) - 1

    def current(self) -> Turn | None:
        """Return the turn at the current position, or ``None`` for an empty transcript."""
        if not self.turns:
            return None
        return self.turns[self._idx]

    # ------------------------------------------------------------------ #
    # Navigation
    # ------------------------------------------------------------------ #

    def next(self) -> bool:
        """Advance one position forward.  Returns ``True`` if position changed."""
        if self._idx < len(self.turns) - 1:
            self._idx += 1
            return True
        return False

    def prev(self) -> bool:
        """Go back one position.  Returns ``True`` if position changed."""
        if self._idx > 0:
            self._idx -= 1
            return True
        return False

    def jump(self, idx: int) -> bool:
        """Jump directly to turn *idx* (0-based).

        Returns ``True`` when *idx* is within ``[0, total)``, ``False`` when
        out of range (position unchanged).
        """
        if 0 <= idx < len(self.turns):
            self._idx = idx
            return True
        return False

    def find_next(self, speaker: str) -> bool:
        """Move forward to the next turn by *speaker* (case-insensitive).

        Searches from ``index + 1`` onward.  Returns ``True`` if a matching
        turn was found and the position was updated; ``False`` otherwise
        (position unchanged).
        """
        needle = speaker.lower()
        for i in range(self._idx + 1, len(self.turns)):
            if self.turns[i].speaker.lower() == needle:
                self._idx = i
                return True
        return False

    def find_prev(self, speaker: str) -> bool:
        """Move backward to the previous turn by *speaker* (case-insensitive).

        Searches from ``index - 1`` down to 0.  Returns ``True`` if found
        (position updated), ``False`` otherwise (position unchanged).
        """
        needle = speaker.lower()
        for i in range(self._idx - 1, -1, -1):
            if self.turns[i].speaker.lower() == needle:
                self._idx = i
                return True
        return False

    # ------------------------------------------------------------------ #
    # Stats
    # ------------------------------------------------------------------ #

    def stats(self) -> dict[str, Any]:
        """Return a statistics summary over **all** turns in the transcript.

        The returned dict mirrors the structure of
        :meth:`~team.bus.Transcript.stats`:

        * ``total_turns``
        * ``turns_by_speaker``   — ``{speaker: count}``
        * ``total_prompt_tokens``
        * ``total_completion_tokens``
        * ``tokens_by_speaker``  — ``{speaker: {"prompt": int, "completion": int}}``
        * ``duration_seconds``   — wall time first→last turn (``None`` if < 2 turns)
        * ``files_written``      — total file paths recorded across all turns
        """
        turns_by_speaker: dict[str, int] = {}
        tokens_by_speaker: dict[str, dict[str, int]] = {}
        total_prompt = 0
        total_completion = 0
        total_files = 0

        for t in self.turns:
            turns_by_speaker[t.speaker] = turns_by_speaker.get(t.speaker, 0) + 1
            sp = tokens_by_speaker.setdefault(t.speaker, {"prompt": 0, "completion": 0})
            sp["prompt"] += t.prompt_tokens
            sp["completion"] += t.completion_tokens
            total_prompt += t.prompt_tokens
            total_completion += t.completion_tokens
            total_files += len(t.files_written)

        duration: float | None = None
        if len(self.turns) >= 2:
            duration = self.turns[-1].timestamp - self.turns[0].timestamp

        return {
            "total_turns": len(self.turns),
            "turns_by_speaker": turns_by_speaker,
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "tokens_by_speaker": tokens_by_speaker,
            "duration_seconds": duration,
            "files_written": total_files,
        }
