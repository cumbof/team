"""Transcript / message bus shared by all members.

Every member sees the same transcript (rendered as user-role messages), so
the conversation stays globally consistent.  The bus also persists the
transcript to disk under ``<workspace>/transcript.jsonl`` so a run can be
inspected (or resumed) after the fact.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass
class Turn:
    index: int
    speaker: str  # member name, "orchestrator", or "human" (injected directive)
    role: str     # member role, "system", or "director"
    content: str
    timestamp: float = field(default_factory=time.time)
    files_written: list[str] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0


class Transcript:
    def __init__(self, persist_path: Path | None = None, resume: bool = False):
        self.turns: list[Turn] = []
        self.persist_path = persist_path
        if persist_path:
            persist_path.parent.mkdir(parents=True, exist_ok=True)
            if resume and persist_path.is_file() and persist_path.stat().st_size > 0:
                # Resuming: load existing turns so the orchestrator can fast-forward
                # through them without re-calling the LLM.
                self._load_from_disk()
            else:
                # Fresh run: truncate any leftover transcript from a previous run.
                persist_path.write_text("", encoding="utf-8")

    def _load_from_disk(self) -> None:
        """Load existing turns from the persisted JSONL file (used when resuming).

        JSONL (newline-delimited JSON) is used because each turn can be appended
        with a single write and the file stays readable even if a run is
        interrupted mid-write (the partial last line is simply skipped).
        """
        for line in self.persist_path.read_text(encoding="utf-8").splitlines():  # type: ignore[union-attr]
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                self.turns.append(Turn(**data))
            except (json.JSONDecodeError, TypeError):
                # Silently skip any malformed lines (e.g. a truncated final write).
                continue

    def append(
        self,
        speaker: str,
        role: str,
        content: str,
        files_written: Iterable[str] = (),
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> Turn:
        turn = Turn(
            index=len(self.turns),
            speaker=speaker,
            role=role,
            content=content,
            files_written=list(files_written),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        self.turns.append(turn)
        if self.persist_path:
            # Append-only write: one JSON object per line.  Keeps disk I/O minimal
            # and allows the file to be tailed live (e.g. by `team transcript`).
            with self.persist_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(asdict(turn), ensure_ascii=False) + "\n")
        return turn

    def stats(self) -> dict[str, Any]:
        """Return a summary of the transcript's statistics.

        Returns a dict with the following keys:

        * ``total_turns``         — total number of recorded turns.
        * ``turns_by_speaker``    — ``{speaker: count}`` mapping.
        * ``total_prompt_tokens`` — sum of prompt tokens across all turns.
        * ``total_completion_tokens`` — sum of completion tokens.
        * ``tokens_by_speaker``   — ``{speaker: {"prompt": int, "completion": int}}``.
        * ``duration_seconds``    — wall time between first and last turn timestamps
          (``None`` when fewer than two turns exist).
        * ``files_written``       — total number of file paths recorded as written.
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

        if len(self.turns) >= 2:
            duration: float | None = self.turns[-1].timestamp - self.turns[0].timestamp
        else:
            duration = None

        return {
            "total_turns": len(self.turns),
            "turns_by_speaker": turns_by_speaker,
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "tokens_by_speaker": tokens_by_speaker,
            "duration_seconds": duration,
            "files_written": total_files,
        }

    def render(
        self,
        viewer: str | None = None,
        max_turns: int | None = None,
        first_n: int | None = None,
    ) -> str:
        """Render the transcript as plain text for inclusion in a prompt.

        ``viewer`` is the member who will *read* this rendering; their own
        previous turns are tagged so they recognise themselves.

        ``max_turns`` caps how many of the *most-recent* turns are included —
        useful for sliding-window context management.

        ``first_n`` keeps only the *oldest* N turns — used when summarizing
        the early portion of a long conversation.  Mutually exclusive with
        ``max_turns``; ``max_turns`` takes precedence when both are provided.
        """
        turns = self.turns
        if max_turns is not None and len(turns) > max_turns:
            turns = turns[-max_turns:]
        elif first_n is not None and len(turns) > first_n:
            turns = turns[:first_n]
        lines: list[str] = []
        for t in turns:
            tag = f"@{t.speaker}"
            if viewer and t.speaker == viewer:
                tag += " (you)"
            header = f"--- Turn {t.index} | {tag} | {t.role} ---"
            lines.append(header)
            lines.append(t.content.rstrip())
            if t.files_written:
                lines.append(f"[wrote files: {', '.join(t.files_written)}]")
            lines.append("")
        return "\n".join(lines).rstrip()
