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
from typing import Iterable


@dataclass
class Turn:
    index: int
    speaker: str  # member name or "orchestrator"
    role: str     # member role or "system"
    content: str
    timestamp: float = field(default_factory=time.time)
    files_written: list[str] = field(default_factory=list)


class Transcript:
    def __init__(self, persist_path: Path | None = None, resume: bool = False):
        self.turns: list[Turn] = []
        self.persist_path = persist_path
        if persist_path:
            persist_path.parent.mkdir(parents=True, exist_ok=True)
            if resume and persist_path.is_file() and persist_path.stat().st_size > 0:
                self._load_from_disk()
            else:
                persist_path.write_text("", encoding="utf-8")

    def _load_from_disk(self) -> None:
        """Load existing turns from the persisted JSONL file (used when resuming)."""
        for line in self.persist_path.read_text(encoding="utf-8").splitlines():  # type: ignore[union-attr]
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                self.turns.append(Turn(**data))
            except (json.JSONDecodeError, TypeError):
                continue

    def append(
        self,
        speaker: str,
        role: str,
        content: str,
        files_written: Iterable[str] = (),
    ) -> Turn:
        turn = Turn(
            index=len(self.turns),
            speaker=speaker,
            role=role,
            content=content,
            files_written=list(files_written),
        )
        self.turns.append(turn)
        if self.persist_path:
            with self.persist_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(asdict(turn), ensure_ascii=False) + "\n")
        return turn

    def render(self, viewer: str | None = None, max_turns: int | None = None) -> str:
        """Render the transcript as plain text for inclusion in a prompt.

        ``viewer`` is the member who will *read* this rendering; their own
        previous turns are tagged so they recognise themselves.
        """
        turns = self.turns
        if max_turns is not None and len(turns) > max_turns:
            turns = turns[-max_turns:]
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
