"""Shared workspace handling and parsing of file blocks from member replies.

The shared workspace lives on the host at ``team.workspace/shared`` and is
bind-mounted as ``/workspace`` inside every member container.  The orchestrator
also writes files there (when parsed from member replies) so that the next
turn's prompt can reference them via :func:`recent_changes`.

We parse file blocks of the form::

    ```file:relative/path.ext
    contents
    ```

with safety checks against path traversal.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path

_FILE_BLOCK_RE = re.compile(
    r"```file:(?P<path>[^\s`]+)\n(?P<body>.*?)```",
    re.DOTALL,
)


@dataclass
class FileWrite:
    path: str  # relative to shared workspace
    bytes_written: int
    created: bool


def parse_file_blocks(text: str) -> list[tuple[str, str]]:
    """Return a list of ``(path, body)`` tuples for every ``file:`` block."""
    return [(m.group("path").strip(), m.group("body")) for m in _FILE_BLOCK_RE.finditer(text)]


def _safe_join(root: Path, rel: str) -> Path:
    rel = rel.lstrip("/")
    target = (root / rel).resolve()
    root_resolved = root.resolve()
    try:
        target.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"path {rel!r} escapes the workspace") from exc
    return target


class SharedWorkspace:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.shared = self.root / "shared"
        self.shared.mkdir(parents=True, exist_ok=True)
        self._touched: dict[str, float] = {}

    def write(self, rel_path: str, body: str) -> FileWrite:
        target = _safe_join(self.shared, rel_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        existed = target.exists()
        data = body.encode("utf-8")
        target.write_bytes(data)
        self._touched[rel_path] = time.time()
        return FileWrite(path=rel_path, bytes_written=len(data), created=not existed)

    def apply_reply(self, text: str) -> list[FileWrite]:
        writes: list[FileWrite] = []
        for path, body in parse_file_blocks(text):
            try:
                writes.append(self.write(path, body))
            except ValueError:
                continue
        return writes

    def list_files(self) -> list[str]:
        return sorted(
            str(p.relative_to(self.shared))
            for p in self.shared.rglob("*")
            if p.is_file()
        )

    def recent_changes(self, limit: int = 10) -> list[str]:
        items = sorted(self._touched.items(), key=lambda kv: kv[1], reverse=True)
        return [p for p, _ in items[:limit]]
