"""Shared workspace handling and parsing of file blocks from member replies.

The shared workspace lives on the host at ``team.workspace/shared`` and is
bind-mounted as ``/workspace`` inside every member container.  The orchestrator
also writes files there (when parsed from member replies) so that the next
turn's prompt can reference them via :func:`recent_changes`.

Each member also has a **private workspace** at ``team.workspace/members/<name>``
on the host, bind-mounted as ``/private`` inside its container.  The orchestrator
lists those files in every turn prompt so the member can reference its own
scratch files.

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

# Matches fenced code blocks whose info-string starts with "file:".
# Named groups: `path` (the relative target path) and `body` (raw file contents).
# re.DOTALL is required so `body` can span multiple lines.
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


def list_dir_files(root: Path, limit: int = 30) -> list[str]:
    """Return relative paths of all files under *root* (up to *limit*).

    Returns an empty list if *root* does not exist or is not a directory.
    """
    if not root.is_dir():
        return []
    return sorted(
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.is_file()
    )[:limit]


def _safe_join(root: Path, rel: str) -> Path:
    # Strip leading slashes so that absolute paths like `/etc/passwd` are
    # treated as relative to the workspace root rather than the filesystem root.
    rel = rel.lstrip("/")
    target = (root / rel).resolve()
    root_resolved = root.resolve()
    try:
        # `relative_to` raises ValueError if `target` is not under `root_resolved`,
        # catching symlink-based traversal attacks as well as plain `../../` paths.
        target.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"path {rel!r} escapes the workspace") from exc
    return target


class SharedWorkspace:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        # Files are always written under the `shared/` sub-directory so the
        # workspace root itself can hold other runtime files (transcript.jsonl,
        # inject.txt, member private dirs, …) without polluting the shared area.
        self.shared = self.root / "shared"
        self.shared.mkdir(parents=True, exist_ok=True)
        # Maps relative path → last-modified timestamp (float, from time.time()).
        # Used to report "recently changed files" to members on each turn.
        self._touched: dict[str, float] = {}

    def write(self, rel_path: str, body: str) -> FileWrite:
        target = _safe_join(self.shared, rel_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        existed = target.exists()
        data = body.encode("utf-8")
        target.write_bytes(data)
        # Record the timestamp so this path appears near the top of
        # `recent_changes()` for the next few turns.
        self._touched[rel_path] = time.time()
        return FileWrite(path=rel_path, bytes_written=len(data), created=not existed)

    def touch(self, rel_path: str) -> None:
        """Mark a path as recently changed (used when resuming a run).

        During a resume, files written in previous turns are not re-written,
        but we still want them to appear in the "recently changed" section of
        the next turn's prompt so members have the right context.
        """
        self._touched[rel_path] = time.time()

    def apply_reply(self, text: str) -> list[FileWrite]:
        writes: list[FileWrite] = []
        for path, body in parse_file_blocks(text):
            try:
                writes.append(self.write(path, body))
            except ValueError:
                # Skip any file block whose path fails the safety check rather
                # than aborting the whole turn — one bad path shouldn't prevent
                # other valid file blocks in the same reply from being written.
                continue
        return writes

    def list_files(self) -> list[str]:
        return sorted(
            str(p.relative_to(self.shared))
            for p in self.shared.rglob("*")
            if p.is_file()
        )

    def recent_changes(self, limit: int = 10) -> list[str]:
        # Sort by timestamp descending so the most recently written file is first.
        items = sorted(self._touched.items(), key=lambda kv: kv[1], reverse=True)
        return [p for p, _ in items[:limit]]
