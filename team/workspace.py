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

import datetime
import logging
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

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

    @property
    def shared_dir(self) -> Path:
        """Alias for :attr:`shared` — the shared workspace directory."""
        return self.shared

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

    def apply_reply(self, text: str, private_root: Path | None = None) -> list[FileWrite]:
        """Parse and apply all file blocks in *text*.

        If *private_root* is provided, any block whose path starts with
        ``private/`` is written to *private_root* (the member's own scratch
        directory) instead of the shared workspace.  The ``private/`` prefix
        is stripped before writing so ``file:private/notes.md`` becomes
        ``private_root/notes.md`` on disk (still reported as ``private/notes.md``
        in the returned :class:`FileWrite` list).
        """
        writes: list[FileWrite] = []
        for path, body in parse_file_blocks(text):
            try:
                if private_root is not None and path.startswith("private/"):
                    rel = path[len("private/"):]
                    if not rel:
                        continue
                    target = _safe_join(private_root, rel)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    existed = target.exists()
                    data = body.encode("utf-8")
                    target.write_bytes(data)
                    writes.append(FileWrite(
                        path=path,
                        bytes_written=len(data),
                        created=not existed,
                    ))
                else:
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
        # Sort by timestamp descending so the most recently written file is first.
        items = sorted(self._touched.items(), key=lambda kv: kv[1], reverse=True)
        return [p for p, _ in items[:limit]]


# --------------------------------------------------------------------------- #
# Checkpoint management
# --------------------------------------------------------------------------- #

_CHECKPOINT_RE = re.compile(r"^(?P<turn>\d+)_(?P<member>[^_]+)_(?P<ts>\d{8}T\d{6})$")


@dataclass
class CheckpointInfo:
    id: str           # directory name used as the stable identifier
    turn: int         # orchestrator turn index at the time of the snapshot
    member: str       # name of the member about to write files
    timestamp: str    # ISO-like string: YYYYMMDDTHHMMSS
    file_count: int   # number of files in the snapshot
    path: Path        # absolute path to the checkpoint directory


class CheckpointManager:
    """Point-in-time snapshots of the shared workspace.

    Checkpoints are stored under ``<workspace_root>/checkpoints/`` with names
    in the form ``<turn:04d>_<member>_<YYYYMMDDTHHMMSS>``.  A new checkpoint
    is created automatically before every live member turn so that any file
    changes made during that turn can be undone by restoring the preceding
    checkpoint.

    Only non-empty workspaces are snapshotted — if ``shared/`` contains no
    files at the time of the call (e.g. the very first turn) the snapshot is
    skipped and :meth:`create` returns ``None``.
    """

    def __init__(self, workspace_root: Path):
        self.root = Path(workspace_root).resolve()
        self.shared = self.root / "shared"
        self.checkpoints_dir = self.root / "checkpoints"

    # ------------------------------------------------------------------ #
    # Creating snapshots
    # ------------------------------------------------------------------ #

    def create(self, turn_index: int, member_name: str) -> Path | None:
        """Snapshot the current shared workspace before a member's turn.

        Returns the path of the newly created checkpoint directory, or
        ``None`` when the shared workspace contains no files (nothing to
        back up).
        """
        if not self.shared.is_dir() or not any(
            p for p in self.shared.rglob("*") if p.is_file()
        ):
            return None

        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        name = f"{turn_index:04d}_{member_name}_{ts}"
        dest = self.checkpoints_dir / name
        shutil.copytree(self.shared, dest)
        log.info("checkpoint: created %s (%d files)", name, sum(1 for p in dest.rglob("*") if p.is_file()))
        return dest

    # ------------------------------------------------------------------ #
    # Listing
    # ------------------------------------------------------------------ #

    def list_checkpoints(self) -> list[CheckpointInfo]:
        """Return all checkpoints sorted chronologically (oldest first)."""
        if not self.checkpoints_dir.is_dir():
            return []
        results: list[CheckpointInfo] = []
        for d in sorted(self.checkpoints_dir.iterdir()):
            if not d.is_dir():
                continue
            m = _CHECKPOINT_RE.match(d.name)
            if not m:
                continue
            file_count = sum(1 for p in d.rglob("*") if p.is_file())
            results.append(CheckpointInfo(
                id=d.name,
                turn=int(m.group("turn")),
                member=m.group("member"),
                timestamp=m.group("ts"),
                file_count=file_count,
                path=d,
            ))
        return results

    # ------------------------------------------------------------------ #
    # Restoring
    # ------------------------------------------------------------------ #

    def restore(self, checkpoint_id: str) -> CheckpointInfo:
        """Replace the shared workspace with the contents of a checkpoint.

        *checkpoint_id* must match the directory name exactly (as shown by
        :meth:`list_checkpoints`).  Raises :class:`ValueError` if the
        checkpoint does not exist.
        """
        src = self.checkpoints_dir / checkpoint_id
        if not src.is_dir():
            raise ValueError(
                f"checkpoint {checkpoint_id!r} not found under {self.checkpoints_dir}"
            )
        if self.shared.exists():
            shutil.rmtree(self.shared)
        shutil.copytree(src, self.shared)
        log.info("checkpoint: restored %s", checkpoint_id)
        # Return metadata so callers (e.g. the CLI) can confirm what was restored.
        m = _CHECKPOINT_RE.match(checkpoint_id)
        file_count = sum(1 for p in self.shared.rglob("*") if p.is_file())
        return CheckpointInfo(
            id=checkpoint_id,
            turn=int(m.group("turn")) if m else -1,
            member=m.group("member") if m else "?",
            timestamp=m.group("ts") if m else "?",
            file_count=file_count,
            path=src,
        )
