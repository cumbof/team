"""Per-agent persistent cross-session memory store.

Each :class:`AgentMemory` is backed by a SQLite database in the team workspace
(``<workspace>/memory/<member_name>.db``).  Memories survive between sessions
so agents accumulate knowledge over time — a postdoc that ran experiments last
week remembers what failed and does not repeat it.

The store supports:

* Free-text search (substring matching on key and value).
* Tag-based filtering (comma-separated tags per memory entry).
* Importance scores (float, default ``1.0``) that influence retrieval ranking.
* ``summary_for_prompt()`` — returns the *n* most important/recent memories as
  a compact Markdown section ready to be injected into a member's system context.

Thread safety: every public method opens and closes its own connection, making
instances safe to call from multiple threads (SQLite WAL mode is enabled).
"""

from __future__ import annotations

import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any


# --------------------------------------------------------------------------- #
# AgentMemory
# --------------------------------------------------------------------------- #


class AgentMemory:
    """Persistent, queryable memory store for a single agent."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id          TEXT PRIMARY KEY,
                    key         TEXT NOT NULL UNIQUE,
                    value       TEXT NOT NULL,
                    tags        TEXT NOT NULL DEFAULT '',
                    importance  REAL NOT NULL DEFAULT 1.0,
                    created_at  REAL NOT NULL,
                    updated_at  REAL NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_key ON memories(key)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_updated ON memories(updated_at)"
            )

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id":         row["id"],
            "key":        row["key"],
            "value":      row["value"],
            "tags":       row["tags"],
            "importance": row["importance"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def remember(
        self,
        key: str,
        value: str,
        tags: list[str] | None = None,
        importance: float = 1.0,
    ) -> str:
        """Store (or update) a memory keyed by *key*.

        If a memory with the same key already exists, its value, tags, and
        importance are updated.  Returns a one-line confirmation string.
        """
        now = time.time()
        tags_str = ",".join(t.strip() for t in (tags or []))
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT id FROM memories WHERE key = ?", (key,)
            ).fetchone()
            if existing:
                conn.execute(
                    "UPDATE memories SET value=?, tags=?, importance=?, updated_at=? WHERE key=?",
                    (value, tags_str, importance, now, key),
                )
                return f"Updated memory: {key!r}"
            mem_id = str(uuid.uuid4())[:8]
            conn.execute(
                "INSERT INTO memories (id, key, value, tags, importance, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (mem_id, key, value, tags_str, importance, now, now),
            )
            return f"Remembered [{mem_id}]: {key!r}"

    def recall(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Search memories by substring match on key and value.

        Results are ordered by importance descending, then by recency.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM memories
                WHERE key LIKE ? OR value LIKE ?
                ORDER BY importance DESC, updated_at DESC
                LIMIT ?
                """,
                (f"%{query}%", f"%{query}%", limit),
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def forget(self, key: str) -> bool:
        """Delete the memory with *key*.  Returns ``True`` if it existed."""
        with self._connect() as conn:
            c = conn.execute("DELETE FROM memories WHERE key = ?", (key,))
            return c.rowcount > 0

    def list_memories(
        self,
        tag: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """List memories, optionally filtered by a single tag.

        Results are ordered by importance descending, then by recency.
        """
        with self._connect() as conn:
            if tag:
                # Match the tag as a whole word inside the comma-separated list.
                rows = conn.execute(
                    """
                    SELECT * FROM memories
                    WHERE (',' || tags || ',') LIKE ?
                    ORDER BY importance DESC, updated_at DESC
                    LIMIT ?
                    """,
                    (f"%,{tag.strip()},%", limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM memories
                    ORDER BY importance DESC, updated_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def recent(self, n: int = 5) -> list[dict[str, Any]]:
        """Return the *n* most recently updated memories."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM memories ORDER BY updated_at DESC LIMIT ?", (n,)
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def count(self) -> int:
        """Return the total number of stored memories."""
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

    def summary_for_prompt(self, n: int = 5) -> str:
        """Compact Markdown summary of the top-*n* recent memories.

        Returns an empty string when there are no memories (so callers can
        cleanly skip the section).
        """
        entries = self.recent(n)
        if not entries:
            return ""
        lines = [f"## Your persistent memories ({len(entries)} shown)"]
        for m in entries:
            tags_part = f" [{m['tags']}]" if m["tags"] else ""
            lines.append(f"- **{m['key']}**{tags_part}: {m['value']}")
        return "\n".join(lines)
