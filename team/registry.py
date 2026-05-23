"""Team Registry — service-discovery layer for cross-team collaboration.

The registry is a lightweight directory of running team clusters.  Each team
can *register* itself with a central registry server, advertising its name,
URL, speciality tags, available tools, and models.  Any member (or another
team) can then *query* the registry to find a team capable of handling a
given task — and immediately delegate to it via the bridge protocol.

Protocol objects
----------------
:class:`RegistryEntry`
    One registered team.  Carries everything a consumer needs to decide
    whether to delegate work to that team and how to reach it.

:class:`TeamRegistry`
    Server-side, thread-safe store of live entries.  A background daemon
    thread evicts entries whose heartbeat has not been refreshed within
    ``entry_ttl_seconds``.

Wire format
-----------
All objects are serialised as plain JSON so any language or tool can
interact with a registry server — no shared Python code is required.

Typical workflow
----------------
1. Team B starts and registers itself::

       POST http://registry:8000/register
       {"name":"lab-b","url":"http://lab-b:7001","tags":["nlp","summarisation"]}

2. Team A asks who can help with summarisation::

       GET http://registry:8000/teams?tag=summarisation

3. Team A receives the entry for Lab B, then delegates via the bridge::

       ```tool:delegate_task
       url: http://lab-b:7001
       goal: Summarise the attached paper.
       ```
"""

from __future__ import annotations

import hashlib
import hmac as _hmac
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any


# --------------------------------------------------------------------------- #
# HMAC-SHA256 request signing (mirrors bridge.py)
# --------------------------------------------------------------------------- #

#: Requests with a timestamp older than this (seconds) are rejected.
TIMESTAMP_TOLERANCE: int = 300


def sign_request(secret: str, timestamp: str, body: bytes) -> str:
    """Return the HMAC-SHA256 hex digest that authenticates a registry request."""
    msg = f"{timestamp}:".encode() + body
    return _hmac.new(secret.encode(), msg, hashlib.sha256).hexdigest()


def verify_signature(
    secret: str, timestamp: str, body: bytes, signature: str
) -> bool:
    """Return ``True`` when *signature* is valid and *timestamp* is fresh."""
    try:
        age = abs(time.time() - float(timestamp))
    except ValueError:
        return False
    if age > TIMESTAMP_TOLERANCE:
        return False
    expected = sign_request(secret, timestamp, body)
    return _hmac.compare_digest(expected, signature)


# --------------------------------------------------------------------------- #
# Registry entry
# --------------------------------------------------------------------------- #


@dataclass
class RegistryEntry:
    """A single registered team cluster.

    Parameters
    ----------
    name:
        Unique, stable identifier for this team (e.g. ``"lab-b"``).
    url:
        Base URL of the team's bridge server (e.g.
        ``"http://lab-b.example.com:7001"``).  Used directly by
        bridge clients to submit tasks.
    description:
        Short human-readable description of what this team does.
    tags:
        Free-form capability labels (e.g. ``["nlp", "summarisation",
        "python"]``).  Used for tag-based discovery queries.
    models:
        LLM model identifiers the team members use (informational).
    tools:
        Built-in or custom tool names that team members have access to.
    capacity:
        Maximum number of tasks this team can handle concurrently.
        ``0`` means unknown / unlimited.
    metadata:
        Arbitrary key-value pairs for application-specific extensions.
    entry_id:
        Stable identifier for this registration; auto-generated.
    registered_at:
        Unix timestamp of when the entry was first registered.
    last_heartbeat:
        Unix timestamp of the most recent heartbeat; updated on each
        ``POST /heartbeat/<name>`` call.
    """

    name: str
    url: str
    description: str = ""
    tags: list[str] = field(default_factory=list)
    models: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    capacity: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    entry_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    registered_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)

    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RegistryEntry":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def matches(self, tags: list[str] | None, keyword: str | None) -> bool:
        """Return ``True`` if this entry satisfies the query predicates.

        Both predicates are AND-combined.  An empty predicate always matches.

        * *tags*: every requested tag must appear in ``self.tags``
          (case-insensitive).
        * *keyword*: the keyword must appear (case-insensitive) in at least
          one of ``name``, ``description``, or ``tags``.
        """
        if tags:
            my_tags_lower = {t.lower() for t in self.tags}
            if not all(t.lower() in my_tags_lower for t in tags):
                return False
        if keyword:
            kw = keyword.lower()
            haystack = (
                self.name.lower()
                + " "
                + self.description.lower()
                + " "
                + " ".join(t.lower() for t in self.tags)
            )
            if kw not in haystack:
                return False
        return True


# --------------------------------------------------------------------------- #
# Server-side registry store
# --------------------------------------------------------------------------- #


class TeamRegistry:
    """Thread-safe registry of live team entries with heartbeat-based TTL.

    Parameters
    ----------
    entry_ttl_seconds:
        How long an entry may live without a heartbeat before being evicted.
        Pass ``0`` to disable eviction (entries live forever — useful for
        tests or short-lived servers).
    """

    def __init__(self, entry_ttl_seconds: float = 300.0) -> None:
        self._lock = threading.Lock()
        self._entries: dict[str, RegistryEntry] = {}  # keyed by name
        self._ttl = entry_ttl_seconds
        if entry_ttl_seconds > 0:
            self._start_eviction()

    # ------------------------------------------------------------------ #
    # Writing
    # ------------------------------------------------------------------ #

    def register(self, entry: RegistryEntry) -> RegistryEntry:
        """Register (or update) a team entry.

        If an entry with the same ``name`` already exists, it is replaced
        entirely; the ``registered_at`` field of the *existing* entry is
        preserved so it reflects the original registration time.
        """
        with self._lock:
            existing = self._entries.get(entry.name)
            if existing:
                entry = RegistryEntry(
                    **{**entry.to_dict(), "registered_at": existing.registered_at}
                )
            entry.last_heartbeat = time.time()
            self._entries[entry.name] = entry
            return entry

    def deregister(self, name: str) -> bool:
        """Remove the entry for *name*.  Returns ``True`` if it existed."""
        with self._lock:
            return self._entries.pop(name, None) is not None

    def heartbeat(self, name: str) -> bool:
        """Refresh the heartbeat timestamp for *name*.

        Returns ``True`` if the entry was found and updated, ``False`` if the
        name is not currently registered (the caller should re-register).
        """
        with self._lock:
            entry = self._entries.get(name)
            if entry is None:
                return False
            entry.last_heartbeat = time.time()
            return True

    # ------------------------------------------------------------------ #
    # Reading
    # ------------------------------------------------------------------ #

    def get(self, name: str) -> RegistryEntry | None:
        """Return the entry for *name*, or ``None``."""
        with self._lock:
            return self._entries.get(name)

    def list_all(self) -> list[RegistryEntry]:
        """Return all entries, most-recently-registered first."""
        with self._lock:
            return sorted(
                self._entries.values(),
                key=lambda e: e.registered_at,
                reverse=True,
            )

    def query(
        self,
        tags: list[str] | None = None,
        keyword: str | None = None,
    ) -> list[RegistryEntry]:
        """Return entries that match all supplied predicates.

        Parameters
        ----------
        tags:
            Required capability tags (all must be present).
        keyword:
            Substring to search across name, description, and tags.

        Results are sorted by most-recently-registered first.
        """
        with self._lock:
            results = [e for e in self._entries.values() if e.matches(tags, keyword)]
        return sorted(results, key=lambda e: e.registered_at, reverse=True)

    def count(self) -> int:
        """Return the total number of registered entries."""
        with self._lock:
            return len(self._entries)

    # ------------------------------------------------------------------ #
    # TTL eviction
    # ------------------------------------------------------------------ #

    def _start_eviction(self) -> None:
        """Start a daemon thread that purges stale entries."""
        interval = min(self._ttl / 4, 60.0)  # at most every 60 s

        def _evict() -> None:
            while True:
                time.sleep(interval)
                cutoff = time.time() - self._ttl
                with self._lock:
                    stale = [
                        name
                        for name, entry in self._entries.items()
                        if entry.last_heartbeat < cutoff
                    ]
                    for name in stale:
                        del self._entries[name]

        threading.Thread(
            target=_evict, daemon=True, name="registry-eviction"
        ).start()
