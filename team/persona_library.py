"""Predefined persona library for *team* members.

Personas are stored as individual YAML files under the ``personas/`` directory
at the root of this repository (one file per persona, filename = key).  The
library is loaded lazily the first time it is needed.

Users can **add their own personas** by:

1. Dropping a YAML file into the built-in ``personas/`` directory (good for
   contributing back to the project).
2. Setting the ``TEAM_PERSONA_DIR`` environment variable to a custom directory
   that is scanned *in addition to* the built-in directory.  Files in the
   custom directory take precedence over built-in ones with the same key.
3. Installing a Python package that declares a ``team.persona_dirs`` entry
   point.  Each entry point must be a callable that returns a
   :class:`pathlib.Path` pointing to a directory of persona YAML files.
   All such directories are merged; later-installed packages take precedence
   over built-in personas with the same key, and ``TEAM_PERSONA_DIR`` takes
   precedence over all entry-point dirs.

   Example ``pyproject.toml`` declaration::

       [project.entry-points."team.persona_dirs"]
       mypkg = "mypkg:personas_dir"   # personas_dir() → Path

Each YAML file must have the following top-level keys:

.. code-block:: yaml

    role: Principal Investigator
    description: One-line summary shown in ``team personas``.
    persona: |
      Full persona text injected into the system prompt.

Members reference a library persona in their YAML with an ``@``-prefixed key::

    members:
      - name: alice
        model: llama3.1:70b
        persona: "@pi"         # expands role + persona text from personas/pi.yaml

The ``role`` field is optional when a library persona is used — the library's
default role is applied automatically.  You may still override it::

    members:
      - name: alice
        model: llama3.1:70b
        persona: "@pi"
        role: "Lab Director"   # overrides the library's "Principal Investigator"
"""

from __future__ import annotations

import logging
import os
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

# Built-in personas directory — shipped inside the team package.
_BUILTIN_DIR = Path(__file__).parent / "personas"


def _persona_dirs() -> list[tuple[Path, str]]:
    """Return the list of ``(directory, source_label)`` pairs to scan for persona YAML files.

    Priority (highest → lowest):
    1. ``TEAM_PERSONA_DIR`` environment variable → labelled ``"env (TEAM_PERSONA_DIR)"``
    2. Directories registered via ``team.persona_dirs`` entry points
       (in installation order, last-installed wins) → labelled with ``ep.name``
    3. Built-in ``personas/`` directory shipped with ``team-core`` → labelled ``"built-in"``
    """
    dirs: list[tuple[Path, str]] = []

    # 1. Explicit env var override — highest priority.
    env = os.environ.get("TEAM_PERSONA_DIR", "").strip()
    if env:
        custom = Path(env).expanduser().resolve()
        if custom.is_dir():
            dirs.append((custom, "env (TEAM_PERSONA_DIR)"))

    # 2. Entry-point registered persona directories.
    known_paths = {p for p, _ in dirs}
    for ep in entry_points(group="team.persona_dirs"):
        try:
            fn = ep.load()
            p = Path(fn()).expanduser().resolve()
            if p.is_dir() and p not in known_paths:
                dirs.append((p, ep.name))
                known_paths.add(p)
                log.debug("persona_library: loaded persona dir from plugin %r: %s", ep.name, p)
        except Exception as exc:  # noqa: BLE001
            log.warning("persona_library: failed to load persona dir from plugin %r: %s", ep.name, exc)

    # 3. Built-in directory — lowest priority.
    if _BUILTIN_DIR.is_dir():
        dirs.append((_BUILTIN_DIR, "built-in"))

    return dirs


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

_CACHE: dict[str, dict[str, str]] | None = None


def _load() -> dict[str, dict[str, str]]:
    """Scan all persona directories and return the merged PERSONAS dict.

    Results are cached after the first call.  Set ``_CACHE = None`` to force
    a reload (useful in tests).

    Each value dict contains ``role``, ``description``, ``persona``, and
    ``source`` (the human-readable label of the directory the persona came from).
    When the same key appears in multiple directories, the higher-priority
    directory wins and a warning is emitted.
    """
    global _CACHE
    if _CACHE is not None:
        return _CACHE

    personas: dict[str, dict[str, str]] = {}
    # Process built-in dir first (lowest priority) so custom dirs can override.
    for directory, source_label in reversed(_persona_dirs()):
        for yaml_path in sorted(directory.glob("*.yaml")):
            key = yaml_path.stem
            try:
                raw: Any = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                # Skip unreadable / malformed files with a warning.
                import warnings
                warnings.warn(
                    f"team persona_library: skipping {yaml_path} — {exc}",
                    stacklevel=2,
                )
                continue
            if not isinstance(raw, dict):
                continue
            missing = [f for f in ("role", "description", "persona") if f not in raw]
            if missing:
                import warnings
                warnings.warn(
                    f"team persona_library: {yaml_path} missing fields {missing} — skipped",
                    stacklevel=2,
                )
                continue
            if key in personas:
                log.warning(
                    "persona @%s from %r overrides definition from %r",
                    key, source_label, personas[key]["source"],
                )
            personas[key] = {
                "role": str(raw["role"]).strip(),
                "description": str(raw["description"]).strip(),
                "persona": str(raw["persona"]).strip(),
                "source": source_label,
            }

    _CACHE = personas
    return _CACHE


# --------------------------------------------------------------------------- #
# Convenience shim — PERSONAS behaves like a dict but loads lazily
# --------------------------------------------------------------------------- #

class _LazyPersonas:
    """Proxy that loads persona files on first access."""

    def __getitem__(self, key: str) -> dict[str, str]:
        return _load()[key]

    def __contains__(self, key: object) -> bool:
        return key in _load()

    def __iter__(self):
        return iter(_load())

    def keys(self):
        return _load().keys()

    def values(self):
        return _load().values()

    def items(self):
        return _load().items()

    def get(self, key: str, default=None):
        return _load().get(key, default)


PERSONAS = _LazyPersonas()


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def resolve(name: str) -> tuple[str, str]:
    """Return ``(role, persona_text)`` for a library key, or raise :exc:`KeyError`."""
    entry = _load()[name]
    return entry["role"], entry["persona"]


def list_all() -> list[dict[str, str]]:
    """Return a list of dicts (``key``, ``role``, ``description``, ``source``) for all personas."""
    return [
        {"key": k, "role": v["role"], "description": v["description"], "source": v["source"]}
        for k, v in _load().items()
    ]

