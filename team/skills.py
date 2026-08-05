"""Markdown "skill" registry — reusable, installable standing-context bundles.

A *skill* here is a Markdown file whose content is injected verbatim into a
member's system prompt via ``extra_context:`` — never executable code.
Custom tools are declared as MCP servers under ``mcp_servers:`` instead (see
``team/mcp/``); this module only covers the context-injection half of what
the pre-MCP ``skills.py`` used to do.

A package registers a skill by declaring a ``team.skills`` entry point whose
value is a Python module path. The module must set ``SKILL_FILE`` to the
absolute path of its bundled Markdown file::

    # pyproject.toml
    [project.entry-points."team.skills"]
    review_checklist = "team_myext.skills.review_checklist"

    # team_myext/skills/review_checklist.py
    from pathlib import Path
    SKILL_FILE = str(Path(__file__).parent / "review_checklist.md")

Then reference it by name in any team YAML — alongside plain relative paths,
which continue to work unchanged::

    defaults:
      extra_context:
        - review_checklist            # resolved via the team.skills entry point
        - ./context/local_notes.md    # a plain relative path still works

``team forge <name>`` scaffolds a ``skills/`` directory pre-wired to this
entry-point group.
"""

from __future__ import annotations

import logging
from importlib import import_module
from importlib.metadata import entry_points
from pathlib import Path

log = logging.getLogger(__name__)


class SkillLoadError(Exception):
    """Raised when a registered skill name cannot be resolved to a file."""


_REGISTRY: dict[str, str] | None = None


def _load_registry() -> dict[str, str]:
    """Build a mapping of registered skill name -> Python module path.

    Scans ``team.skills`` entry points. Results are cached after the first
    call; set ``_REGISTRY = None`` to force a reload (used by tests).
    """
    global _REGISTRY
    if _REGISTRY is not None:
        return _REGISTRY

    registry: dict[str, str] = {}
    for ep in entry_points(group="team.skills"):
        registry[ep.name] = ep.value
        log.debug("skill registry: registered %r -> %s", ep.name, ep.value)

    _REGISTRY = registry
    return _REGISTRY


def resolve_skill(name: str) -> str:
    """Resolve a registered skill *name* to an absolute Markdown file path.

    Raises :class:`SkillLoadError` if *name* is not registered, its module
    cannot be imported, or the module does not define ``SKILL_FILE`` pointing
    at an existing file.
    """
    registry = _load_registry()
    if name not in registry:
        known = sorted(registry)
        hint = f" Registered skills: {known}." if known else " No skills are registered via entry points."
        raise SkillLoadError(f"no skill named {name!r} is registered.{hint}")

    module_path = registry[name]
    try:
        module = import_module(module_path)
    except ImportError as exc:
        raise SkillLoadError(
            f"skill {name!r} failed to import ({module_path}): {exc}"
        ) from exc

    skill_file = getattr(module, "SKILL_FILE", None)
    if not skill_file:
        raise SkillLoadError(
            f"skill {name!r} ({module_path}) does not define SKILL_FILE"
        )

    path = Path(skill_file)
    if not path.is_file():
        raise SkillLoadError(f"skill {name!r} points at a missing file: {skill_file}")
    return str(path)
