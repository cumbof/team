"""Skill loader — extend the agent toolbox with custom tools (F4 extension).

A *skill* is either a Python file that exports one or more tool callables,
or a Markdown file whose content is injected verbatim into the member's
system prompt as background knowledge.

Skills can be loaded from:

* **Local files** — any path on the host filesystem (trusted by default).
* **Remote URLs** (``http://`` / ``https://``) — fetched via ``requests``.
* **Registered names** — short names registered by installed packages via the
  ``team.skills`` entry-point group.  Any package can contribute skills that
  are then referenceable by name in any team YAML without knowing the
  installation path.

  .. warning::
     Remote skill files execute **arbitrary code on the host** with the
     privileges of the ``team`` process.  Treat a remote skill URL with the
     same caution as ``curl URL | python``.

Plugin skills via entry points
-------------------------------
A Python package registers skills by declaring ``team.skills`` entry points
in its ``pyproject.toml``::

    [project.entry-points."team.skills"]
    bioblend      = "team_galaxy.skills.bioblend"
    iuc_standards = "team_galaxy.skills.iuc_standards"

The entry-point value is a Python module path.  For Python (``.py``) skills
the module is imported and its ``TOOLS`` / ``TOOL_NAME`` / ``execute``
interface is used.  For Markdown (``.md``) context-injection skills the
value should be a module path whose ``__file__`` attribute resolves to
the ``.md`` file (place the ``.md`` alongside a thin ``__init__.py`` that
exposes it, or use the ``path:`` form instead).

In practice, the simplest approach is to point entry-point values directly
at Python modules for ``.py`` skills.  For ``.md`` skills, register a
helper module that sets ``SKILL_FILE`` to the absolute path of the
Markdown file::

    # team_mypkg/skills/my_context.py  (thin wrapper for a .md file)
    from pathlib import Path
    SKILL_FILE = str(Path(__file__).parent / "my_context.md")

Then in ``pyproject.toml``::

    [project.entry-points."team.skills"]
    my_context = "team_mypkg.skills.my_context"

Alternatively, point to the Markdown file directly using the ``path:`` form
in the team YAML — registered names are most useful for ``.py`` skills.

Once registered, reference a skill by its entry-point name in any team YAML::

    defaults:
      skills:
        - bioblend        # resolved via team.skills entry point
        - iuc_standards   # same
        - path: ./local.py          # local file — works as before
        - url: https://example.com/skill.py  # remote — works as before

Skill file format
-----------------
**Markdown skills** (``.md`` extension) — context-injection skills::

    # Review Checklist
    ...any markdown content...

The content is injected into the member's system prompt automatically.
No tool call is required; the LLM receives it as background knowledge.
This is useful for guidelines, checklists, templates, and domain rules
that should always be visible rather than looked up on demand.

**Python skills** — one of two callable formats (or both):

**Single-tool format** — ``TOOL_NAME`` string + ``execute`` callable::

    TOOL_NAME = "my_calculator"
    TOOL_DESCRIPTION = "Evaluate a Python arithmetic expression."

    def execute(body, *, workspace_path=None, timeout=30, **kwargs):
        try:
            return str(eval(body.strip(), {"__builtins__": {}}, {}))
        except Exception as exc:
            return f"ERROR: {exc}"

**Multi-tool format** — ``TOOLS`` dict (+ optional ``TOOL_DESCRIPTIONS``)::

    def _add(body, **kw):
        a, b = map(float, body.split())
        return str(a + b)

    def _mul(body, **kw):
        a, b = map(float, body.split())
        return str(a * b)

    TOOLS = {"add": _add, "multiply": _mul}
    TOOL_DESCRIPTIONS = {
        "add":      "Add two numbers.",
        "multiply": "Multiply two numbers.",
    }

Both formats can coexist in the same file; ``TOOLS`` is read first, then
any ``TOOL_NAME`` / ``execute`` pair is merged in.

A Python skill may also set ``INJECT_INTO_CONTEXT`` to a non-empty string
to have that text injected into the system prompt *in addition to* being
a callable tool::

    TOOL_NAME = "style_guide"
    TOOL_DESCRIPTION = "Return the project style guide."
    INJECT_INTO_CONTEXT = "## Style guide\\n- Use snake_case for variables.\\n..."

    def execute(body, **kwargs):
        return INJECT_INTO_CONTEXT

Return values
-------------
:func:`load_skill` and :func:`load_skills` return a 3-tuple
``(tools, descriptions, injected_context)`` where *injected_context* is
a list of strings to be prepended to the member's system prompt.

Config example
--------------
::

    defaults:
      skills:
        - bioblend                            # registered name (team.skills entry point)
        - iuc_standards                       # same
        - path: ./skills/review_checklist.md  # markdown → system-prompt injection
        - path: ./skills/my_tool.py           # python  → callable tool
        - url: https://example.com/tool.py    # remote (loaded with a warning)
          checksum: sha256:abc123...           # optional integrity check
        - ./skills/another.py                 # plain string = local path
        - https://raw.githubusercontent.com/org/repo/main/skill.py

    members:
      - name: researcher
        tools: [web_search, my_tool]           # built-ins AND skill tools by name
        skills:
          - ./skills/domain_search.py          # member-specific skill

Checksum verification
---------------------
Provide a ``checksum`` key (``"<algo>:<hexdigest>"``) for any skill whose
integrity you want to verify:

::

    skills:
      - url: https://example.com/skill.py
        checksum: sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
      - path: ./skills/local.py
        checksum: sha256:d41d8cd98f00b204e9800998ecf8427e   # also works for local

"""

from __future__ import annotations

import hashlib
import importlib
import logging
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger(__name__)

#: Type alias for a tool callable.
ToolFn = Callable[..., str]


class SkillLoadError(Exception):
    """Raised when a skill cannot be loaded or does not export valid tools."""


# --------------------------------------------------------------------------- #
# Entry-point skill registry
# --------------------------------------------------------------------------- #

_SKILL_REGISTRY: dict[str, str] | None = None


def _load_skill_registry() -> dict[str, str]:
    """Build a mapping of registered skill name → Python module path.

    Scans ``team.skills`` entry points.  Results are cached after the first
    call; set ``_SKILL_REGISTRY = None`` to force a reload (useful in tests).
    """
    global _SKILL_REGISTRY
    if _SKILL_REGISTRY is not None:
        return _SKILL_REGISTRY

    registry: dict[str, str] = {}
    for ep in entry_points(group="team.skills"):
        registry[ep.name] = ep.value
        log.debug("skill registry: registered %r → %s", ep.name, ep.value)

    _SKILL_REGISTRY = registry
    return _SKILL_REGISTRY


def _load_skill_by_name(name: str) -> tuple[dict[str, ToolFn], dict[str, str], list[str]]:
    """Load a skill by its registered entry-point name.

    The entry-point value is a Python module path (e.g.
    ``team_galaxy.skills.bioblend``).  The module is imported and inspected
    for ``TOOLS`` / ``TOOL_NAME`` / ``execute`` just like a ``.py`` skill
    file, *except* that Markdown-only skills may instead set ``SKILL_FILE``
    to an absolute path to a ``.md`` file.

    Raises :class:`SkillLoadError` if the name is not registered or the
    module cannot be loaded.
    """
    registry = _load_skill_registry()
    if name not in registry:
        known = sorted(registry)
        hint = f"  Registered names: {known}" if known else "  No skills are registered via entry points."
        raise SkillLoadError(
            f"unknown skill name {name!r}.  "
            f"Install a package that registers it via the 'team.skills' entry-point group.\n{hint}"
        )

    module_path = registry[name]
    try:
        mod = importlib.import_module(module_path)
    except ImportError as exc:
        raise SkillLoadError(
            f"failed to import skill module {module_path!r} for skill {name!r}: {exc}"
        ) from exc

    # Check for a SKILL_FILE attribute pointing to a .md file.
    skill_file = getattr(mod, "SKILL_FILE", None)
    if skill_file and Path(skill_file).suffix.lower() == ".md":
        content = Path(skill_file).read_text(encoding="utf-8")
        return {}, {}, ([content] if content.strip() else [])

    # Otherwise treat the module as a normal Python skill — re-exec its source
    # so it runs in an isolated namespace (same behaviour as a .py file skill).
    source_file = getattr(mod, "__file__", None)
    if not source_file or not Path(source_file).is_file():
        raise SkillLoadError(
            f"skill module {module_path!r} has no readable __file__ attribute"
        )
    code = Path(source_file).read_text(encoding="utf-8")
    return _exec_skill_code(code, module_path)


# --------------------------------------------------------------------------- #
# Low-level execution helpers
# --------------------------------------------------------------------------- #


def _exec_skill_code(
    code: str,
    source_label: str,
) -> tuple[dict[str, ToolFn], dict[str, str], list[str]]:
    """Execute *code* in an isolated namespace and extract tool definitions.

    Returns ``(tools_dict, descriptions_dict, injected_context)``.
    *injected_context* is a (possibly empty) list of strings to be injected
    into the member's system prompt.
    Raises :class:`SkillLoadError` if the module does not export valid tools.
    """
    namespace: dict[str, Any] = {}
    try:
        exec(compile(code, source_label, "exec"), namespace)  # noqa: S102
    except Exception as exc:
        raise SkillLoadError(
            f"error executing skill {source_label!r}: {exc}"
        ) from exc

    tools: dict[str, ToolFn] = {}
    descriptions: dict[str, str] = {}

    # Multi-tool format: TOOLS dict.
    if "TOOLS" in namespace:
        raw = namespace["TOOLS"]
        if not isinstance(raw, dict):
            raise SkillLoadError(
                f"skill {source_label!r}: TOOLS must be a dict mapping name → callable"
            )
        for name, fn in raw.items():
            if not callable(fn):
                raise SkillLoadError(
                    f"skill {source_label!r}: TOOLS[{name!r}] is not callable"
                )
            tools[name] = fn
        raw_descs = namespace.get("TOOL_DESCRIPTIONS", {})
        if isinstance(raw_descs, dict):
            descriptions.update(raw_descs)

    # Single-tool format: TOOL_NAME + execute (can coexist with multi-tool).
    if "TOOL_NAME" in namespace and "execute" in namespace:
        name = namespace["TOOL_NAME"]
        fn = namespace["execute"]
        if not callable(fn):
            raise SkillLoadError(
                f"skill {source_label!r}: execute must be callable"
            )
        tools[name] = fn
        descriptions.setdefault(
            name,
            namespace.get("TOOL_DESCRIPTION", f"Custom skill: {name}"),
        )

    if not tools:
        raise SkillLoadError(
            f"skill {source_label!r} must define either a TOOLS dict or "
            "both TOOL_NAME (str) and execute (callable)"
        )

    # Optional context injection: INJECT_INTO_CONTEXT = "markdown text"
    injected: list[str] = []
    ctx = namespace.get("INJECT_INTO_CONTEXT")
    if isinstance(ctx, str) and ctx.strip():
        injected.append(ctx.strip())

    log.debug("loaded skill %r: tools=%s", source_label, list(tools))
    return tools, descriptions, injected


# --------------------------------------------------------------------------- #
# Checksum verification
# --------------------------------------------------------------------------- #


def _verify_checksum(data: bytes, checksum: str, label: str) -> None:
    """Verify *data* against *checksum* (``"<algo>:<hexdigest>"`` format).

    Raises :class:`SkillLoadError` on mismatch.
    """
    try:
        algo, expected = checksum.split(":", 1)
    except ValueError:
        raise SkillLoadError(
            f"checksum must be in 'algo:hexdigest' format, got: {checksum!r}"
        )
    algo = algo.lower()
    try:
        h = hashlib.new(algo)
    except ValueError as exc:
        raise SkillLoadError(f"unsupported hash algorithm: {algo!r}") from exc
    h.update(data)
    actual = h.hexdigest()
    if actual != expected:
        raise SkillLoadError(
            f"checksum mismatch for skill {label!r}: "
            f"expected {expected!r}, got {actual!r}"
        )
    log.debug("skill %r: checksum OK (%s)", label, algo)


# --------------------------------------------------------------------------- #
# Source loading helpers
# --------------------------------------------------------------------------- #


def _load_local(path_str: str, checksum: str | None = None) -> str:
    """Read a local skill file, optionally verifying its checksum."""
    p = Path(path_str).expanduser()
    if not p.is_file():
        raise SkillLoadError(f"skill file not found: {p}")
    code = p.read_text(encoding="utf-8")
    if checksum:
        _verify_checksum(code.encode("utf-8"), checksum, str(p))
    return code


def _load_remote(url: str, checksum: str | None = None) -> str:
    """Fetch a remote skill file, optionally verifying its checksum."""
    log.warning(
        "loading REMOTE skill from %s — this executes arbitrary code on the "
        "host machine with the privileges of the 'team' process.  Only load "
        "skills from URLs you fully trust.",
        url,
    )
    try:
        import requests  # already a project dependency
        resp = requests.get(url, timeout=30, headers={"User-Agent": "team-skills/0.1"})
        resp.raise_for_status()
        code = resp.text
    except Exception as exc:
        raise SkillLoadError(f"failed to fetch remote skill {url!r}: {exc}") from exc
    if checksum:
        _verify_checksum(code.encode("utf-8"), checksum, url)
    return code


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def load_skill(
    source: str | dict,
) -> tuple[dict[str, ToolFn], dict[str, str], list[str]]:
    """Load a single skill from a registered name, local path, or remote URL.

    *source* can be:

    * A plain **string**:

      - If it starts with ``http://`` or ``https://``, it is fetched remotely.
      - If it has a file extension (``.py``, ``.md``) or contains a path
        separator, it is treated as a local file path.
      - Otherwise it is treated as a **registered skill name** resolved via the
        ``team.skills`` entry-point group (e.g. ``"bioblend"``).

    * A **dict** with one of ``name``, ``path``, or ``url`` key:

      - ``name`` — registered skill name (same as the plain-string name form).
      - ``path`` — local file path, with optional ``checksum`` key.
      - ``url``  — remote URL, with optional ``checksum`` key.

    Returns ``(tools_dict, descriptions_dict, injected_context)``.
    *injected_context* is a (possibly empty) list of strings to be prepended
    to the member's system prompt.  Markdown files (``.md``) produce only an
    injected context entry (no callable tools); Python files may produce both.

    Raises :class:`SkillLoadError` on any error.
    """
    if isinstance(source, str):
        is_url = source.startswith("http://") or source.startswith("https://")
        if is_url:
            code = _load_remote(source)
            return _exec_skill_code(code, source)

        suffix = Path(source).expanduser().suffix.lower()
        has_path_chars = "/" in source or "\\" in source or suffix in (".py", ".md")
        if has_path_chars:
            # Local file path.
            if suffix == ".md":
                content = _load_local(source)
                return {}, {}, ([content] if content.strip() else [])
            code = _load_local(source)
            return _exec_skill_code(code, source)

        # Plain name with no path chars — resolve via registry.
        return _load_skill_by_name(source)

    elif isinstance(source, dict):
        exclusive = [k for k in ("name", "path", "url") if k in source]
        if len(exclusive) > 1:
            raise SkillLoadError(
                f"skill entry must have exactly one of 'name', 'path', or 'url' "
                f"(got {exclusive})"
            )
        checksum = source.get("checksum")

        if "name" in source:
            if checksum:
                log.warning(
                    "checksum is not supported for registered skill names; ignoring"
                )
            return _load_skill_by_name(source["name"])

        if "url" in source:
            code = _load_remote(source["url"], checksum=checksum)
            return _exec_skill_code(code, source["url"])

        if "path" in source:
            path = source["path"]
            if Path(path).expanduser().suffix.lower() == ".md":
                content = _load_local(path, checksum=checksum)
                return {}, {}, ([content] if content.strip() else [])
            code = _load_local(path, checksum=checksum)
            return _exec_skill_code(code, path)

        raise SkillLoadError("skill entry dict must have 'name', 'path', or 'url'")

    else:
        raise SkillLoadError(
            f"skill source must be a string or dict, got {type(source).__name__!r}"
        )


def load_skills(
    sources: list[str | dict],
) -> tuple[dict[str, ToolFn], dict[str, str], list[str]]:
    """Load multiple skills and merge their tools into a single registry.

    Returns ``(merged_tools, merged_descriptions, merged_injected_context)``.
    Later entries override earlier ones on name collision (a warning is logged).
    Errors for individual skills are logged and skipped rather than aborting
    the whole load, so one bad skill does not prevent the others from loading.
    """
    merged_tools: dict[str, ToolFn] = {}
    merged_descs: dict[str, str] = {}
    merged_context: list[str] = []
    for source in sources:
        try:
            tools, descs, injected = load_skill(source)
        except SkillLoadError as exc:
            log.error("skipping skill %r: %s", source, exc)
            continue
        for name in tools:
            if name in merged_tools:
                log.warning(
                    "skill %r overrides existing tool %r", source, name
                )
        merged_tools.update(tools)
        merged_descs.update(descs)
        merged_context.extend(injected)
    return merged_tools, merged_descs, merged_context
