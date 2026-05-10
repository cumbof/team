"""Skill loader — extend the agent toolbox with custom tools (F4 extension).

A *skill* is a plain Python file that exports one or more tool callables.
Skills can be loaded from:

* **Local files** — any path on the host filesystem (trusted by default).
* **Remote URLs** (``http://`` / ``https://``) — fetched via ``requests``.

  .. warning::
     Remote skill files execute **arbitrary code on the host** with the
     privileges of the ``team`` process.  Treat a remote skill URL with the
     same caution as ``curl URL | python``.

Skill file format
-----------------
A skill file must expose its tools in **one of two formats** (or both):

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

Config example
--------------
::

    defaults:
      skills:
        - path: ./skills/my_tool.py          # local file (relative to CWD)
        - url: https://example.com/tool.py   # remote (loaded with a warning)
          checksum: sha256:abc123...          # optional integrity check
        - ./skills/another.py                # plain string = local path
        - https://raw.githubusercontent.com/org/repo/main/skill.py

    members:
      - name: researcher
        tools: [web_search, my_tool]          # built-ins AND skill tools by name
        skills:
          - ./skills/domain_search.py         # member-specific skill

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
import logging
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger(__name__)

#: Type alias for a tool callable.
ToolFn = Callable[..., str]


class SkillLoadError(Exception):
    """Raised when a skill cannot be loaded or does not export valid tools."""


# --------------------------------------------------------------------------- #
# Low-level execution helpers
# --------------------------------------------------------------------------- #


def _exec_skill_code(
    code: str,
    source_label: str,
) -> tuple[dict[str, ToolFn], dict[str, str]]:
    """Execute *code* in an isolated namespace and extract tool definitions.

    Returns ``(tools_dict, descriptions_dict)``.
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

    log.debug("loaded skill %r: tools=%s", source_label, list(tools))
    return tools, descriptions


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
) -> tuple[dict[str, ToolFn], dict[str, str]]:
    """Load a single skill from a local path or remote URL.

    *source* can be:

    * A plain **string**: treated as a local path unless it starts with
      ``http://`` or ``https://``, in which case it is fetched remotely.
    * A **dict** with either ``path`` or ``url`` key, and optional
      ``checksum`` key.

    Returns ``(tools_dict, descriptions_dict)``.
    Raises :class:`SkillLoadError` on any error.
    """
    if isinstance(source, str):
        is_url = source.startswith("http://") or source.startswith("https://")
        if is_url:
            code = _load_remote(source)
        else:
            code = _load_local(source)
        label = source
    elif isinstance(source, dict):
        if "path" in source and "url" in source:
            raise SkillLoadError(
                "skill entry must have 'path' OR 'url', not both"
            )
        checksum = source.get("checksum")
        if "url" in source:
            code = _load_remote(source["url"], checksum=checksum)
            label = source["url"]
        elif "path" in source:
            code = _load_local(source["path"], checksum=checksum)
            label = source["path"]
        else:
            raise SkillLoadError("skill entry dict must have 'path' or 'url'")
    else:
        raise SkillLoadError(
            f"skill source must be a string or dict, got {type(source).__name__!r}"
        )
    return _exec_skill_code(code, label)


def load_skills(
    sources: list[str | dict],
) -> tuple[dict[str, ToolFn], dict[str, str]]:
    """Load multiple skills and merge their tools into a single registry.

    Returns ``(merged_tools, merged_descriptions)``.
    Later entries override earlier ones on name collision (a warning is logged).
    Errors for individual skills are logged and skipped rather than aborting
    the whole load, so one bad skill does not prevent the others from loading.
    """
    merged_tools: dict[str, ToolFn] = {}
    merged_descs: dict[str, str] = {}
    for source in sources:
        try:
            tools, descs = load_skill(source)
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
    return merged_tools, merged_descs
