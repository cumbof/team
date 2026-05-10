"""Built-in tools for agentic members (F4).

Each tool is invoked when a member emits a fenced ``tool:`` block in its reply::

    ```tool:run_python
    import pandas as pd
    df = pd.read_csv('/workspace/data.csv')
    print(df.describe())
    ```

    ```tool:web_search
    query: IPCC AR6 key findings 2023
    ```

The orchestrator parses these blocks, executes the corresponding tool,
and injects the results back to the member for a follow-up LLM call — all
within the same logical "turn" so the transcript only sees the final reply.

Available tools
---------------
- ``run_python``  — execute Python code; cwd = shared workspace.
- ``run_bash``    — execute a bash command; cwd = shared workspace.
- ``web_search``  — query DuckDuckGo instant answers.
- ``read_url``    — fetch and extract text from a URL.
- ``read_file``   — read a file from the shared workspace by relative path.

Security note
-------------
``run_python`` and ``run_bash`` execute code on the **host machine** with the
privileges of the process that runs ``team``.  Only enable these tools for
members whose prompts you trust, and always review the generated code before
a run when security matters.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import tempfile
import urllib.parse
from pathlib import Path
from typing import Any

import requests

log = logging.getLogger(__name__)

# Maximum characters returned by any tool (prevents overwhelming the context).
_MAX_OUTPUT = 8192
_MAX_SEARCH_OUTPUT = 4096


# --------------------------------------------------------------------------- #
# Block parsing
# --------------------------------------------------------------------------- #

_TOOL_BLOCK_RE = re.compile(
    r"```tool:(?P<name>[a-z_]+)\n(?P<body>.*?)```",
    re.DOTALL,
)


def parse_tool_blocks(text: str) -> list[tuple[str, str]]:
    """Return ``(tool_name, body)`` pairs for every ``tool:`` block in *text*."""
    return [
        (m.group("name"), m.group("body").strip())
        for m in _TOOL_BLOCK_RE.finditer(text)
    ]


def _parse_kv(body: str, key: str) -> str | None:
    """Extract a ``key: value`` line from the body; returns the value or None."""
    for line in body.splitlines():
        line = line.strip()
        if line.lower().startswith(f"{key}:"):
            return line[len(key) + 1:].strip()
    return None


def _truncate(text: str, limit: int = _MAX_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n[… truncated at {limit} chars]"


# --------------------------------------------------------------------------- #
# Individual tool implementations
# --------------------------------------------------------------------------- #


def _run_python(
    body: str,
    *,
    workspace_path: Path | None = None,
    timeout: int = 30,
    **_: Any,
) -> str:
    """Execute *body* as Python code and return stdout + stderr."""
    code = body.strip()
    env = os.environ.copy()
    cwd = str(workspace_path) if workspace_path and workspace_path.is_dir() else None
    if cwd:
        env["WORKSPACE"] = cwd

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as fh:
        fh.write(code)
        script = fh.name

    try:
        result = subprocess.run(
            ["python", script],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            cwd=cwd,
        )
        out = result.stdout
        if result.stderr:
            out += ("\n" if out else "") + f"STDERR:\n{result.stderr}"
        return _truncate(out or "(no output)")
    except subprocess.TimeoutExpired:
        return f"ERROR: execution timed out after {timeout}s"
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: {exc}"
    finally:
        try:
            os.unlink(script)
        except OSError:
            pass


def _run_bash(
    body: str,
    *,
    workspace_path: Path | None = None,
    timeout: int = 30,
    **_: Any,
) -> str:
    """Execute *body* as a bash command and return stdout + stderr."""
    cmd = body.strip()
    cwd = str(workspace_path) if workspace_path and workspace_path.is_dir() else None
    try:
        result = subprocess.run(
            ["bash", "-c", cmd],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
        out = result.stdout
        if result.stderr:
            out += ("\n" if out else "") + f"STDERR:\n{result.stderr}"
        return _truncate(out or "(no output)")
    except subprocess.TimeoutExpired:
        return f"ERROR: execution timed out after {timeout}s"
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: {exc}"


def _web_search(body: str, *, timeout: int = 15, **_: Any) -> str:
    """Search via the DuckDuckGo Instant Answer API (no API key required)."""
    query = _parse_kv(body, "query") or body.strip()
    if not query:
        return "ERROR: provide query: <search terms>"

    url = "https://api.duckduckgo.com/"
    params = {
        "q": query,
        "format": "json",
        "no_html": "1",
        "skip_disambig": "1",
    }
    try:
        r = requests.get(
            url,
            params=params,
            timeout=timeout,
            headers={"User-Agent": "team-agent/0.1"},
        )
        r.raise_for_status()
        data = r.json()
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: web_search failed: {exc}"

    parts: list[str] = []
    if data.get("Heading"):
        parts.append(f"**{data['Heading']}**")
    if data.get("AbstractText"):
        parts.append(data["AbstractText"])
    if data.get("Answer"):
        parts.append(f"Answer: {data['Answer']}")
    for topic in data.get("RelatedTopics", [])[:8]:
        if isinstance(topic, dict) and topic.get("Text"):
            parts.append(f"- {topic['Text']}")
    if not parts:
        return (
            f"No instant answer found for: {query!r}\n"
            "Tip: DuckDuckGo instant answers work best for factual queries. "
            "For broader results, try read_url with a specific page URL."
        )
    return _truncate("\n".join(parts), _MAX_SEARCH_OUTPUT)


def _read_url(body: str, *, timeout: int = 15, **_: Any) -> str:
    """Fetch *url* and return its text content (HTML tags stripped)."""
    url = _parse_kv(body, "url") or body.strip()
    if not url:
        return "ERROR: provide url: <url>"
    try:
        r = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "team-agent/0.1"},
        )
        r.raise_for_status()
        content_type = r.headers.get("Content-Type", "")
        if "text/html" in content_type:
            text = re.sub(r"<style[^>]*>.*?</style>", " ", r.text, flags=re.DOTALL)
            text = re.sub(r"<script[^>]*>.*?</script>", " ", text, flags=re.DOTALL)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"&[a-z]+;", " ", text)
            text = re.sub(r"\s{2,}", " ", text).strip()
        else:
            text = r.text
        return _truncate(text)
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: read_url failed: {exc}"


def _read_file(
    body: str,
    *,
    workspace_path: Path | None = None,
    **_: Any,
) -> str:
    """Read a file from the shared workspace by relative path."""
    rel = _parse_kv(body, "path") or body.strip()
    if not rel:
        return "ERROR: provide path: <relative path>"
    if workspace_path is None:
        return "ERROR: no workspace available"
    try:
        target = (workspace_path / rel).resolve()
        workspace_path.resolve()
        target.relative_to(workspace_path.resolve())  # traversal guard
    except ValueError:
        return f"ERROR: path {rel!r} escapes the workspace"
    if not target.is_file():
        return f"ERROR: {rel!r} not found in shared workspace"
    try:
        return _truncate(target.read_text(encoding="utf-8", errors="replace"))
    except Exception as exc:  # noqa: BLE001
        return f"ERROR reading file: {exc}"


# --------------------------------------------------------------------------- #
# Registry & dispatch
# --------------------------------------------------------------------------- #

#: All built-in tools, keyed by name.  Members opt-in via ``tools:`` config.
TOOLS: dict[str, Any] = {
    "run_python": _run_python,
    "run_bash": _run_bash,
    "web_search": _web_search,
    "read_url": _read_url,
    "read_file": _read_file,
}

#: Human-readable one-line description of each tool (used in system prompts).
TOOL_DESCRIPTIONS: dict[str, str] = {
    "run_python": "Execute Python code; cwd is the shared workspace.",
    "run_bash":   "Execute a bash command; cwd is the shared workspace.",
    "web_search": "Search the web via DuckDuckGo instant answers.",
    "read_url":   "Fetch and return the text content of a URL.",
    "read_file":  "Read a file from the shared workspace by relative path.",
}


def execute_tool(
    name: str,
    body: str,
    *,
    workspace_path: Path | None = None,
    timeout: int = 30,
    tools: dict | None = None,
) -> str:
    """Execute the named tool and return its string output.

    Parameters
    ----------
    name:
        Tool name to execute.
    body:
        Body text passed to the tool callable.
    workspace_path:
        Shared workspace directory, forwarded to the tool.
    timeout:
        Per-tool execution timeout in seconds.
    tools:
        Optional tool registry override.  Defaults to the built-in
        :data:`TOOLS` dict.  Pass a merged built-ins + skills dict to
        support custom skill tools.

    Raises :class:`KeyError` if *name* is not in the registry.
    All exceptions from the tool implementation are caught and returned
    as error strings so a single bad tool call does not abort the turn.
    """
    registry = tools if tools is not None else TOOLS
    fn = registry[name]
    log.info("tool:%s executing", name)
    result = fn(body, workspace_path=workspace_path, timeout=timeout)
    log.debug("tool:%s → %d chars", name, len(result))
    return result
