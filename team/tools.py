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
- ``write_file``  — write (create or overwrite) a file in the shared workspace.
- ``append_file`` — append text to a file in the shared workspace.
- ``list_files``  — list files in the shared workspace with optional glob filter.

Security note
-------------
``run_python`` and ``run_bash`` execute code on the **host machine** with the
privileges of the process that runs ``team``.  Only enable these tools for
members whose prompts you trust, and always review the generated code before
a run when security matters.
"""

from __future__ import annotations

import fnmatch
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


def _split_path_content(body: str) -> tuple[str | None, str | None]:
    """Extract ``path`` and content from a write/append tool body.

    Expected format::

        path: relative/path.txt
        ---
        File content goes here.
        Multiple lines are fine.

    The ``---`` separator line marks the boundary between the header and the
    file content.  Everything after the first ``\\n---\\n`` is treated as the
    raw content to write.
    """
    path = _parse_kv(body, "path")
    idx = body.find("\n---\n")
    content = body[idx + 5:] if idx != -1 else None
    return path, content


def _write_file(
    body: str,
    *,
    workspace_path: Path | None = None,
    **_: Any,
) -> str:
    """Write (create or overwrite) a file in the shared workspace.

    Body format::

        path: relative/path.txt
        ---
        File content goes here.

    The ``---`` line separates the path header from the content.
    The file and any parent directories are created automatically.
    Existing content is **replaced**.
    """
    if workspace_path is None:
        return "ERROR: no workspace available"
    rel, content = _split_path_content(body)
    if not rel:
        return "ERROR: provide path: <relative path> on the first line, then --- then content"
    if content is None:
        return "ERROR: missing --- separator between path and content"
    try:
        target = (workspace_path / rel).resolve()
        target.relative_to(workspace_path.resolve())  # traversal guard
    except ValueError:
        return f"ERROR: path {rel!r} escapes the workspace"
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"wrote {len(content)} chars to {rel}"
    except Exception as exc:  # noqa: BLE001
        return f"ERROR writing file: {exc}"


def _append_file(
    body: str,
    *,
    workspace_path: Path | None = None,
    **_: Any,
) -> str:
    """Append text to a file in the shared workspace.

    Body format::

        path: relative/path.txt
        ---
        Text to append.

    If the file does not exist it is created.  A newline is **not**
    automatically inserted before the appended text — include one in the
    content if needed.
    """
    if workspace_path is None:
        return "ERROR: no workspace available"
    rel, content = _split_path_content(body)
    if not rel:
        return "ERROR: provide path: <relative path> on the first line, then --- then content"
    if content is None:
        return "ERROR: missing --- separator between path and content"
    try:
        target = (workspace_path / rel).resolve()
        target.relative_to(workspace_path.resolve())  # traversal guard
    except ValueError:
        return f"ERROR: path {rel!r} escapes the workspace"
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as fh:
            fh.write(content)
        return f"appended {len(content)} chars to {rel}"
    except Exception as exc:  # noqa: BLE001
        return f"ERROR appending to file: {exc}"


def _list_files(
    body: str,
    *,
    workspace_path: Path | None = None,
    **_: Any,
) -> str:
    """List files in the shared workspace, optionally filtered by a glob pattern.

    Body format (all optional)::

        pattern: **/*.py

    If no pattern is provided (or the body is empty) all files are listed.
    Returns a newline-separated list of relative paths, or a message when no
    files are found.
    """
    if workspace_path is None:
        return "ERROR: no workspace available"
    pattern = _parse_kv(body, "pattern") or body.strip() or ""

    if not workspace_path.is_dir():
        return "(workspace is empty)"

    try:
        all_files = sorted(
            str(p.relative_to(workspace_path))
            for p in workspace_path.rglob("*")
            if p.is_file()
        )
    except Exception as exc:  # noqa: BLE001
        return f"ERROR listing workspace: {exc}"

    if not all_files:
        return "(workspace is empty)"

    if pattern:
        # Use Path.match() which properly handles ** glob patterns.
        matched = [f for f in all_files if Path(f).match(pattern)]
        if not matched:
            return f"(no files match pattern {pattern!r})"
    else:
        matched = all_files
    return _truncate("\n".join(matched))


def _delegate_task(
    body: str,
    *,
    workspace_path: Path | None = None,
    timeout: int = 600,
    **_: Any,
) -> str:
    """Delegate a sub-task to a remote team cluster and return its results.

    This is the core *inter-team collaboration* tool.  It submits a task to
    a remote ``team serve`` endpoint, waits for the remote team to complete
    its full workflow, then writes any files the remote team produced into the
    local shared workspace and returns a summary.

    Body format::

        url: http://lab-b.example.com:7001
        goal: Run the survival analysis on the preprocessed BRCA dataset.
        context: Data is in data/preprocessed.csv (1 142 samples, 38 features).
        files: data/preprocessed.csv, data/metadata.json
        timeout: 600

    All fields except ``url`` and ``goal`` are optional.

    * ``context`` — free-text background passed to the remote team.
    * ``files``   — comma-separated relative paths of workspace files to
      send with the task.  The remote team receives them in its own shared
      workspace before its workflow starts.
    * ``timeout`` — seconds to wait for the remote team to finish
      (overrides the tool-level timeout; default: 600).

    Returns a text summary of what the remote team accomplished, followed by
    a list of files it returned.  Any returned files are written into the
    local workspace so subsequent tool calls (``read_file``, etc.) can access
    them immediately.
    """
    from team.bridge import BridgeTask
    from team.bridge_client import BridgeClient, BridgeClientError

    url = _parse_kv(body, "url")
    if not url:
        return "ERROR: provide url: <remote bridge server URL>"
    goal = _parse_kv(body, "goal")
    if not goal:
        return "ERROR: provide goal: <what the remote team should accomplish>"

    context = _parse_kv(body, "context") or ""
    files_str = _parse_kv(body, "files") or ""
    task_timeout_str = _parse_kv(body, "timeout")
    task_timeout = float(task_timeout_str) if task_timeout_str else float(timeout)

    # Read requested local files to embed in the task.
    input_files: dict[str, str] = {}
    if files_str and workspace_path:
        for rel in [f.strip() for f in files_str.split(",") if f.strip()]:
            try:
                target = (workspace_path / rel).resolve()
                target.relative_to(workspace_path.resolve())
                if target.is_file():
                    input_files[rel] = target.read_text(encoding="utf-8", errors="replace")
                else:
                    log.warning("delegate_task: file %r not found, skipping", rel)
            except (ValueError, OSError) as exc:
                log.warning("delegate_task: skipping %r: %s", rel, exc)

    task = BridgeTask(
        goal=goal,
        context=context,
        files=input_files,
        sender="local-team",
    )

    client = BridgeClient(url)
    log.info("delegate_task: submitting task to %s (goal: %.60s…)", url, goal)
    try:
        task_id = client.submit_task(task)
        result = client.wait_for_result(task_id, timeout=task_timeout)
    except BridgeClientError as exc:
        return f"ERROR: bridge communication failed: {exc}"

    if result.status == "error":
        return f"ERROR: remote team failed: {result.error}"

    # Write remote files into the local workspace.
    written: list[str] = []
    if workspace_path and result.files:
        for rel, content in result.files.items():
            try:
                target = (workspace_path / rel).resolve()
                target.relative_to(workspace_path.resolve())
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
                written.append(rel)
            except (ValueError, OSError) as exc:
                log.warning("delegate_task: could not write %r: %s", rel, exc)

    parts = [f"Remote team completed the task.\n\nSummary:\n{result.summary}"]
    if written:
        parts.append(f"\nFiles received from remote team ({len(written)}):")
        parts.extend(f"  - {p}" for p in written)
    return _truncate("\n".join(parts))


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
    "write_file": _write_file,
    "append_file": _append_file,
    "list_files": _list_files,
    "delegate_task": _delegate_task,
}

#: Human-readable one-line description of each tool (used in system prompts).
TOOL_DESCRIPTIONS: dict[str, str] = {
    "run_python":    "Execute Python code; cwd is the shared workspace.",
    "run_bash":      "Execute a bash command; cwd is the shared workspace.",
    "web_search":    "Search the web via DuckDuckGo instant answers.",
    "read_url":      "Fetch and return the text content of a URL.",
    "read_file":     "Read a file from the shared workspace by relative path.",
    "write_file":    "Write (create or overwrite) a file in the shared workspace.",
    "append_file":   "Append text to a file in the shared workspace.",
    "list_files":    "List files in the shared workspace with optional glob filter.",
    "delegate_task": (
        "Delegate a sub-task to a remote team cluster (team serve) and wait "
        "for its results; files produced by the remote team are written into "
        "the local workspace automatically."
    ),
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
