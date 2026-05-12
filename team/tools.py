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
- ``write_file``    — write (create or overwrite) a file in the shared workspace.
- ``append_file``   — append text to a file in the shared workspace.
- ``list_files``    — list files in the shared workspace with optional glob filter.
- ``remember``      — store a memory in the member's persistent cross-session memory store.
- ``recall``        — search the member's persistent memory by keyword.
- ``forget``        — delete a memory by key.
- ``list_memories`` — list all memories (optionally filtered by tag).
- ``assert_belief``   — add a claim to the team's shared belief board.
- ``contest_belief``  — contest an existing belief (moves it to contested status).
- ``accept_belief``   — cast an accept vote for an existing belief.
- ``list_beliefs``    — list the team's belief board (optionally by status).
- ``log_decision``    — append a timestamped decision entry to decisions.md in the shared workspace.
- ``read_decisions``  — read the full decisions log (decisions.md) from the shared workspace.

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


# --------------------------------------------------------------------------- #
# Memory tools (per-agent persistent cross-session memory)
# --------------------------------------------------------------------------- #


def _remember(
    body: str,
    *,
    memory: Any = None,
    **_: Any,
) -> str:
    """Store a memory in the member's persistent cross-session memory store.

    Body format::

        key: experiment_baseline_2024
        tags: results, chemistry
        importance: 0.9
        ---
        AlphaFold3 achieved RMSD 1.2 Å vs RoseTTAFold 2.1 Å on 1 000 monomers.

    The ``---`` separator marks the boundary between the header fields and the
    memory value.  The value may span multiple lines.  ``tags`` and
    ``importance`` are optional (defaults: no tags, importance = 1.0).

    If a memory with the same key already exists it is **updated** in place.
    """
    if memory is None:
        return "ERROR: memory is not enabled for this member (set memory.enabled: true)"
    key = _parse_kv(body, "key")
    if not key:
        return "ERROR: provide key: <memory key>"
    tags_raw = _parse_kv(body, "tags") or ""
    tags = [t.strip() for t in tags_raw.split(",") if t.strip()] if tags_raw else []
    importance_raw = _parse_kv(body, "importance")
    try:
        importance = float(importance_raw) if importance_raw else 1.0
    except ValueError:
        return f"ERROR: importance must be a float, got {importance_raw!r}"
    idx = body.find("\n---\n")
    if idx == -1:
        return "ERROR: missing --- separator between header and memory value"
    value = body[idx + 5:].strip()
    if not value:
        return "ERROR: memory value (after ---) must not be empty"
    return memory.remember(key, value, tags=tags, importance=importance)


def _recall(
    body: str,
    *,
    memory: Any = None,
    **_: Any,
) -> str:
    """Search the member's persistent memory by keyword.

    Body format::

        query: protein folding
        limit: 5

    Returns a Markdown list of matching memories (key, tags, value).
    ``limit`` defaults to 5.
    """
    if memory is None:
        return "ERROR: memory is not enabled for this member (set memory.enabled: true)"
    query = _parse_kv(body, "query") or body.strip()
    if not query:
        return "ERROR: provide query: <search terms>"
    limit_raw = _parse_kv(body, "limit")
    try:
        limit = int(limit_raw) if limit_raw else 5
    except ValueError:
        limit = 5
    results = memory.recall(query, limit=limit)
    if not results:
        return f"No memories found matching {query!r}."
    lines = [f"Found {len(results)} memory(ies) matching {query!r}:"]
    for m in results:
        tags_part = f" [{m['tags']}]" if m["tags"] else ""
        lines.append(f"- **{m['key']}** (imp: {m['importance']:.1f}){tags_part}: {m['value']}")
    return _truncate("\n".join(lines))


def _forget(
    body: str,
    *,
    memory: Any = None,
    **_: Any,
) -> str:
    """Delete a memory by key.

    Body format::

        key: experiment_baseline_2024

    Returns a confirmation or an error if the key was not found.
    """
    if memory is None:
        return "ERROR: memory is not enabled for this member (set memory.enabled: true)"
    key = _parse_kv(body, "key") or body.strip()
    if not key:
        return "ERROR: provide key: <memory key>"
    deleted = memory.forget(key)
    return f"Deleted memory: {key!r}" if deleted else f"No memory found with key: {key!r}"


def _list_memories(
    body: str,
    *,
    memory: Any = None,
    **_: Any,
) -> str:
    """List all memories in the member's persistent store.

    Body format (all optional)::

        tag: results
        limit: 20

    Returns a Markdown list ordered by importance then recency.
    ``limit`` defaults to 20.
    """
    if memory is None:
        return "ERROR: memory is not enabled for this member (set memory.enabled: true)"
    tag = _parse_kv(body, "tag") or None
    limit_raw = _parse_kv(body, "limit")
    try:
        limit = int(limit_raw) if limit_raw else 20
    except ValueError:
        limit = 20
    entries = memory.list_memories(tag=tag, limit=limit)
    if not entries:
        msg = f"No memories found with tag {tag!r}." if tag else "No memories stored yet."
        return msg
    lines = [f"{len(entries)} memory(ies)" + (f" tagged {tag!r}" if tag else "") + ":"]
    for m in entries:
        tags_part = f" [{m['tags']}]" if m["tags"] else ""
        lines.append(f"- [{m['id']}] **{m['key']}** (imp: {m['importance']:.1f}){tags_part}: {m['value']}")
    return _truncate("\n".join(lines))


# --------------------------------------------------------------------------- #
# Belief-board tools (shared team collective knowledge)
# --------------------------------------------------------------------------- #


def _assert_belief(
    body: str,
    *,
    beliefs: Any = None,
    member_name: str = "unknown",
    **_: Any,
) -> str:
    """Add a claim to the team's shared belief board.

    Body format::

        confidence: 0.85
        evidence: RMSD analysis on n=1000, peer-reviewed dataset
        ---
        AlphaFold3 is the best available method for monomer structure prediction.

    The ``---`` separator marks the boundary between header fields and the
    claim text.  ``confidence`` (0.0 – 1.0) and ``evidence`` are optional.
    The member who asserts a belief automatically casts an *accept* vote.
    """
    if beliefs is None:
        return "ERROR: beliefs are not enabled (set beliefs.enabled: true)"
    confidence_raw = _parse_kv(body, "confidence")
    try:
        confidence = float(confidence_raw) if confidence_raw else 0.5
        confidence = max(0.0, min(1.0, confidence))
    except ValueError:
        return f"ERROR: confidence must be a float 0–1, got {confidence_raw!r}"
    evidence = _parse_kv(body, "evidence") or ""
    idx = body.find("\n---\n")
    if idx == -1:
        return "ERROR: missing --- separator between header and claim text"
    claim = body[idx + 5:].strip()
    if not claim:
        return "ERROR: claim text (after ---) must not be empty"
    b = beliefs.assert_belief(claim, author=member_name, confidence=confidence, evidence=evidence)
    return (
        f"Belief [{b.id}] asserted (status: {b.status}):\n"
        f"  Claim: {b.claim}\n"
        f"  Confidence: {b.confidence:.0%}"
    )


def _contest_belief(
    body: str,
    *,
    beliefs: Any = None,
    member_name: str = "unknown",
    **_: Any,
) -> str:
    """Contest an existing belief, moving it to 'contested' status.

    Body format::

        id: abc12345
        reason: The dataset is too small (n<100) to support this claim.

    ``reason`` is optional but strongly recommended so the team can understand
    the objection.
    """
    if beliefs is None:
        return "ERROR: beliefs are not enabled (set beliefs.enabled: true)"
    belief_id = _parse_kv(body, "id") or body.strip()
    if not belief_id:
        return "ERROR: provide id: <belief id>"
    reason = _parse_kv(body, "reason") or ""
    try:
        b = beliefs.contest_belief(belief_id, voter=member_name, reason=reason)
    except KeyError:
        return f"ERROR: no belief found with id {belief_id!r}"
    return (
        f"Belief [{b.id}] is now contested.\n"
        f"  Claim: {b.claim}\n"
        f"  Reason given: {reason or '(none)'}"
    )


def _accept_belief(
    body: str,
    *,
    beliefs: Any = None,
    member_name: str = "unknown",
    **_: Any,
) -> str:
    """Cast an accept vote for an existing belief.

    Body format::

        id: abc12345

    If enough members accept, the belief transitions to 'accepted'.
    """
    if beliefs is None:
        return "ERROR: beliefs are not enabled (set beliefs.enabled: true)"
    belief_id = _parse_kv(body, "id") or body.strip()
    if not belief_id:
        return "ERROR: provide id: <belief id>"
    try:
        b = beliefs.accept_belief(belief_id, voter=member_name)
    except KeyError:
        return f"ERROR: no belief found with id {belief_id!r}"
    return (
        f"Voted to accept belief [{b.id}] (status: {b.status}).\n"
        f"  Votes for: {len(b.votes_for)}, against: {len(b.votes_against)}"
    )


def _list_beliefs(
    body: str,
    *,
    beliefs: Any = None,
    **_: Any,
) -> str:
    """List the team's belief board.

    Body format (all optional)::

        status: pending

    Valid status values: ``pending``, ``accepted``, ``contested``, ``rejected``.
    Omit to list all beliefs.
    """
    if beliefs is None:
        return "ERROR: beliefs are not enabled (set beliefs.enabled: true)"
    status = _parse_kv(body, "status") or None
    if status and status not in ("pending", "accepted", "contested", "rejected"):
        return f"ERROR: unknown status {status!r}; use pending|accepted|contested|rejected"
    items = beliefs.list_beliefs(status=status)
    if not items:
        msg = f"No beliefs with status {status!r}." if status else "The belief board is empty."
        return msg
    _ICONS = {"accepted": "✓", "contested": "⚡", "rejected": "✗", "pending": "?"}
    lines = [f"Belief board — {len(items)} belief(s)" + (f" [status={status}]" if status else "") + ":"]
    for b in items:
        icon = _ICONS.get(b.status, "?")
        conf = f"{b.confidence:.0%}"
        lines.append(
            f"  [{icon}] `{b.id}` {b.claim}\n"
            f"       conf={conf}, by @{b.author}, "
            f"for={len(b.votes_for)}, against={len(b.votes_against)}, status={b.status}"
        )
    return _truncate("\n".join(lines))


def _delegate_task(
    body: str,
    *,
    workspace_path: Path | None = None,
    timeout: int = 600,
    bridge_secret: str | None = None,
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

    client = BridgeClient(url, secret=bridge_secret)
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
# Decision log tools (shared team decision record)
# --------------------------------------------------------------------------- #

_DECISIONS_FILE = "decisions.md"


def _log_decision(
    body: str,
    *,
    workspace_path: Path | None = None,
    member_name: str = "unknown",
    **_: Any,
) -> str:
    """Append a timestamped decision entry to ``decisions.md`` in the shared workspace.

    Body format::

        title: Chose pandas over polars for data wrangling
        rationale: Polars ecosystem is too immature; pandas is already a dependency.
        alternatives: polars, dask, vaex

    ``rationale`` and ``alternatives`` are optional.  A formatted entry is
    appended to ``decisions.md`` (created on first use) so the decision
    history accumulates across the full run.
    """
    import datetime

    if workspace_path is None:
        return "ERROR: no workspace available"

    title = _parse_kv(body, "title") or body.strip()
    if not title:
        return "ERROR: provide title: <short decision title>"

    rationale = _parse_kv(body, "rationale") or ""
    alternatives = _parse_kv(body, "alternatives") or ""
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    lines = [
        f"## Decision: {title}",
        f"**Date:** {now}  ",
        f"**By:** @{member_name}  ",
    ]
    if rationale:
        lines += ["", f"**Rationale:** {rationale}"]
    if alternatives:
        lines += ["", f"**Alternatives considered:** {alternatives}"]
    lines += ["", "---", ""]

    entry = "\n".join(lines)
    target = (workspace_path / _DECISIONS_FILE).resolve()
    try:
        target.relative_to(workspace_path.resolve())
    except ValueError:
        return "ERROR: path traversal guard triggered for decisions.md"
    try:
        with target.open("a", encoding="utf-8") as fh:
            fh.write(entry)
        return f"Decision logged: {title!r} (appended to {_DECISIONS_FILE})"
    except Exception as exc:  # noqa: BLE001
        return f"ERROR writing decisions.md: {exc}"


def _read_decisions(
    body: str,
    *,
    workspace_path: Path | None = None,
    **_: Any,
) -> str:
    """Read the full decision log (``decisions.md``) from the shared workspace.

    No body parameters are required — the entire file is returned.
    Returns a helpful message when no decisions have been logged yet.
    """
    if workspace_path is None:
        return "ERROR: no workspace available"
    target = workspace_path / _DECISIONS_FILE
    if not target.is_file():
        return "No decisions have been logged yet (decisions.md does not exist)."
    try:
        return _truncate(target.read_text(encoding="utf-8", errors="replace"))
    except Exception as exc:  # noqa: BLE001
        return f"ERROR reading decisions.md: {exc}"


# --------------------------------------------------------------------------- #
# Native function-calling schemas
# --------------------------------------------------------------------------- #

def _fn(name: str, description: str, props: dict, required: list[str]) -> dict:
    """Build an OpenAI/Ollama-compatible function-tool dict."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": props,
                "required": required,
            },
        },
    }


#: OpenAI/Ollama-compatible JSON Schema definitions for every built-in tool.
#: Used when ``tool_mode: native`` is configured.
TOOL_SCHEMAS: dict[str, dict] = {
    "run_python": _fn(
        "run_python",
        "Execute Python code in the shared workspace directory and return stdout/stderr.",
        {"code": {"type": "string", "description": "Python source code to execute."}},
        ["code"],
    ),
    "run_bash": _fn(
        "run_bash",
        "Execute a bash command in the shared workspace directory and return stdout/stderr.",
        {"command": {"type": "string", "description": "Bash command to execute."}},
        ["command"],
    ),
    "web_search": _fn(
        "web_search",
        "Search the web via the DuckDuckGo instant-answer API and return a summary.",
        {"query": {"type": "string", "description": "Search terms."}},
        ["query"],
    ),
    "read_url": _fn(
        "read_url",
        "Fetch a URL and return its text content (HTML tags stripped).",
        {"url": {"type": "string", "description": "URL to fetch."}},
        ["url"],
    ),
    "read_file": _fn(
        "read_file",
        "Read a file from the shared workspace by relative path and return its contents.",
        {"path": {"type": "string", "description": "Relative path inside the shared workspace."}},
        ["path"],
    ),
    "write_file": _fn(
        "write_file",
        "Write (create or overwrite) a file in the shared workspace.",
        {
            "path":    {"type": "string", "description": "Relative path inside the shared workspace."},
            "content": {"type": "string", "description": "File content to write."},
        },
        ["path", "content"],
    ),
    "append_file": _fn(
        "append_file",
        "Append text to a file in the shared workspace (creates the file if it does not exist).",
        {
            "path":    {"type": "string", "description": "Relative path inside the shared workspace."},
            "content": {"type": "string", "description": "Text to append."},
        },
        ["path", "content"],
    ),
    "list_files": _fn(
        "list_files",
        "List files in the shared workspace, optionally filtered by a glob pattern.",
        {"pattern": {"type": "string", "description": "Glob pattern, e.g. **/*.py. Omit to list all."}},
        [],
    ),
    "remember": _fn(
        "remember",
        "Store a named memory in your persistent cross-session memory store.",
        {
            "key":        {"type": "string",  "description": "Unique memory key."},
            "value":      {"type": "string",  "description": "Value to store (multi-line text allowed)."},
            "tags":       {"type": "string",  "description": "Comma-separated tags (optional)."},
            "importance": {"type": "number",  "description": "Importance score 0–1 (default 1.0)."},
        },
        ["key", "value"],
    ),
    "recall": _fn(
        "recall",
        "Search your persistent memory by keyword.",
        {
            "query": {"type": "string",  "description": "Keywords to search for."},
            "limit": {"type": "integer", "description": "Maximum results to return (default 5)."},
        },
        ["query"],
    ),
    "forget": _fn(
        "forget",
        "Delete a memory from your persistent store by key.",
        {"key": {"type": "string", "description": "Key of the memory to delete."}},
        ["key"],
    ),
    "list_memories": _fn(
        "list_memories",
        "List memories in your persistent store, optionally filtered by tag.",
        {
            "tag":   {"type": "string",  "description": "Filter by tag (optional)."},
            "limit": {"type": "integer", "description": "Maximum entries to return (default 20)."},
        },
        [],
    ),
    "assert_belief": _fn(
        "assert_belief",
        "Add a claim to the team's shared belief board.",
        {
            "claim":      {"type": "string", "description": "The claim text."},
            "confidence": {"type": "number", "description": "Confidence 0–1 (default 0.5)."},
            "evidence":   {"type": "string", "description": "Supporting evidence (optional)."},
        },
        ["claim"],
    ),
    "contest_belief": _fn(
        "contest_belief",
        "Contest an existing team belief, moving it to contested status.",
        {
            "id":     {"type": "string", "description": "Belief ID to contest."},
            "reason": {"type": "string", "description": "Reason for contesting (optional but recommended)."},
        },
        ["id"],
    ),
    "accept_belief": _fn(
        "accept_belief",
        "Cast an accept vote for an existing team belief.",
        {"id": {"type": "string", "description": "Belief ID to accept."}},
        ["id"],
    ),
    "list_beliefs": _fn(
        "list_beliefs",
        "List the team's shared belief board, optionally filtered by status.",
        {
            "status": {
                "type": "string",
                "enum": ["pending", "accepted", "contested", "rejected"],
                "description": "Filter by status (optional, omit to list all).",
            }
        },
        [],
    ),
    "delegate_task": _fn(
        "delegate_task",
        "Delegate a sub-task to a remote team cluster (team serve) and wait for its results.",
        {
            "url":     {"type": "string",  "description": "Remote bridge server URL."},
            "goal":    {"type": "string",  "description": "What the remote team should accomplish."},
            "context": {"type": "string",  "description": "Optional background for the remote team."},
            "files":   {"type": "string",  "description": "Comma-separated relative paths of files to send."},
            "timeout": {"type": "integer", "description": "Seconds to wait for the remote team (default 600)."},
        },
        ["url", "goal"],
    ),
    "log_decision": _fn(
        "log_decision",
        "Append a timestamped decision entry to decisions.md in the shared workspace.",
        {
            "title":        {"type": "string", "description": "Short decision title."},
            "rationale":    {"type": "string", "description": "Why this decision was made (optional)."},
            "alternatives": {"type": "string", "description": "Alternatives considered (optional)."},
        },
        ["title"],
    ),
    "read_decisions": _fn(
        "read_decisions",
        "Read the full decision log (decisions.md) from the shared workspace.",
        {},
        [],
    ),
}


def args_to_body(tool_name: str, args: dict) -> str:
    """Convert a JSON argument dict (from native LLM function calling) to the
    text body format expected by the existing tool implementations.

    This lets the same tool functions handle both text-block invocations and
    native function-call invocations without duplication.
    """
    if tool_name in ("run_python",):
        return args.get("code", "")

    if tool_name in ("run_bash",):
        return args.get("command", "")

    if tool_name == "web_search":
        q = args.get("query", "")
        return f"query: {q}"

    if tool_name == "read_url":
        return f"url: {args.get('url', '')}"

    if tool_name == "read_file":
        return f"path: {args.get('path', '')}"

    if tool_name in ("write_file", "append_file"):
        path = args.get("path", "")
        content = args.get("content", "")
        return f"path: {path}\n---\n{content}"

    if tool_name == "list_files":
        pat = args.get("pattern", "")
        return f"pattern: {pat}" if pat else ""

    if tool_name == "remember":
        lines = [f"key: {args.get('key', '')}"]
        if args.get("tags"):
            lines.append(f"tags: {args['tags']}")
        if args.get("importance") is not None:
            lines.append(f"importance: {args['importance']}")
        lines.append("---")
        lines.append(args.get("value", ""))
        return "\n".join(lines)

    if tool_name == "recall":
        body = f"query: {args.get('query', '')}"
        if args.get("limit"):
            body += f"\nlimit: {args['limit']}"
        return body

    if tool_name == "forget":
        return f"key: {args.get('key', '')}"

    if tool_name == "list_memories":
        lines = []
        if args.get("tag"):
            lines.append(f"tag: {args['tag']}")
        if args.get("limit"):
            lines.append(f"limit: {args['limit']}")
        return "\n".join(lines)

    if tool_name == "assert_belief":
        lines = []
        if args.get("confidence") is not None:
            lines.append(f"confidence: {args['confidence']}")
        if args.get("evidence"):
            lines.append(f"evidence: {args['evidence']}")
        lines.append("---")
        lines.append(args.get("claim", ""))
        return "\n".join(lines)

    if tool_name == "contest_belief":
        body = f"id: {args.get('id', '')}"
        if args.get("reason"):
            body += f"\nreason: {args['reason']}"
        return body

    if tool_name == "accept_belief":
        return f"id: {args.get('id', '')}"

    if tool_name == "list_beliefs":
        status = args.get("status", "")
        return f"status: {status}" if status else ""

    if tool_name == "delegate_task":
        lines = [f"url: {args.get('url', '')}", f"goal: {args.get('goal', '')}"]
        if args.get("context"):
            lines.append(f"context: {args['context']}")
        if args.get("files"):
            lines.append(f"files: {args['files']}")
        if args.get("timeout"):
            lines.append(f"timeout: {args['timeout']}")
        return "\n".join(lines)

    if tool_name == "log_decision":
        lines = [f"title: {args.get('title', '')}"]
        if args.get("rationale"):
            lines.append(f"rationale: {args['rationale']}")
        if args.get("alternatives"):
            lines.append(f"alternatives: {args['alternatives']}")
        return "\n".join(lines)

    if tool_name == "read_decisions":
        return ""

    # Unknown tool or custom skill — pass args as JSON so the skill can parse them.
    import json as _json
    return _json.dumps(args, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# Registry & dispatch
# --------------------------------------------------------------------------- #

#: All built-in tools, keyed by name.  Members opt-in via ``tools:`` config.
TOOLS: dict[str, Any] = {
    "run_python":     _run_python,
    "run_bash":       _run_bash,
    "web_search":     _web_search,
    "read_url":       _read_url,
    "read_file":      _read_file,
    "write_file":     _write_file,
    "append_file":    _append_file,
    "list_files":     _list_files,
    "remember":       _remember,
    "recall":         _recall,
    "forget":         _forget,
    "list_memories":  _list_memories,
    "assert_belief":  _assert_belief,
    "contest_belief": _contest_belief,
    "accept_belief":  _accept_belief,
    "list_beliefs":   _list_beliefs,
    "delegate_task":  _delegate_task,
    "log_decision":   _log_decision,
    "read_decisions": _read_decisions,
}

#: Human-readable one-line description of each tool (used in system prompts).
TOOL_DESCRIPTIONS: dict[str, str] = {
    "run_python":     "Execute Python code with full system access; cwd is the shared workspace. You may import any installed library or install missing ones first with run_bash.",
    "run_bash":       "Execute any bash command with full system access; cwd is the shared workspace. Use this to install packages (pip install, apt-get install), run CLIs, inspect the system, etc.",
    "web_search":     "Search the web via DuckDuckGo instant answers.",
    "read_url":       "Fetch and return the text content of a URL.",
    "read_file":      "Read a file from the shared workspace by relative path.",
    "write_file":     "Write (create or overwrite) a file in the shared workspace.",
    "append_file":    "Append text to a file in the shared workspace.",
    "list_files":     "List files in the shared workspace with optional glob filter.",
    "remember":       "Store a memory in your persistent cross-session memory store (key + multi-line value).",
    "recall":         "Search your persistent memory by keyword; returns matching entries.",
    "forget":         "Delete a memory by key from your persistent store.",
    "list_memories":  "List your stored memories, optionally filtered by tag.",
    "assert_belief":  "Add a claim to the team's shared belief board with confidence score.",
    "contest_belief": "Contest an existing team belief (moves it to contested status).",
    "accept_belief":  "Cast an accept vote for an existing team belief.",
    "list_beliefs":   "List the team's shared belief board, optionally filtered by status.",
    "delegate_task":  (
        "Delegate a sub-task to a remote team cluster (team serve) and wait "
        "for its results; files produced by the remote team are written into "
        "the local workspace automatically."
    ),
    "log_decision":   (
        "Append a timestamped decision entry to decisions.md in the shared workspace. "
        "Provide title, rationale, and alternatives considered."
    ),
    "read_decisions": "Read the full decision log (decisions.md) from the shared workspace.",
}


def execute_tool(
    name: str,
    body: str,
    *,
    workspace_path: Path | None = None,
    timeout: int = 30,
    tools: dict | None = None,
    memory: Any = None,
    beliefs: Any = None,
    member_name: str = "unknown",
    bridge_secret: str | None = None,
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
    memory:
        :class:`~team.memory.AgentMemory` instance for this member, or
        ``None`` when memory is disabled.
    beliefs:
        :class:`~team.beliefs.BeliefBoard` instance shared by all members,
        or ``None`` when the belief board is disabled.
    member_name:
        Name of the calling member (forwarded to belief tools for attribution).

    Raises :class:`KeyError` if *name* is not in the registry.
    All exceptions from the tool implementation are caught and returned
    as error strings so a single bad tool call does not abort the turn.
    """
    registry = tools if tools is not None else TOOLS
    fn = registry[name]
    log.info("tool:%s executing", name)
    result = fn(
        body,
        workspace_path=workspace_path,
        timeout=timeout,
        memory=memory,
        beliefs=beliefs,
        member_name=member_name,
        bridge_secret=bridge_secret,
    )
    log.debug("tool:%s → %d chars", name, len(result))
    return result
