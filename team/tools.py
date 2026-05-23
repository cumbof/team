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
- ``delegate_to_expert`` — send a prompt to an external cloud LLM (OpenAI, Anthropic, Google)
  when the local model needs expert assistance.
- ``log_decision``    — append a timestamped decision entry to decisions.md in the shared workspace.
- ``read_decisions``  — read the full decisions log (decisions.md) from the shared workspace.

Security note
-------------
``run_python`` and ``run_bash`` execute code on the **host machine** with the
privileges of the process that runs ``team``.  Only enable these tools for
members whose prompts you trust, and always review the generated code before
a run when security matters.

``delegate_to_expert`` sends the prompt text to an external API (OpenAI,
Anthropic, or Google).  Only enable it for members you trust to handle
sensitive data appropriately, and ensure the corresponding API key is set
in the environment.  Required env vars: ``OPENAI_API_KEY``,
``ANTHROPIC_API_KEY``, or ``GOOGLE_API_KEY`` (depending on the provider used).
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

# Valid sandbox identifiers for run_python / run_bash.
_VALID_TOOL_SANDBOXES = {"none", "firejail", "bubblewrap"}


# --------------------------------------------------------------------------- #
# Sandbox helpers
# --------------------------------------------------------------------------- #


def _sandbox_available(name: str) -> bool:
    """Return True if *name* (e.g. ``"firejail"``, ``"bwrap"``) is on PATH."""
    import shutil
    return shutil.which(name) is not None


def _build_sandboxed_cmd(
    cmd: list[str],
    workspace_path: Path | None,
    sandbox: str,
) -> list[str]:
    """Wrap *cmd* with the requested sandbox tool.

    Parameters
    ----------
    cmd:
        The bare command list to sandbox (e.g. ``["python", "/tmp/x.py"]``).
    workspace_path:
        Shared workspace directory that must be writable inside the sandbox.
    sandbox:
        One of ``"firejail"``, ``"bubblewrap"``.  ``"none"`` is a no-op and
        returns *cmd* unchanged.

    When the requested sandbox binary is not available the function falls back
    to *cmd* unchanged and logs a warning — the tool still executes, just
    unsandboxed.
    """
    if sandbox == "none":
        return cmd

    ws = str(workspace_path) if workspace_path else None

    if sandbox == "firejail":
        if not _sandbox_available("firejail"):
            log.warning(
                "tool_sandbox=firejail requested but 'firejail' is not on PATH; "
                "running without sandbox. Install firejail to enable isolation."
            )
            return cmd
        args = [
            "firejail",
            "--quiet",
            "--noprofile",
            "--private-tmp",
            "--noroot",
        ]
        if ws:
            # Allow read-write access to the workspace; all other user paths
            # are invisible inside the jail.
            args.append(f"--whitelist={ws}")
        args.append("--")
        return args + cmd

    if sandbox == "bubblewrap":
        bwrap = "bwrap"
        if not _sandbox_available(bwrap):
            log.warning(
                "tool_sandbox=bubblewrap requested but 'bwrap' is not on PATH; "
                "running without sandbox. Install bubblewrap to enable isolation."
            )
            return cmd
        # Build a minimal read-only rootfs.  Only the workspace is writable.
        args = [bwrap]
        for sys_dir in ("/usr", "/bin", "/sbin", "/lib", "/lib32", "/lib64",
                        "/etc", "/run"):
            p = Path(sys_dir)
            if p.exists():
                args += ["--ro-bind", sys_dir, sys_dir]
        # Proc, dev, and a private /tmp.
        args += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]
        # No access to $HOME or arbitrary host paths.
        args += ["--tmpfs", "/home", "--tmpfs", "/root"]
        # Writable workspace bind.
        if ws:
            args += ["--bind", ws, ws]
        args += ["--chdir", ws or "/tmp", "--die-with-parent", "--"]
        return args + cmd

    # Unknown sandbox value — already validated upstream but be safe.
    log.warning("Unknown tool_sandbox value %r; running without sandbox.", sandbox)
    return cmd


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
    sandbox: str = "none",
    **_: Any,
) -> str:
    """Execute *body* as Python code and return stdout + stderr."""
    code = body.strip()
    env = os.environ.copy()
    cwd = str(workspace_path) if workspace_path and workspace_path.is_dir() else None
    if cwd:
        env["WORKSPACE"] = cwd

    # When sandboxing, write the script inside the workspace so it is
    # accessible from inside the sandbox (which has no access to /tmp).
    script_dir = workspace_path if sandbox != "none" and workspace_path and workspace_path.is_dir() else None

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8",
        dir=script_dir,
    ) as fh:
        fh.write(code)
        script = fh.name

    try:
        base_cmd = ["python", script]
        cmd = _build_sandboxed_cmd(base_cmd, workspace_path, sandbox)
        result = subprocess.run(
            cmd,
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
    sandbox: str = "none",
    **_: Any,
) -> str:
    """Execute *body* as a bash command and return stdout + stderr."""
    cmd_str = body.strip()
    cwd = str(workspace_path) if workspace_path and workspace_path.is_dir() else None
    try:
        base_cmd = ["bash", "-c", cmd_str]
        cmd = _build_sandboxed_cmd(base_cmd, workspace_path, sandbox)
        result = subprocess.run(
            cmd,
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
    peers: dict[str, str] | None = None,
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

    Instead of a raw ``url``, you can use a named peer from the team config::

        peer: lab-b
        goal: Run the survival analysis on the preprocessed BRCA dataset.

    All fields except ``url``/``peer`` and ``goal`` are optional.

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

    # Resolve URL — accept either a raw url: or a named peer: from the registry.
    url = _parse_kv(body, "url")
    if not url:
        peer_name = _parse_kv(body, "peer")
        if peer_name:
            peer_registry = peers or {}
            url = peer_registry.get(peer_name)
            if not url:
                return (
                    f"ERROR: peer {peer_name!r} is not in the configured peers registry. "
                    f"Add it under bridge.peers in your team YAML."
                )
        else:
            return "ERROR: provide url: <remote bridge server URL> or peer: <peer name>"

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
# Federation tools: peer registry, broadcast, cancel
# --------------------------------------------------------------------------- #


def _list_peers(
    body: str,
    *,
    peers: dict[str, str] | None = None,
    bridge_secret: str | None = None,
    **_: Any,
) -> str:
    """List configured peer teams and check their health status.

    Iterates over all peers in the team's ``bridge.peers`` registry, queries
    each one's ``/health`` endpoint, and reports the result.

    Body: (empty — no parameters required)

    Returns a table of peer names, URLs, and live health data (``pending``
    and ``running`` task counts).  Unreachable peers are flagged so you can
    decide whether to delegate to them.
    """
    from team.bridge_client import BridgeClient, BridgeClientError

    peer_registry = peers or {}
    if not peer_registry:
        return "No peers configured. Add a peers: section under bridge: in your team YAML."

    lines = ["Configured peers:"]
    for name, url in peer_registry.items():
        try:
            client = BridgeClient(url, secret=bridge_secret)
            h = client.health()
            lines.append(
                f"  {name}: {url}"
                f"  [ok · pending={h.get('pending', '?')}"
                f" · running={h.get('running', '?')}]"
            )
        except BridgeClientError as exc:
            lines.append(f"  {name}: {url}  [unreachable: {exc}]")
    return "\n".join(lines)


def _broadcast_task(
    body: str,
    *,
    workspace_path: Path | None = None,
    timeout: int = 600,
    bridge_secret: str | None = None,
    peers: dict[str, str] | None = None,
    **_: Any,
) -> str:
    """Submit the same goal to multiple peer teams concurrently and collect all results.

    Sends identical tasks to each listed peer in parallel, waits for all of
    them to finish, and returns a combined summary.  Files returned by each
    peer are written into the local workspace under a per-peer sub-directory
    (``<peer-name>/<original-relative-path>``) to avoid name collisions.

    Body format::

        goal: Run the regression analysis on the preprocessed dataset.
        peers: lab-b, lab-c, lab-d
        context: Data is in data/preprocessed.csv (1 142 samples).
        files: data/preprocessed.csv
        timeout: 600

    ``peers`` is a comma-separated list of peer names (from ``bridge.peers``
    in the team YAML) or direct ``http://…`` URLs.  All other fields work
    the same as in ``delegate_task``.

    Useful for ensemble processing (send to N specialist teams, compare
    results), redundancy (take the first successful answer), or
    embarrassingly parallel sub-tasks.
    """
    import concurrent.futures

    from team.bridge import BridgeTask
    from team.bridge_client import BridgeClient, BridgeClientError

    goal = _parse_kv(body, "goal")
    if not goal:
        return "ERROR: provide goal: <what all remote teams should accomplish>"
    peers_str = _parse_kv(body, "peers")
    if not peers_str:
        return "ERROR: provide peers: <comma-separated peer names or URLs>"

    context = _parse_kv(body, "context") or ""
    files_str = _parse_kv(body, "files") or ""
    task_timeout_str = _parse_kv(body, "timeout")
    task_timeout = float(task_timeout_str) if task_timeout_str else float(timeout)

    peer_registry = peers or {}
    resolved: dict[str, str] = {}
    for raw in [p.strip() for p in peers_str.split(",") if p.strip()]:
        if raw in peer_registry:
            resolved[raw] = peer_registry[raw]
        elif raw.startswith("http"):
            resolved[raw] = raw
        else:
            return (
                f"ERROR: {raw!r} is not a known peer name and does not look like a URL. "
                f"Known peers: {', '.join(peer_registry) or '(none)'}"
            )

    if not resolved:
        return "ERROR: no valid peers to broadcast to"

    input_files: dict[str, str] = {}
    if files_str and workspace_path:
        for rel in [f.strip() for f in files_str.split(",") if f.strip()]:
            try:
                target = (workspace_path / rel).resolve()
                target.relative_to(workspace_path.resolve())
                if target.is_file():
                    input_files[rel] = target.read_text(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass

    def _submit_one(
        peer_name: str, url: str
    ) -> tuple[str, str, dict[str, str] | None, str | None]:
        try:
            task = BridgeTask(
                goal=goal, context=context, files=input_files, sender="local-team"
            )
            client = BridgeClient(url, secret=bridge_secret)
            task_id = client.submit_task(task)
            result = client.wait_for_result(task_id, timeout=task_timeout)
            if result.status == "error":
                return peer_name, "", None, result.error
            return peer_name, result.summary, result.files, None
        except BridgeClientError as exc:
            return peer_name, "", None, str(exc)

    result_parts: list[str] = [
        f"Broadcast to {len(resolved)} peer(s) · goal: '{goal[:60]}'\n"
    ]
    all_files: dict[str, str] = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(resolved)) as pool:
        futures = {
            pool.submit(_submit_one, name, url): name
            for name, url in resolved.items()
        }
        for future in concurrent.futures.as_completed(futures):
            peer_name, summary, files, error = future.result()
            if error:
                result_parts.append(f"### {peer_name}: ERROR\n{error}\n")
            else:
                result_parts.append(f"### {peer_name}: complete\n{summary[:500]}\n")
                if files:
                    for rel, content in files.items():
                        all_files[f"{peer_name}/{rel}"] = content

    written: list[str] = []
    if workspace_path and all_files:
        for rel, content in all_files.items():
            try:
                target = (workspace_path / rel).resolve()
                target.relative_to(workspace_path.resolve())
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
                written.append(rel)
            except (ValueError, OSError) as exc:
                log.warning("broadcast_task: could not write %r: %s", rel, exc)

    if written:
        result_parts.append(f"\nFiles received ({len(written)} total, namespaced by peer):")
        result_parts.extend(f"  - {p}" for p in written)

    return _truncate("\n".join(result_parts))


def _cancel_remote_task(
    body: str,
    *,
    peers: dict[str, str] | None = None,
    bridge_secret: str | None = None,
    **_: Any,
) -> str:
    """Cancel a running or queued task on a remote bridge server.

    Body format::

        url: http://lab-b.example.com:7001
        task_id: <task UUID returned by a previous delegate_task call>

    As with ``delegate_task``, you can use a peer name instead of a raw URL::

        peer: lab-b
        task_id: <task UUID>

    Returns ``"Cancelled: <task_id>"`` on success.  Returns an error string
    if the task is not found, already finished, or the server is unreachable.
    """
    from team.bridge_client import BridgeClient, BridgeClientError

    url = _parse_kv(body, "url")
    if not url:
        peer_name = _parse_kv(body, "peer")
        if peer_name:
            peer_registry = peers or {}
            url = peer_registry.get(peer_name)
            if not url:
                return (
                    f"ERROR: peer {peer_name!r} is not in the configured peers registry."
                )
        else:
            return "ERROR: provide url: <remote bridge server URL> or peer: <peer name>"

    task_id = _parse_kv(body, "task_id")
    if not task_id:
        return "ERROR: provide task_id: <task UUID to cancel>"

    try:
        client = BridgeClient(url, secret=bridge_secret)
        client.cancel_task(task_id)
        return f"Cancelled: {task_id}"
    except BridgeClientError as exc:
        return f"ERROR: {exc}"



# --------------------------------------------------------------------------- #
# Expert delegation: send a prompt to an external cloud LLM
# --------------------------------------------------------------------------- #

#: Supported external providers and their defaults.
_EXPERT_PROVIDERS: dict[str, dict] = {
    "openai": {
        "env_key": "OPENAI_API_KEY",
        "default_model": "gpt-4o",
        "api_url": "https://api.openai.com/v1/chat/completions",
    },
    "anthropic": {
        "env_key": "ANTHROPIC_API_KEY",
        "default_model": "claude-opus-4-5",
        "api_url": "https://api.anthropic.com/v1/messages",
    },
    "google": {
        "env_key": "GOOGLE_API_KEY",
        "default_model": "gemini-1.5-pro",
        "api_url": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
    },
}


def _call_openai(
    api_url: str,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout: int,
) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    r = requests.post(api_url, json=payload, headers=headers, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    choices = data.get("choices") or []
    if not choices:
        return "ERROR: OpenAI returned no choices in response"
    return choices[0].get("message", {}).get("content", "").strip()


def _call_anthropic(
    api_url: str,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout: int,
) -> str:
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    r = requests.post(api_url, json=payload, headers=headers, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    content = data.get("content") or []
    texts = [block.get("text", "") for block in content if block.get("type") == "text"]
    return "\n".join(texts).strip() or "ERROR: Anthropic returned no text content"


def _call_google(
    api_url_template: str,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout: int,
) -> str:
    url = api_url_template.format(model=model)
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature": temperature,
        },
    }
    params = {"key": api_key}
    r = requests.post(url, json=payload, params=params, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    candidates = data.get("candidates") or []
    if not candidates:
        return "ERROR: Google returned no candidates in response"
    parts = candidates[0].get("content", {}).get("parts") or []
    texts = [p.get("text", "") for p in parts]
    return "\n".join(texts).strip() or "ERROR: Google returned no text content"


def _delegate_to_expert(
    body: str,
    *,
    timeout: int = 120,
    **_: Any,
) -> str:
    """Send a prompt to an external subscription-based cloud LLM and return its reply.

    Use this when a task exceeds the local model's capabilities — the response
    is returned as a tool result so you can incorporate it into your own reply.

    **Privacy notice**: the prompt text is sent to an external API. Only use
    this tool when the data is safe to share with the chosen provider.

    Required environment variable (set on the host before running ``team``):

    * ``OPENAI_API_KEY``   — for ``provider: openai``
    * ``ANTHROPIC_API_KEY`` — for ``provider: anthropic``
    * ``GOOGLE_API_KEY``   — for ``provider: google``

    Body format (multi-line prompt via ``---`` separator)::

        provider: openai
        model: gpt-4o
        max_tokens: 2048
        temperature: 0.3
        ---
        Explain quantum entanglement in simple terms.
        Provide at least three analogies.

    Or a single-line prompt::

        provider: anthropic
        model: claude-opus-4-5
        prompt: What is the capital of France?

    Fields
    ------
    provider:
        Required.  One of ``openai``, ``anthropic``, ``google``.
    model:
        Optional.  Model identifier accepted by the provider API.
        Defaults: ``gpt-4o`` (OpenAI), ``claude-opus-4-5`` (Anthropic),
        ``gemini-1.5-pro`` (Google).
    prompt:
        The prompt to send (single-line form).  Ignored when the body
        contains a ``---`` separator — the text after ``---`` is used instead.
    max_tokens:
        Maximum tokens in the response (default: 2048).
    temperature:
        Sampling temperature 0–2 (default: 0.2).
    """
    provider_name = (_parse_kv(body, "provider") or "").strip().lower()
    if not provider_name:
        return (
            "ERROR: provide provider: <name> — supported: "
            + ", ".join(_EXPERT_PROVIDERS)
        )
    if provider_name not in _EXPERT_PROVIDERS:
        return (
            f"ERROR: unknown provider {provider_name!r}. "
            f"Supported: {', '.join(_EXPERT_PROVIDERS)}"
        )

    info = _EXPERT_PROVIDERS[provider_name]
    api_key = os.environ.get(info["env_key"], "").strip()
    if not api_key:
        return (
            f"ERROR: {info['env_key']} environment variable is not set. "
            f"Export it on the host before running team to use provider {provider_name!r}."
        )

    model = (_parse_kv(body, "model") or "").strip() or info["default_model"]

    # Multi-line prompt: text after the first \n---\n separator.
    idx = body.find("\n---\n")
    if idx != -1:
        prompt = body[idx + 5:].strip()
    else:
        prompt = (_parse_kv(body, "prompt") or "").strip()
    if not prompt:
        return "ERROR: provide a prompt (use 'prompt: ...' or place the text after a '---' line)"

    max_tokens_raw = _parse_kv(body, "max_tokens")
    try:
        max_tokens = int(max_tokens_raw) if max_tokens_raw else 2048
    except ValueError:
        return f"ERROR: max_tokens must be an integer, got {max_tokens_raw!r}"

    temperature_raw = _parse_kv(body, "temperature")
    try:
        temperature = float(temperature_raw) if temperature_raw else 0.2
    except ValueError:
        return f"ERROR: temperature must be a float, got {temperature_raw!r}"

    log.info(
        "delegate_to_expert: calling %s/%s (max_tokens=%d)",
        provider_name, model, max_tokens,
    )

    try:
        if provider_name == "openai":
            reply = _call_openai(
                info["api_url"], api_key, model, prompt,
                max_tokens, temperature, timeout,
            )
        elif provider_name == "anthropic":
            reply = _call_anthropic(
                info["api_url"], api_key, model, prompt,
                max_tokens, temperature, timeout,
            )
        else:  # google
            reply = _call_google(
                info["api_url"], api_key, model, prompt,
                max_tokens, temperature, timeout,
            )
    except requests.exceptions.Timeout:
        return f"ERROR: request to {provider_name} timed out after {timeout}s"
    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "?"
        return f"ERROR: {provider_name} API returned HTTP {status}: {exc}"
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: delegate_to_expert failed ({provider_name}): {exc}"

    if reply.startswith("ERROR"):
        return reply

    header = f"[Expert response from {provider_name}/{model}]\n\n"
    return _truncate(header + reply)


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
# Registry discovery tool
# --------------------------------------------------------------------------- #


def _query_registry(
    body: str,
    *,
    registry_url: str | None = None,
    **_: Any,
) -> str:
    """Query a team registry server to find teams matching a keyword or tag.

    Use this tool to discover remote teams that can handle a specialised
    sub-task.  Once you know a team's URL, delegate work to it with
    ``delegate_task``.

    Body format::

        url: http://registry.example.com:8000
        tag: biology
        tag: python
        keyword: survival analysis
        limit: 5

    ``url`` is the registry server URL.  If omitted, the team's configured
    ``registry.url`` is used (set via the ``--registry-url`` flag or the YAML
    ``registry.url`` field).  ``tag`` may be repeated for multiple required
    tags.  ``keyword`` is a substring searched across team names, descriptions,
    and tags.  ``limit`` caps the number of results (default: 10).

    Returns a Markdown list of matching teams with their name, URL,
    description, and capability tags.
    """
    from team.registry_client import RegistryClient, RegistryClientError

    # Collect all tags from body (handles one or many "tag: X" lines).
    tags: list[str] = []
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("tag:"):
            t = stripped[4:].strip()
            if t:
                tags.append(t)

    keyword = _parse_kv(body, "keyword") or None
    limit_raw = _parse_kv(body, "limit")
    try:
        limit = int(limit_raw) if limit_raw else 10
    except ValueError:
        limit = 10

    # URL resolution: explicit body param → injected registry_url → error.
    url = _parse_kv(body, "url") or registry_url
    if not url:
        return (
            "ERROR: provide url: <registry URL> or configure registry.url in the team YAML"
        )

    try:
        client = RegistryClient(url)
        entries = client.query(tags=tags or None, keyword=keyword)
    except RegistryClientError as exc:
        return f"ERROR: registry query failed: {exc}"

    if not entries:
        filters = []
        if tags:
            filters.append(f"tags={tags}")
        if keyword:
            filters.append(f"keyword={keyword!r}")
        desc = " and ".join(filters) if filters else "(no filters)"
        return f"No registered teams match {desc}."

    shown = entries[:limit]
    lines = [
        f"Found {len(entries)} matching team(s)"
        + (f" (showing {len(shown)})" if len(entries) > limit else "")
        + ":"
    ]
    for e in shown:
        tags_str = ", ".join(e.tags) if e.tags else "(none)"
        models_str = ", ".join(e.models) if e.models else "(unknown)"
        lines.append(
            f"\n**{e.name}**  \n"
            f"  URL: {e.url}  \n"
            f"  Description: {e.description or '(none)'}  \n"
            f"  Tags: {tags_str}  \n"
            f"  Models: {models_str}"
        )
    return _truncate("\n".join(lines))


def _sync_beliefs(
    body: str,
    *,
    beliefs: Any = None,
    workspace_path: Path | None = None,
    member_name: str = "unknown",
    bridge_secret: str | None = None,
    peers: dict[str, str] | None = None,
    **_kwargs: object,
) -> str:
    """Synchronize beliefs with a remote team via the bridge protocol.

    Supports three directions:
    * ``pull`` (default) — fetch accepted beliefs from the remote team and
      import them locally as pending for consensus validation.
    * ``push`` — export local beliefs and send them to the remote team.
    * ``both`` — do pull then push in sequence.

    Parameters parsed from *body* (key: value lines):

    * ``url:`` — bridge URL of the remote team (required).  ``peer:`` is also
      accepted to resolve from the configured peer table.
    * ``direction: pull | push | both`` (default: ``pull``).
    * ``status:`` — filter beliefs by status when pushing or pulling
      (e.g. ``accepted``).  ``None`` means all statuses.
    * ``limit:`` — max beliefs to transfer per direction (default: 50).
    * ``local_team:`` — name to use as source when pushing (default: ``local``).
    """
    from team.bridge_client import BridgeClient, BridgeClientError

    # --- Resolve URL (peer: name or url: <url>) ---------------------------
    raw_url = _parse_kv(body, "url")
    peer_name = _parse_kv(body, "peer")
    if peer_name and peers:
        raw_url = peers.get(peer_name) or raw_url
    if not raw_url:
        return "ERROR: provide url: <bridge URL> or peer: <name>"

    direction = (_parse_kv(body, "direction") or "pull").lower().strip()
    if direction not in ("pull", "push", "both"):
        direction = "pull"
    status_filter = _parse_kv(body, "status") or None
    limit_raw = _parse_kv(body, "limit")
    try:
        limit = int(limit_raw) if limit_raw else 50
    except ValueError:
        limit = 50
    local_team = _parse_kv(body, "local_team") or "local"

    # --- Lazy-load board from workspace when not injected -----------------
    board = beliefs
    if board is None and workspace_path is not None:
        beliefs_path = workspace_path / "beliefs.json"
        if beliefs_path.exists():
            try:
                from team.beliefs import BeliefBoard
                board = BeliefBoard(beliefs_path, [])
            except Exception:  # noqa: BLE001
                pass

    if board is None:
        return "ERROR: no belief board available — enable beliefs in the team YAML"

    client = BridgeClient(raw_url, secret=bridge_secret)
    lines: list[str] = []

    # --- Pull direction ---------------------------------------------------
    if direction in ("pull", "both"):
        try:
            bundle = client.pull_beliefs(status=status_filter, limit=limit)
        except BridgeClientError as exc:
            lines.append(f"Pull failed: {exc}")
            bundle = None
        if bundle is None:
            lines.append("Remote team does not expose a belief board.")
        else:
            from team.belief_federation import import_beliefs
            imported, skipped = import_beliefs(board, bundle)
            lines.append(
                f"Pulled from **{bundle.source_team}**: "
                f"{imported} belief(s) imported as pending, "
                f"{skipped} skipped (already known)."
            )
            if imported:
                lines.append(
                    "Tip: use `list_beliefs` to review the new pending beliefs "
                    "and `accept_belief` / `contest_belief` to vote on them."
                )

    # --- Push direction ---------------------------------------------------
    if direction in ("push", "both"):
        try:
            from team.belief_federation import export_beliefs
            bundle_out = export_beliefs(
                board, local_team, raw_url, status=status_filter, limit=limit
            )
            result = client.push_beliefs(bundle_out)
        except BridgeClientError as exc:
            lines.append(f"Push failed: {exc}")
            result = None
        if result is None:
            lines.append("Remote team does not support belief import.")
        else:
            r_imported, r_skipped = result
            lines.append(
                f"Pushed to remote ({raw_url}): "
                f"remote imported {r_imported} belief(s), "
                f"skipped {r_skipped}."
            )

    return "\n".join(lines) if lines else "No sync performed."

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
            "url":     {"type": "string",  "description": "Remote bridge server URL (alternative to peer)."},
            "peer":    {"type": "string",  "description": "Named peer from bridge.peers config (alternative to url)."},
            "goal":    {"type": "string",  "description": "What the remote team should accomplish."},
            "context": {"type": "string",  "description": "Optional background for the remote team."},
            "files":   {"type": "string",  "description": "Comma-separated relative paths of files to send."},
            "timeout": {"type": "integer", "description": "Seconds to wait for the remote team (default 600)."},
        },
        ["goal"],
    ),
    "list_peers": _fn(
        "list_peers",
        "List all configured peer teams and their live health status.",
        {},
        [],
    ),
    "broadcast_task": _fn(
        "broadcast_task",
        "Submit the same goal to multiple peer teams concurrently and collect all results.",
        {
            "goal":    {"type": "string", "description": "What all remote teams should accomplish."},
            "peers":   {"type": "string", "description": "Comma-separated peer names or URLs to broadcast to."},
            "context": {"type": "string", "description": "Optional background for the remote teams."},
            "files":   {"type": "string", "description": "Comma-separated relative paths of files to send."},
            "timeout": {"type": "integer", "description": "Seconds to wait per peer (default 600)."},
        },
        ["goal", "peers"],
    ),
    "cancel_remote_task": _fn(
        "cancel_remote_task",
        "Cancel a queued or running task on a remote bridge server by task ID.",
        {
            "url":     {"type": "string", "description": "Remote bridge server URL (alternative to peer)."},
            "peer":    {"type": "string", "description": "Named peer from bridge.peers config (alternative to url)."},
            "task_id": {"type": "string", "description": "Task UUID to cancel (returned by delegate_task)."},
        },
        ["task_id"],
    ),
    "delegate_to_expert": _fn(
        "delegate_to_expert",
        "Send a prompt to an external cloud LLM (OpenAI, Anthropic, Google) for expert assistance.",
        {
            "provider":    {
                "type": "string",
                "enum": ["openai", "anthropic", "google"],
                "description": "Cloud provider to use.",
            },
            "model":       {"type": "string",  "description": "Model identifier (optional; provider default used if omitted)."},
            "prompt":      {"type": "string",  "description": "The prompt or question to send to the expert model."},
            "max_tokens":  {"type": "integer", "description": "Maximum tokens in the response (default 2048)."},
            "temperature": {"type": "number",  "description": "Sampling temperature 0–2 (default 0.2)."},
        },
        ["provider", "prompt"],
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
    "query_registry": _fn(
        "query_registry",
        "Query a team registry server to discover teams matching capability tags or a keyword.",
        {
            "url":     {"type": "string", "description": "Registry server URL (optional if registry.url is configured)."},
            "tag":     {"type": "string", "description": "Required capability tag (call multiple times for AND logic)."},
            "keyword": {"type": "string", "description": "Substring searched across team names, descriptions, and tags."},
            "limit":   {"type": "integer", "description": "Maximum number of results to return (default 10)."},
        },
        [],
    ),
    "sync_beliefs": _fn(
        "sync_beliefs",
        "Synchronize the team belief board with a remote team cluster (pull, push, or both).",
        {
            "url":        {"type": "string", "description": "Bridge URL of the remote team."},
            "peer":       {"type": "string", "description": "Named peer (resolved from bridge.peers config)."},
            "direction":  {"type": "string", "enum": ["pull", "push", "both"], "description": "Sync direction (default: pull)."},
            "status":     {"type": "string", "description": "Filter beliefs by status (e.g. 'accepted'). Omit for all."},
            "limit":      {"type": "integer", "description": "Max beliefs to transfer per direction (default: 50)."},
            "local_team": {"type": "string", "description": "Name to use as source when pushing (default: 'local')."},
        },
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
        lines = []
        if args.get("peer"):
            lines.append(f"peer: {args['peer']}")
        else:
            lines.append(f"url: {args.get('url', '')}")
        lines.append(f"goal: {args.get('goal', '')}")
        if args.get("context"):
            lines.append(f"context: {args['context']}")
        if args.get("files"):
            lines.append(f"files: {args['files']}")
        if args.get("timeout"):
            lines.append(f"timeout: {args['timeout']}")
        return "\n".join(lines)

    if tool_name == "broadcast_task":
        lines = [f"goal: {args.get('goal', '')}", f"peers: {args.get('peers', '')}"]
        if args.get("context"):
            lines.append(f"context: {args['context']}")
        if args.get("files"):
            lines.append(f"files: {args['files']}")
        if args.get("timeout"):
            lines.append(f"timeout: {args['timeout']}")
        return "\n".join(lines)

    if tool_name == "cancel_remote_task":
        lines = []
        if args.get("peer"):
            lines.append(f"peer: {args['peer']}")
        else:
            lines.append(f"url: {args.get('url', '')}")
        lines.append(f"task_id: {args.get('task_id', '')}")
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

    if tool_name == "query_registry":
        lines = []
        if args.get("url"):
            lines.append(f"url: {args['url']}")
        # tag can be a single value; wrap in list for uniform handling
        tag = args.get("tag")
        if tag:
            for t in ([tag] if isinstance(tag, str) else tag):
                lines.append(f"tag: {t}")
        if args.get("keyword"):
            lines.append(f"keyword: {args['keyword']}")
        if args.get("limit") is not None:
            lines.append(f"limit: {args['limit']}")
        return "\n".join(lines)

    if tool_name == "sync_beliefs":
        lines = []
        if args.get("url"):
            lines.append(f"url: {args['url']}")
        if args.get("peer"):
            lines.append(f"peer: {args['peer']}")
        if args.get("direction"):
            lines.append(f"direction: {args['direction']}")
        if args.get("status"):
            lines.append(f"status: {args['status']}")
        if args.get("limit") is not None:
            lines.append(f"limit: {args['limit']}")
        if args.get("local_team"):
            lines.append(f"local_team: {args['local_team']}")
        return "\n".join(lines)

    if tool_name == "delegate_to_expert":
        lines = [f"provider: {args.get('provider', '')}"]
        if args.get("model"):
            lines.append(f"model: {args['model']}")
        if args.get("max_tokens") is not None:
            lines.append(f"max_tokens: {args['max_tokens']}")
        if args.get("temperature") is not None:
            lines.append(f"temperature: {args['temperature']}")
        lines.append("---")
        lines.append(args.get("prompt", ""))
        return "\n".join(lines)

    # Unknown tool or custom skill — pass args as JSON so the skill can parse them.
    import json as _json
    return _json.dumps(args, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# Registry & dispatch
# --------------------------------------------------------------------------- #

#: All built-in tools, keyed by name.  Members opt-in via ``tools:`` config.
TOOLS: dict[str, Any] = {
    "run_python":          _run_python,
    "run_bash":            _run_bash,
    "web_search":          _web_search,
    "read_url":            _read_url,
    "read_file":           _read_file,
    "write_file":          _write_file,
    "append_file":         _append_file,
    "list_files":          _list_files,
    "remember":            _remember,
    "recall":              _recall,
    "forget":              _forget,
    "list_memories":       _list_memories,
    "assert_belief":       _assert_belief,
    "contest_belief":      _contest_belief,
    "accept_belief":       _accept_belief,
    "list_beliefs":        _list_beliefs,
    "delegate_task":       _delegate_task,
    "list_peers":          _list_peers,
    "broadcast_task":      _broadcast_task,
    "cancel_remote_task":  _cancel_remote_task,
    "delegate_to_expert":  _delegate_to_expert,
    "log_decision":        _log_decision,
    "read_decisions":      _read_decisions,
    "query_registry":      _query_registry,
    "sync_beliefs":        _sync_beliefs,
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
        "the local workspace automatically. Use 'peer: <name>' for named peers "
        "or 'url: <URL>' for direct addressing."
    ),
    "list_peers": (
        "List all configured peer teams and their live health status "
        "(pending/running task counts). Useful before delegating to confirm "
        "peers are reachable."
    ),
    "broadcast_task": (
        "Submit the same goal to multiple peer teams concurrently and collect "
        "all their results. Ideal for ensemble processing, redundancy, or "
        "embarrassingly parallel sub-tasks across the federation."
    ),
    "cancel_remote_task": (
        "Cancel a queued or running task on a remote bridge server by task ID. "
        "Use after a delegate_task call when you no longer need the result."
    ),
    "delegate_to_expert": (
        "Send a prompt to an external cloud LLM (OpenAI, Anthropic, or Google) "
        "and return its reply. Use this when the current task exceeds your "
        "capabilities and you need expert assistance. Requires the provider's "
        "API key to be set in the environment (OPENAI_API_KEY, ANTHROPIC_API_KEY, "
        "or GOOGLE_API_KEY). WARNING: the prompt is sent to an external service."
    ),
    "log_decision":   (
        "Append a timestamped decision entry to decisions.md in the shared workspace. "
        "Provide title, rationale, and alternatives considered."
    ),
    "read_decisions": "Read the full decision log (decisions.md) from the shared workspace.",
    "query_registry": (
        "Query a team registry server to find teams matching capability tags or a keyword. "
        "Returns each matching team's name, URL, description, and tags so you can then "
        "use delegate_task to route work to the right specialist team."
    ),
    "sync_beliefs": (
        "Synchronize the team's belief board with a remote team cluster. "
        "Pull accepted beliefs from a remote team (they arrive as pending for local voting), "
        "push your team's beliefs to a remote team, or do both. "
        "Use url: <bridge URL> or peer: <name>."
    ),
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
    peers: dict[str, str] | None = None,
    sandbox: str = "none",
    registry_url: str | None = None,
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
    bridge_secret:
        Shared HMAC secret for bridge authentication.
    peers:
        Mapping of peer name → URL from the team's ``bridge.peers`` config.
        Forwarded to federation tools (``delegate_task``, ``broadcast_task``,
        etc.) so they can resolve named peers.
    sandbox:
        Sandbox mode for code-execution tools (``run_python``, ``run_bash``).
        One of ``"none"`` (default), ``"firejail"``, or ``"bubblewrap"``.
        Passed through via ``**kwargs`` to the tool; non-code tools ignore it.
    registry_url:
        Base URL of the team registry server, forwarded to ``query_registry``
        as the default URL when none is specified in the tool body.

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
        peers=peers,
        sandbox=sandbox,
        registry_url=registry_url,
    )
    log.debug("tool:%s → %d chars", name, len(result))
    return result
