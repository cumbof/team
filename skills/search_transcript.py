"""search_transcript — keyword search over the current run's transcript.

This skill lets a member query what was said earlier in the run without
loading the entire transcript into the context window.

Tool
----
search_transcript

Body syntax
-----------
    keyword: <search terms>
    speaker: <member name>   (optional — filter by speaker)
    limit: <N>               (optional — max results, default 5)

Only ``keyword`` is required.  ``speaker`` and ``limit`` are optional
filters.  All matching turns are returned with their index, speaker, and
a short excerpt of the content.

The transcript is read from ``transcript.jsonl`` in the team workspace
(one level above the shared workspace directory that tools normally see).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_DEFAULT_LIMIT = 5
_EXCERPT_LENGTH = 300


def _parse_kv(body: str, key: str) -> str | None:
    for line in body.splitlines():
        line = line.strip()
        if line.lower().startswith(f"{key}:"):
            return line[len(key) + 1:].strip()
    return None


def execute(
    body: str,
    *,
    workspace_path: Path | None = None,
    **_: Any,
) -> str:
    if workspace_path is None:
        return "ERROR: workspace_path not available"

    # The shared workspace is <team_workspace>/shared; the transcript is
    # one level up at <team_workspace>/transcript.jsonl.
    transcript_path = workspace_path.parent / "transcript.jsonl"
    if not transcript_path.is_file():
        return f"ERROR: transcript not found at {transcript_path}"

    keyword_raw = _parse_kv(body, "keyword")
    if keyword_raw is None:
        # No "keyword:" key — treat the whole body as the search term.
        keyword_raw = body.strip()
    speaker_filter = (_parse_kv(body, "speaker") or "").lower()
    try:
        limit = int(_parse_kv(body, "limit") or _DEFAULT_LIMIT)
    except ValueError:
        limit = _DEFAULT_LIMIT

    if not keyword_raw:
        return "ERROR: provide keyword: <search terms>"

    pattern = re.compile(re.escape(keyword_raw), re.IGNORECASE)

    results: list[str] = []
    for raw_line in transcript_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        try:
            turn = json.loads(raw_line)
        except json.JSONDecodeError:
            continue

        speaker = turn.get("speaker", "")
        content = turn.get("content", "")

        if speaker_filter and speaker.lower() != speaker_filter:
            continue
        if not pattern.search(content):
            continue

        idx = turn.get("index", "?")
        excerpt = content[:_EXCERPT_LENGTH].replace("\n", " ")
        if len(content) > _EXCERPT_LENGTH:
            excerpt += "…"
        results.append(f"[turn {idx}] @{speaker}: {excerpt}")

        if len(results) >= limit:
            break

    if not results:
        desc = f"keyword={keyword_raw!r}"
        if speaker_filter:
            desc += f", speaker={speaker_filter!r}"
        return f"No matches found for {desc}."

    header = f"{len(results)} match(es) for {keyword_raw!r}:"
    return header + "\n\n" + "\n\n".join(results)


TOOL_NAME = "search_transcript"
TOOL_DESCRIPTION = (
    "Search the run transcript for turns containing a keyword. "
    "Body: 'keyword: <terms>' with optional 'speaker: <name>' and 'limit: <N>'. "
    "Returns matching turn excerpts with index and speaker."
)
