"""Test runner for ``team test``.

Each entry in the ``tests:`` section of a team YAML is an *assertion dict*.
``run_assertions()`` evaluates all assertions against the run artefacts and
returns a list of :class:`AssertionResult` objects.

Supported assertion types
-------------------------
``file_exists``
    The file at ``path`` (relative to the shared workspace) must exist.

``file_not_exists``
    The file at ``path`` must *not* exist.

``file_contains``
    The file at ``path`` must contain the substring ``text``.

``file_not_contains``
    The file at ``path`` must *not* contain the substring ``text``.

``json_valid``
    The file at ``path`` must be parseable as JSON.

``json_schema``
    The file at ``path`` must be valid JSON and must match the JSON Schema
    provided in the ``schema`` key.

``transcript_contains``
    The run transcript must contain a turn whose ``content`` field includes
    ``text``.  Optionally filter by ``speaker`` (member name without ``@``).

``transcript_count``
    The total number of member turns in the transcript must equal ``count``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AssertionResult:
    """Result of a single test assertion."""
    name: str
    passed: bool
    detail: str


def run_assertions(
    assertions: list[dict],
    workspace: Path,
    transcript_path: Path,
) -> list[AssertionResult]:
    """Evaluate *assertions* against the run artefacts.

    Parameters
    ----------
    assertions:
        Raw assertion dicts from the ``tests:`` section of the team YAML.
    workspace:
        Path to the team workspace root (the shared files live under
        ``workspace / "shared"``).
    transcript_path:
        Path to the ``transcript.jsonl`` file produced by the run.
    """
    return [_run_one(a, workspace, transcript_path) for a in assertions]


def _run_one(a: dict, workspace: Path, transcript_path: Path) -> AssertionResult:
    name = a.get("name") or a.get("type", "unnamed")
    atype = a.get("type", "")
    try:
        passed, detail = _check(atype, a, workspace, transcript_path)
    except Exception as exc:
        passed = False
        detail = f"error: {exc}"
    return AssertionResult(name=name, passed=passed, detail=detail)


def _check(
    atype: str,
    a: dict,
    workspace: Path,
    transcript_path: Path,
) -> tuple[bool, str]:
    shared = workspace / "shared"

    if atype == "file_exists":
        path = shared / a["path"]
        if path.is_file():
            return True, f"{a['path']} exists"
        return False, f"{a['path']} not found"

    if atype == "file_not_exists":
        path = shared / a["path"]
        if not path.exists():
            return True, f"{a['path']} does not exist"
        return False, f"{a['path']} exists but should not"

    if atype == "file_contains":
        path = shared / a["path"]
        if not path.is_file():
            return False, f"{a['path']} not found"
        content = path.read_text(encoding="utf-8", errors="replace")
        text = a["text"]
        if text in content:
            return True, f"found {text!r}"
        return False, f"{text!r} not found in {a['path']}"

    if atype == "file_not_contains":
        path = shared / a["path"]
        if not path.is_file():
            return False, f"{a['path']} not found"
        content = path.read_text(encoding="utf-8", errors="replace")
        text = a["text"]
        if text not in content:
            return True, f"{text!r} not present"
        return False, f"{text!r} found in {a['path']} but should not be"

    if atype == "json_valid":
        path = shared / a["path"]
        if not path.is_file():
            return False, f"{a['path']} not found"
        try:
            json.loads(path.read_text(encoding="utf-8"))
            return True, "valid JSON"
        except json.JSONDecodeError as exc:
            return False, f"invalid JSON: {exc}"

    if atype == "json_schema":
        path = shared / a["path"]
        if not path.is_file():
            return False, f"{a['path']} not found"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return False, f"invalid JSON: {exc}"
        schema = a.get("schema", {})
        try:
            import jsonschema  # type: ignore[import]
            jsonschema.validate(data, schema)
            return True, "matches schema"
        except ImportError:
            return False, "jsonschema is not installed (pip install jsonschema)"
        except Exception as exc:  # jsonschema.ValidationError
            return False, f"schema error: {getattr(exc, 'message', exc)}"

    if atype == "transcript_contains":
        turns = _load_turns(transcript_path)
        speaker_filter = a.get("speaker")
        text = a["text"]
        for t in turns:
            if speaker_filter and t.get("speaker") != speaker_filter:
                continue
            if text in t.get("content", ""):
                return True, f"found in turn by @{t['speaker']}"
        who = f"@{speaker_filter}" if speaker_filter else "any speaker"
        return False, f"{text!r} not found in transcript ({who})"

    if atype == "transcript_count":
        turns = _load_turns(transcript_path)
        member_turns = [
            t for t in turns
            if t.get("speaker") not in ("orchestrator", "human")
        ]
        expected = int(a["count"])
        actual = len(member_turns)
        if actual == expected:
            return True, f"{actual} member turn(s)"
        return False, f"expected {expected} turn(s), got {actual}"

    return False, f"unknown assertion type {atype!r}"


def _load_turns(transcript_path: Path) -> list[dict]:
    """Load all turns from a ``transcript.jsonl`` file."""
    if not transcript_path.is_file():
        return []
    result = []
    for line in transcript_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return result
