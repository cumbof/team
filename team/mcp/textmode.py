"""Text-mode tool calling: fenced ``tool:`` blocks with JSON bodies.

Text mode is the front-half syntax that lets *any* model (including small local
models with no native tool-calling template) invoke tools purely through
prompting.  It converges on the same :class:`~team.mcp.bus.MemberToolset` as
native mode — this module only parses the model's text into an ``(wire_name,
arguments)`` pair and renders the prompt-side protocol.

Block format::

    ```tool:workspace__write_file
    {"path": "report.md", "content": "# Findings\\n..."}
    ```

The body is JSON.  As a fallback for small models, when the body is not a JSON
object and the tool has exactly one required string parameter, the raw body
becomes that parameter's value — preserving the "raw code in the fence" idiom
for ``code__run_python`` / ``code__run_bash``.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from team.mcp.bus import ToolInfo

# Widened from the old ``[a-z_]+`` so qualified wire names (server__tool, digits)
# match.
_TOOL_BLOCK_RE = re.compile(
    r"```tool:(?P<name>[A-Za-z0-9_-]+)\n(?P<body>.*?)```",
    re.DOTALL,
)


def parse_tool_blocks(text: str) -> list[tuple[str, str]]:
    """Return ``(wire_name, body)`` pairs for every ``tool:`` block in *text*."""
    return [
        (m.group("name"), m.group("body").strip())
        for m in _TOOL_BLOCK_RE.finditer(text)
    ]


def arg_summary(tool: "ToolInfo") -> str:
    """Compact one-line summary of a tool's top-level arguments.

    e.g. ``{"path": string (required), "content": string (required)}``.
    Nested object schemas are summarised as ``object``.
    """
    schema = tool.input_schema or {}
    props: dict = schema.get("properties", {})
    required = set(schema.get("required", []))
    parts = []
    for name, spec in props.items():
        t = spec.get("type", "any")
        if t == "object" or "properties" in spec:
            t = "object"
        flag = " (required)" if name in required else ""
        parts.append(f'"{name}": {t}{flag}')
    return "{" + ", ".join(parts) + "}"


def parse_tool_body(body: str, tool: "ToolInfo") -> tuple[dict | None, str | None]:
    """Parse a tool block *body* into an arguments dict.

    Returns ``(arguments, None)`` on success or ``(None, error_message)`` on
    failure.  The error message is suitable for injecting back as a tool result
    so the model self-corrects.
    """
    stripped = body.strip()
    err: str
    if stripped:
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as exc:
            err = str(exc)
        else:
            if isinstance(parsed, dict):
                return parsed, None
            err = "expected a JSON object"
    else:
        err = "empty tool body"

    # Fallback: a single required string parameter takes the raw body verbatim.
    schema = tool.input_schema or {}
    required = schema.get("required", [])
    props = schema.get("properties", {})
    if len(required) == 1 and props.get(required[0], {}).get("type") == "string":
        return {required[0]: stripped}, None

    return None, f"ERROR: invalid tool arguments ({err}). Expected: {arg_summary(tool)}"


_HEADER = """\
## Tool use

You have access to external tools.  Invoke a tool by emitting a fenced block
whose info-string is ``tool:<name>`` and whose body is a JSON object of
arguments — as your entire reply or as part of it::

    ```tool:workspace__write_file
    {"path": "report.md", "content": "# Findings\\n..."}
    ```

Rules:
* After each tool block the orchestrator executes the tool and sends you the
  result as a follow-up message — you then continue reasoning.
* You may call multiple tools across several rounds; each round counts against
  your turn budget.
* Once you are done, write your **final reply in plain text** (no tool blocks)
  so it is recorded in the transcript.
* Never invent tool results — always wait for the actual output.
"""

_CODE_GUIDANCE = """\
* **You have full system access** via ``code__run_python`` / ``code__run_bash``.
  If a library or CLI you need is missing, install it first::

    ```tool:code__run_bash
    {"command": "pip install scikit-learn seaborn --quiet"}
    ```
"""


def render_tool_protocol(tools: "list[ToolInfo]", *, code_enabled: bool = False) -> str:
    """Render the text-mode tool protocol section listing *tools* compactly."""
    lines = [_HEADER]
    if code_enabled:
        lines.append(_CODE_GUIDANCE)
    lines.append("Available tools:")
    for t in tools:
        desc = (t.description or "").strip().split("\n")[0]
        lines.append(f"* {t.wire_name} — {desc} args: {arg_summary(t)}")
    return "\n".join(lines)
