# Changelog

## 0.18.0 — MCP-native tool architecture

This release replaces `team`'s bespoke tool system and the tool-loading half
of its skill system with the
[Model Context Protocol](https://modelcontextprotocol.io) (MCP) as the single,
first-class tool layer. Built-in tools and external plugins are now
indistinguishable — every tool call flows through one uniform bus. The
context-injection half of skills survives, narrowed to Markdown only.

### Breaking changes

- **Python 3.10+ required** (was 3.9). Adds a dependency on the official
  `mcp` SDK (`mcp>=1.9,<2`).
- **Tool names are now qualified** as `server/tool` (or `server/*` wildcards)
  in the `tools:` list. Bare names are a config error:
  - `web_search` → `web/web_search`
  - `run_python` / `run_bash` → `code/run_python` / `code/run_bash`
  - `read_file`, `write_file`, `append_file`, `list_files` → `workspace/...`
  - `remember`, `recall`, `forget`, `list_memories` → `memory/...`
  - `assert_belief`, `contest_belief`, `accept_belief`, `list_beliefs` → `beliefs/...`
  - `log_decision`, `read_decisions` → `decisions/...`
  - `delegate_task`, `broadcast_task`, `list_peers`, `cancel_remote_task`,
    `sync_beliefs`, `query_registry` → `federation/...`
  - `delegate_to_expert` → `expert/delegate_to_expert`

  The 25 built-in tools are grouped into 8 reserved servers: `code`, `web`,
  `workspace`, `memory`, `beliefs`, `decisions`, `federation`, `expert`.
- **Text-mode tool bodies are now JSON.** A `tool:` block body is a JSON object
  of arguments; the model-visible tool name is `server__tool`:

  ````
  ```tool:workspace__write_file
  {"path": "report.md", "content": "# Findings"}
  ```
  ````

  As a fallback for small models, a tool with exactly one required string
  argument (e.g. `code__run_python`) still accepts a raw body verbatim.
- **The skills system is now Markdown-only.** `team/skills.py` is rewritten:
  the `exec()`-based (and remote-URL, checksum-verified) Python tool loader is
  gone, along with the `skills:` config key.
  - Python tool skills → **MCP servers**: declare external `stdio`/`http`
    servers under `mcp_servers:`, or register in-process servers via the
    `team.mcp_servers` entry-point group (`transport: entry_point`).
  - Markdown context skills → the **`extra_context:`** config key. Each entry
    is tried as a relative path first, falling back to a name registered via
    the (still-present, now Markdown-only) `team.skills` entry-point group —
    a thin Python module pointing at a bundled `.md` file via `SKILL_FILE`,
    same idea as before minus the tool-loading half.
- **`team forge` scaffolds both a `servers/` slot** (`team.mcp_servers`) **and
  a `skills/` slot** (`team.skills`, Markdown only). The extension CLI keeps
  separate `servers` and `skills` subcommands — `skills` no longer lists
  Python tools, only Markdown context bundles.

### Added

- `mcp_servers:` config block — connect any MCP server over stdio or Streamable
  HTTP. Stdio servers receive the shared workspace path via a `TEAM_WORKSPACE`
  environment variable (injected automatically).
- `extra_context:` config key on `defaults` and members, resolving each entry
  as a relative path or a registered `team.skills` name.
- `examples/mcp/team_helpers_server.py` — an example stdio MCP server bundling
  the former task-board / transcript-search / progress / critique helpers.
- `examples/mcp_team.yaml` — a runnable MCP-native team.

### Fixed

- Native tool mode previously dropped the `peers` registry when dispatching
  federation tools, silently breaking named-peer delegation. All tool state now
  flows through per-member server instances, fixing this.
