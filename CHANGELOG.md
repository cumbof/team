# Changelog

## 0.18.0 — MCP-native tool architecture

This release replaces `team`'s bespoke tool and skill systems with the
[Model Context Protocol](https://modelcontextprotocol.io) (MCP) as the single,
first-class tool layer. Built-in tools and external plugins are now
indistinguishable — every tool call flows through one uniform bus.

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
- **The skills system is removed.** `team/skills.py`, the `team.skills`
  entry-point group, and the `skills:` config key are gone, along with the
  `exec()`-based (and remote-URL) skill loader.
  - Python tool skills → **MCP servers**: declare external `stdio`/`http`
    servers under `mcp_servers:`, or register in-process servers via the new
    `team.mcp_servers` entry-point group (`transport: entry_point`).
  - Markdown context skills → the new **`extra_context:`** config key (a list
    of Markdown file paths, relative to the team YAML, injected into the system
    prompt).
- **`team forge` scaffolds a `servers/` slot** and the `team.mcp_servers`
  entry-point group instead of `skills/` / `team.skills`. The extension CLI's
  `skills` subcommand is now `servers`.

### Added

- `mcp_servers:` config block — connect any MCP server over stdio or Streamable
  HTTP. Stdio servers receive the shared workspace path via a `TEAM_WORKSPACE`
  environment variable (injected automatically).
- `extra_context:` config key on `defaults` and members.
- `examples/mcp/team_helpers_server.py` — an example stdio MCP server bundling
  the former task-board / transcript-search / progress / critique helpers.
- `examples/mcp_team.yaml` — a runnable MCP-native team.

### Fixed

- Native tool mode previously dropped the `peers` registry when dispatching
  federation tools, silently breaking named-peer delegation. All tool state now
  flows through per-member server instances, fixing this.
