# Agent Instructions

This file provides context and guidelines for AI agents working on this repository.

## Project Overview

`team` (PyPI: `team-core`) is a Python framework for orchestrating a cluster of
containerized local LLMs. Each LLM ("member") runs in its own Docker + Ollama
container with its own persona, role, and model. An orchestrator drives a
turn-based conversation between members according to a chosen workflow strategy
until a shared goal is reached or a round limit is hit.

Key traits that shape the codebase:

- Pure Python (≥ 3.9), no compiled extensions.
- Docker is a **runtime** dependency (containers), not a build dependency.
- All user-facing configuration is YAML; internal representation uses dataclasses.
- The package is installed as `team-core`; the CLI entry-point is `team`.

## Repository Layout

```
team/                  # main package
  cli.py               # Click CLI (`team run`, `team new`, `team stats`, …)
  config.py            # YAML → TeamConfig / MemberConfig dataclasses
  orchestrator.py      # Orchestrator: ties workflow + members + workspace
  member.py            # Member: couples config with its container and Ollama client
  workflows.py         # Pluggable turn schedulers (round_robin, manager_driven, …)
  tools.py             # 19 built-in agent tools (run_python, web_search, …)
  skills.py            # Skill loader: Python (.py) or Markdown (.md) extensions
  personas.py          # System-prompt renderer
  persona_library.py   # @-shorthand resolver (e.g. @pi → personas/pi.yaml)
  beliefs.py           # Shared team belief board (SQLite-backed)
  memory.py            # Per-member persistent memory (SQLite-backed)
  bus.py               # Transcript (append-only log of turns)
  container.py         # Docker container lifecycle helpers
  ollama_client.py     # Thin HTTP client for Ollama / OpenAI-compatible APIs
  workspace.py         # SharedWorkspace and CheckpointManager
  bridge.py            # Cross-team federation (HTTP bridge server + client)
  pipeline.py          # Multi-stage team pipelines
  pricing.py           # Cost estimation for cloud models
  export.py            # Run statistics and Markdown report export
  replay.py            # Interactive transcript viewer
  wizard.py            # `team new` interactive YAML wizard
  visualize.py         # `team visualize` workflow graph renderer

personas/              # Ready-made persona YAML files (@pi, @engineer, …)
skills/                # Built-in skill files (.py tools + .md context injections)
examples/              # Example team YAML configs (academic_lab, software_team)
tests/                 # Pytest unit tests (one file per module)
  integration/         # End-to-end tests that require Docker + live Ollama
.github/workflows/     # CI: tests.yml (pytest), publish.yml (PyPI release)
```

## Development Setup

```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Run the unit test suite
pytest -q

# Run integration tests (requires Docker daemon + Ollama image)
pytest -m integration
```

CI runs `pytest -q` (unit tests only) against Python 3.10, 3.11, and 3.12.
Integration tests are excluded from CI by default — do not change the `pytest.ini`
`markers` configuration without updating the CI workflow too.

## Core Concepts

**TeamConfig / MemberConfig** (`config.py`)  
All configuration lives in a YAML file. `load_team(path)` parses it and returns
a `TeamConfig` dataclass with fully resolved defaults. Never reach into raw YAML
dicts outside of `config.py`.

**Orchestrator** (`orchestrator.py`)  
Owns the run lifecycle: spins up containers, creates `Member` instances, picks the
workflow, drives turns, manages checkpoints, and tears down on exit. The public
surface used by workflows is: `run_turn(name, prompt=None)`, `members`, and
`transcript`.

**Member** (`member.py`)  
Couples a `MemberConfig` with its `MemberRuntime` (Docker) and `OllamaClient`.
`Member.take_turn()` builds the prompt, calls the LLM, parses tool blocks (or
native function calls), executes tools, and returns a `TurnResult`.

**Workflows** (`workflows.py`)  
A workflow is a plain function `run(orch: Orchestrator) -> None`. Available:
`round_robin`, `manager_driven`, `review_loop`, `sequential_chain`, `debate`,
`parallel_review`. Register a new workflow in `get_workflow()` at the bottom of
the file.

**Tools** (`tools.py`)  
Tool calls are either *text mode* (fenced `` ```tool:<name> `` blocks in the LLM
reply) or *native mode* (OpenAI function-calling JSON). The orchestrator parses
the reply, calls `execute_tool(name, body, ...)`, and injects the result back
before the final turn reply is recorded. The sentinel `DONE_TOKEN = "[[TEAM_DONE]]"`
signals the workflow to stop.

**Skills** (`skills.py`)  
Extend the toolbox without touching core code. A `.py` skill adds new callable
tools; a `.md` skill injects its content into the member's system prompt as
background knowledge. Skills can be local files or remote URLs — treat remote
skill URLs as arbitrary code execution.

**Personas** (`personas.py`, `persona_library.py`)  
A persona is a YAML file with `role`, `goal`, and `system_prompt` fields. Members
can reference personas by `@name` shorthand (resolved from `personas/` or the
`TEAM_PERSONA_DIR` environment variable).

**Workspace** (`workspace.py`)  
`SharedWorkspace` is a host directory mounted into all containers. `context.md`
placed at the workspace root is automatically injected into every member's context
on every turn. `CheckpointManager` snapshots the workspace before each turn so
`team rollback` can restore any prior state.

## Coding Conventions

- Use `from __future__ import annotations` at the top of every module.
- Prefer dataclasses (not dicts) for structured data passed between modules.
- Keep Docker/container logic inside `container.py`; keep LLM HTTP calls inside
  `ollama_client.py`. Do not scatter `requests` calls across modules.
- New built-in tools go in `tools.py` — add the callable to `TOOLS`, a one-line
  description to `TOOL_DESCRIPTIONS`, and a JSON Schema entry to `TOOL_SCHEMAS`.
- New workflow strategies go in `workflows.py` — add the function and register it
  in `get_workflow()`.
- Module-level docstrings are mandatory for every new module and should follow the
  existing format (summary paragraph + usage example + section headers).
- Use the standard `logging` module (`log = logging.getLogger(__name__)`); never
  `print()` to stdout in library code (CLI rendering uses `rich`).

## Testing Guidelines

- Every new module or feature must have a corresponding `tests/test_<module>.py`.
- Unit tests must **not** require Docker or network access. Mock `ContainerManager`,
  `OllamaClient`, and any `requests` calls with `pytest-mock`.
- Integration tests go in `tests/integration/` and must be marked `@pytest.mark.integration`.
- Do not remove or weaken existing assertions; do not add `# noqa` or `# type: ignore`
  without a comment explaining why.
- Run `pytest -q` locally before opening a PR and confirm it passes on all three
  supported Python versions if possible.

## Important Gotchas

- `run_python` and `run_bash` tools execute code **on the host machine** with the
  privileges of the `team` process. Never enable them for untrusted member prompts.
- Do not import optional feature modules (`memory`, `beliefs`) at the top level —
  they are guarded behind `TYPE_CHECKING` or lazy imports to keep startup fast.
- `TeamConfigError` is the canonical exception for malformed YAML; raise it (not
  `ValueError` directly) from `config.py`.
- `[[TEAM_DONE]]` is the stop signal for all workflows. If you add a new workflow,
  respect `TurnResult.declared_done` to honour it.
- Adding a new required field to `TeamConfig` or `MemberConfig` is a breaking
  change — always provide a default or a migration path.

## Pull Request Expectations

- Disclose AI assistance in the PR description (e.g., *"co-authored with GitHub
  Copilot"*) so reviewers can calibrate their review.
- Include or update tests for every behaviour change.
- Do not commit secrets, credentials, or real Ollama model weights.
- Keep PRs focused; separate refactors from feature additions.
