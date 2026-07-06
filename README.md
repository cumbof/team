# team

Orchestrate a cluster of containerized local LLMs — each with its own
persona, role, and goal — that collaborate until the work is done.

![PyPI - Version](https://img.shields.io/pypi/v/team-core)
![Build Status](https://img.shields.io/github/actions/workflow/status/cumbof/team/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/cumbof/team/blob/main/LICENSE)

![team](https://raw.githubusercontent.com/cumbof/team/refs/heads/main/assets/logo.png)

<p align="center"><b>⭐ Star this repository to stay updated with new releases ⭐</b></p>
<br>
<br>

`team` lets you describe a small "organisation" of LLMs in a single YAML
file and then bring it to life: every member runs in **its own isolated
Docker container** with its own [Ollama](https://ollama.com/) daemon and
its own model, the orchestrator drives a turn-based conversation between
them, and the members produce real artifacts (code, manuscripts, reports,
…) in a shared workspace.

You can mix and match model sizes per role — e.g. a 70B generalist as a
Principal Investigator, a 7B coder as a Data Scientist, an 8B model as a
Reviewer — and pick a workflow that matches how the work should flow:
**round-robin**, **manager-driven**, or **review-loop until consensus**.

> [!WARNING]  
>
> **Work in Progress:** This repository is currently under active development.
> While the core functionality is present, some features may be incomplete or
> not fully work as expected, and you may encounter unexpected bugs. Please
> test thoroughly before using this in any critical pipelines.

> [!NOTE]
>
> A significant portion of the code and documentation in this repository
> was written **with the assistance of a Large Language Model (LLM)**.
> All LLM-generated contributions have been reviewed, tested, and curated
> by the human maintainers, but — as with any software — bugs may exist.
> Please review the code critically, run the test suite, and open an issue
> if you find something unexpected.
>
> **Pull requests are very welcome**, including those written or
> co-authored with the help of an LLM.  We only ask that you review and
> test your changes before submitting, and disclose AI assistance in your
> PR description (e.g. *"co-authored with GitHub Copilot"*) so reviewers
> can calibrate their review accordingly.

---

## Feature overview

| Feature | Description |
| --- | --- |
| **Containerised members** | Every LLM runs in its own Docker + Ollama container with configurable CPU, RAM, and GPU limits. |
| **Flexible workflows** | `round_robin`, `manager`, `review_loop`, `sequential_chain`, `debate`, `parallel_review` — pick or combine. |
| **Shared workspace** | Members read and write real files (code, reports, data) to a host directory. |
| **Agent tool use** | 25 built-in tools across 8 MCP servers (`code`, `web`, `workspace`, `memory`, `beliefs`, `decisions`, `federation`, `expert`); `tool_mode: text` (fenced blocks, JSON body) or `tool_mode: native` (OpenAI/Ollama function-calling API); connect any external MCP server over stdio or Streamable HTTP. |
| **Predefined persona library** | 16 ready-made personas (`@pi`, `@engineer`, `@reviewer` …) stored as individual YAML files in `personas/`; extend with your own via `TEAM_PERSONA_DIR`. |
| **Per-agent persistent memory** | SQLite-backed memory that survives between runs; agents `remember` and `recall` across sessions. |
| **Shared team belief board** | Structured collective knowledge with confidence scores, voting, and consensus tracking. |
| **Cross-team federation (bridge)** | Two independent `team` clusters can delegate tasks to each other over HTTP — academic-lab-style collaboration. |
| **Shared institutional context** | Drop a `context.md` in the workspace root and every member sees it on every turn — no per-member config needed. |
| **Decision log** | Members call `log_decision` to append timestamped, rationale-rich entries to `decisions.md`; any member can `read_decisions` at any time. |
| **Workspace time-travel** | `team rollback` restores the workspace to any past checkpoint and lets you resume from there. |
| **Human-in-the-loop** | Interrupt a live run, read the transcript, inject a message, and let the team continue. |
| **OpenAI-compatible backends** | Swap Ollama for any OpenAI-compatible API (GPT-4o, Mistral, Together AI, …) per member. |
| **Context window management** | `sliding_window`, `truncate`, or `summarize` strategies keep long runs within token budgets. |
| **Workspace checkpoints** | Automatic snapshots before every member turn; `team restore` rolls back to any point. |
| **Run statistics & reports** | Per-member token usage, turn counts, elapsed time — exportable as a Markdown report. |
| **Interactive wizard** | `team new` walks you through YAML creation. |
| **Structured JSON output** | Force a member to reply with valid JSON; optionally validate against a JSON Schema with automatic retry. |
| **Per-turn timeout** | Hard wall-clock deadline per member turn; raises `TurnTimeoutError` if the LLM doesn't respond in time. |
| **`team test`** | Define assertions in the YAML and run them automatically after a team workflow to verify outputs in CI. |
| **Parallel member execution** | `workflow: type: parallel` — all members run simultaneously in each round, bounded by the slowest rather than the sum. |
| **`team replay`** | Step through a saved transcript turn-by-turn in an interactive terminal viewer; navigate, search by speaker, and view stats. |
| **Token budget** | Hard-cap total tokens per member per run; gracefully stops with `TokenBudgetError` when exhausted. |
| **Conditional routing** | Members declare the next speaker via simple YAML rules (`if_contains`, `if_match`, `default`), enabling dynamic branching and state-machine-like workflows. |
| **LLM retry with backoff** | Automatic retry with exponential backoff on transient errors (5xx, connection refused, timeout); configurable per member. Raises `LLMRetryExhaustedError` when all attempts fail. |
| **Cost estimation** | Estimated USD cost displayed in the token-usage table after every run (`team run`, `team stats`). Built-in pricing for OpenAI, Anthropic, Google, and Mistral; local Ollama models show `$0.00 (local)`. |
| **Multi-team pipelines** | Chain multiple team runs with `team pipeline`; upstream artifacts and transcript summaries are automatically injected into downstream stages via `inject_files`, `inject_context`, and `goal_override` templates. |
| **Team registry (service discovery)** | A lightweight HTTP directory where running team clusters advertise their capabilities (tags, models, tools). Other teams discover and delegate to specialist clusters via `query_registry` or the CLI. |
| **Federated belief board** | Independent team clusters share their collective knowledge across the bridge. Pull accepted beliefs from a partner team (they arrive as pending for local consensus), push local beliefs outward, or bidirectional sync — via the `sync_beliefs` tool or `team beliefs-sync` CLI. |

---

## Table of contents

- [Why?](#why)
- [How it works](#how-it-works)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Defining a team](#defining-a-team)
  - [Top-level fields](#top-level-fields)
  - [`defaults`](#defaults)
  - [`workflow`](#workflow)
  - [`members`](#members)
- [The collaboration protocol](#the-collaboration-protocol)
- [Predefined persona library](#predefined-persona-library)
  - [How personas are stored](#how-personas-are-stored)
  - [Available personas](#available-personas)
  - [Using a persona in YAML](#using-a-persona-in-yaml)
  - [Adding your own personas](#adding-your-own-personas)
- [Workflows](#workflows)
- [Workspaces and artifacts](#workspaces-and-artifacts)
- [Containers, isolation, and root](#containers-isolation-and-root)
- [GPU support](#gpu-support)
  - [Apple Silicon / no-Docker Ollama](#apple-silicon--no-docker-ollama)
- [OpenAI-compatible backends](#openai-compatible-backends)
- [Remote / no-Docker Ollama](#remote--no-docker-ollama)
- [Custom Ollama image](#custom-ollama-image)
- [Context window management](#context-window-management)
- [Model retention (`keep_alive`)](#model-retention-keep_alive)
- [CLI reference](#cli-reference)
- [Interactive wizard](#interactive-wizard)
- [Pre-flight checks](#pre-flight-checks)
- [Streaming output](#streaming-output)
- [Per-turn timeout](#per-turn-timeout)
- [LLM retry with backoff](#llm-retry-with-backoff)
- [Resuming an interrupted run](#resuming-an-interrupted-run)
- [Human-in-the-loop intervention](#human-in-the-loop-intervention)
- [Agent mode and tool use](#agent-mode-and-tool-use)
  - [Available built-in tools](#available-built-in-tools)
  - [Connecting MCP servers](#connecting-mcp-servers)
- [Shared institutional context](#shared-institutional-context)
- [Decision log](#decision-log)
- [Structured JSON output](#structured-json-output)
- [Conditional routing](#conditional-routing)
- [Token budget](#token-budget)
- [Per-agent persistent memory](#per-agent-persistent-memory)
  - [Enabling memory](#enabling-memory)
  - [Memory tools](#memory-tools)
  - [Memory config reference](#memory-config-reference)
- [Shared team belief board](#shared-team-belief-board)
  - [Enabling the belief board](#enabling-the-belief-board)
  - [Belief tools](#belief-tools)
  - [Inspecting beliefs with team beliefs](#inspecting-beliefs-with-team-beliefs)
  - [Belief config reference](#belief-config-reference)
- [Workspace checkpoints](#workspace-checkpoints)
- [Workspace time-travel (`team rollback`)](#workspace-time-travel-team-rollback)
- [Token usage tracking](#token-usage-tracking)
- [Cost estimation](#cost-estimation)
- [Run statistics](#run-statistics)
- [Exporting a run report](#exporting-a-run-report)
- [`team replay` — interactive transcript browser](#team-replay--interactive-transcript-browser)
- [Automated testing with `team test`](#automated-testing-with-team-test)
- [Multi-team pipelines](#multi-team-pipelines)
- [Cross-team collaboration (bridge)](#cross-team-collaboration-bridge)
  - [How it works](#how-it-works-1)
  - [Exposing a team as a bridge server](#exposing-a-team-as-a-bridge-server)
  - [Delegating work from another team](#delegating-work-from-another-team)
  - [Named peer registry](#named-peer-registry)
  - [Broadcasting to multiple teams](#broadcasting-to-multiple-teams)
  - [Cancelling a remote task](#cancelling-a-remote-task)
  - [Server HTTP API reference](#server-http-api-reference)
  - [Bridge config reference](#bridge-config-reference)
  - [Security — HMAC-SHA256 shared secret](#security--hmac-sha256-shared-secret)
  - [Additional security considerations](#additional-security-considerations)
- [Team registry (service discovery)](#team-registry-service-discovery)
- [Federated belief board](#federated-belief-board)
- [Examples](#examples)
- [Architecture overview](#architecture-overview)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Why?

A single LLM is a generalist. Real work — research, engineering, writing —
is usually done by **several specialists** that disagree, revise, and
converge.  `team` makes it easy to assemble such a group locally:

* **Heterogeneous models, one per role.** Use a small, fast model for
  routine tasks and a large model only where it matters.
* **Strong isolation.** Every member is a separate `ollama serve`
  process in a separate container, on a private Docker network, with its
  own model cache.  A misbehaving member cannot reach into another's
  filesystem, network namespace, or model store.
* **Real deliverables.** Members write actual files (code, prose, data)
  into a shared workspace; you keep them after the run.
* **Pluggable workflows.** Pick how the team coordinates — and add your
  own in a few lines of Python.

---


## How it works

```
                 ┌────────────────── orchestrator (host) ───────────────────┐
                 │                                                          │
                 │   transcript.jsonl     shared workspace (./runs/<team>)  │
                 │        ▲                       ▲                         │
                 │        │ append every turn     │ files written by members│
                 └────┬───┴────────────┬──────────┴─────────────┬───────────┘
                      │                │                        │
                      ▼                ▼                        ▼
       ┌──────────────────┐  ┌───────────────────┐     ┌──────────────────┐
       │ container: pi    │  │ container: postdoc│     │ container: ...   │
       │ ollama serve     │  │ ollama serve      │     │                  │
       │ model: 70B       │  │ model: 8B         │     │                  │
       │ /workspace (ro+) │  │ /workspace (ro+)  │     │ /workspace (ro+) │
       │ /private         │  │ /private          │     │ /private         │
       └──────────────────┘  └───────────────────┘     └──────────────────┘
                       \\              |                //
                        \\             |               //
                       team-<name>-net (private bridge network)
```

For each member, the orchestrator:

1. Starts a dedicated Ollama container, on a per-team Docker network, with
   the team's shared workspace bind-mounted at `/workspace` and a
   per-member private workspace at `/private`.
2. Pulls the model the member is configured to use (cached in the
   member's own named Docker volume).
3. Builds a system prompt from the member's persona, the team goal, the
   list of teammates, and the [collaboration protocol](#the-collaboration-protocol).
4. Asks the chosen [workflow](#workflows) to drive the conversation.

At every turn the orchestrator hands the speaking member the **full
shared transcript** plus a snapshot of the workspace; the member's reply
is parsed for fenced `file:` blocks (which become real files on disk) and
for control tokens (`[[TEAM_DONE]]`, `NEXT: @<member>`, `APPROVED`, …).

---


## Requirements

* **Linux** host (tested) — macOS works if Docker Desktop has enough
  resources for your models.
* **Docker** (engine ≥ 20.10) reachable by the host user.
* **Python 3.9+**.
* For GPU acceleration: NVIDIA GPU + the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
* **Disk and RAM/VRAM** sized for your largest model — Ollama itself is
  small but model weights aren't.

---


## Installation

Install from PyPI:

```bash
pip install team-core
```

Or clone the repository for the latest development version:

```bash
git clone https://github.com/cumbof/team.git
cd team
python -m venv .venv
. .venv/bin/activate
pip install -e .
```

Installs the `team` CLI into your virtualenv.  Verify:

```bash
team --version
team --help
```

For development extras (pytest):

```bash
pip install -e ".[dev]"
pytest -q
```

---


## Quick start

1. Generate a starter spec:

   ```bash
   team init my-team.yaml
   ```

2. Edit `my-team.yaml`: pick model names that exist in Ollama, write a
   real `goal`, and tweak the personas.

3. Run it end-to-end (containers come up, models get pulled if needed,
   workflow runs, containers come down):

   ```bash
   team run my-team.yaml
   ```

4. Inspect the deliverables:

   ```bash
   ls runs/my-team/shared/
   team transcript my-team.yaml
   ```

5. Or manage the lifecycle by hand:

   ```bash
   team up my-team.yaml          # start all member containers
   team status my-team.yaml      # show container state
   team logs my-team.yaml        # tail Ollama logs per member
   team run my-team.yaml --no-up --keep-up   # run more rounds
   team run my-team.yaml --resume            # resume after a crash
   team down my-team.yaml --purge            # tear down + delete model caches
   ```

---


## Defining a team

A team is a single YAML file.  Annotated minimal example:

```yaml
name: my-team                # [a-z][a-z0-9_-]{0,30}
goal: |
  Plain-English statement of what the team must accomplish.

workspace: ./runs/my-team    # host directory; created on demand

workflow:
  type: round_robin          # round_robin | manager | review_loop
  max_rounds: 6

defaults:
  ollama_image: ollama/ollama:latest
  context_window: 8192
  temperature: 0.4
  gpus: none                 # "all" | "none" | [0, 1, ...]
  memory_limit: "16g"        # optional Docker memory cap per member
  cpu_limit: 4               # optional Docker CPU cap per member (cores)
  pull_timeout: 1800
  request_timeout: 600

members:
  - name: lead
    role: Project Lead
    model: llama3.1:8b
    persona: |
      You coordinate the team.
  - name: worker
    role: Engineer
    model: qwen2.5-coder:7b
    persona: |
      You implement code and produce concrete artifacts.
```

### Top-level fields

| field | required | description |
| --- | --- | --- |
| `name` | yes | DNS-safe team name; used in container/volume/network names. |
| `goal` | yes | The shared objective every member sees in its system prompt. |
| `workspace` | no | Host directory for shared/private workspaces and the transcript.  Defaults to `./runs/<name>`. |
| `workflow` | no | See below.  Defaults to `round_robin` with 6 rounds. |
| `defaults` | no | Defaults inherited by every member that doesn't override them. |
| `members` | yes | Non-empty list of member specs (see below). |

### `defaults`

| key | type | default | meaning |
| --- | --- | --- | --- |
| `ollama_image` | string | `ollama/ollama:latest` | Image used for member containers. |
| `context_window` | int | `8192` | `num_ctx` passed to Ollama (`/api/chat` `options`). |
| `temperature` | float | `0.4` | Sampling temperature. |
| `top_p` | float | `0.9` | Top-p sampling. |
| `memory_limit` | string | unset | Docker `mem_limit` per member (e.g. `"12g"`). |
| `cpu_limit` | float | unset | Docker CPU cap per member (cores; e.g. `4`). |
| `gpus` | str / list | `none` | `"all"`, `"none"`, or list of GPU indices. |
| `pull_timeout` | int | `1800` | Seconds allowed for a model pull. |
| `request_timeout` | int | `600` | HTTP timeout per chat call. |
| `backend` | string | `ollama` | LLM backend: `"ollama"` or `"openai_compat"`. |
| `api_key` | string | unset | API key for `openai_compat` backend; supports `"env:VAR"`. |
| `context_strategy` | string | `none` | Context management: `"none"`, `"sliding_window"`, `"truncate"`, `"summarize"`. |
| `context_budget` | int | `0` | Budget for context management: max turns (`sliding_window`) or approx token count (`truncate`/`summarize`). |
| `tools` | list | `[]` | Tools enabled for all members by default, as qualified `server/tool` or `server/*` patterns (e.g. `code/*`, `workspace/read_file`). |
| `max_tool_rounds` | int | `10` | Maximum agentic tool-call rounds per member turn. |
| `tool_timeout` | int | `300` | Seconds budget per individual tool execution (generous default to allow package installs). |
| `tool_mode` | string | `"text"` | Tool invocation mode: `"text"` (fenced blocks, JSON body) or `"native"` (LLM function-calling API). |
| `extra_context` | list | `[]` | Markdown files (paths relative to the team YAML) injected into every member's system prompt. |
| `ollama_url` | string | unset | Route **all** members to an existing Ollama instance at this URL instead of starting Docker containers. Per-member `ollama_url` overrides this. See [Apple Silicon / no-Docker](#apple-silicon--no-docker-ollama). |
| `keep_alive` | string | `"-1"` | How long Ollama keeps a model loaded in RAM after a request. `"-1"` (default) means keep forever — models stay resident between turns. Accepts any Ollama duration string (`"5m"`, `"1h"`) or `"0"` to unload immediately after each call. |

### `workflow`

```yaml
workflow:
  type: review_loop
  max_rounds: 4
  producer: postdoc
  reviewer: reviewer
  approve_token: APPROVED   # only review_loop; default "APPROVED"
  manager: tech_lead        # only when type=manager
  prompt_template: |        # only sequential_chain; {prev_speaker} and {prev_content} available
    @{prev_speaker} produced the following. Refine it:
    {prev_content}
```

| `type` | extra options |
| --- | --- |
| `round_robin` | none |
| `manager` | `manager: <member name>` |
| `review_loop` | `producer: <member>`, `reviewer: <member>`, optional `approve_token` |
| `sequential_chain` | optional `prompt_template` (supports `{prev_speaker}`, `{prev_content}`) |
| `debate` | `pro: <member>`, `con: <member>`, `judge: <member>`, optional `rounds` |
| `parallel_review` | `producer: <member>`, `reviewers: [m1, m2, …]` (≥2), `synthesizer: <member>`, optional `approve_token` |

### `members`

| key | required | notes |
| --- | --- | --- |
| `name` | yes | DNS-safe; used as `@handle` in the protocol. |
| `role` | yes | Free-text role label. |
| `model` | yes | Any tag known to Ollama (`llama3.1:8b`, `qwen2.5-coder:7b`, …). |
| `persona` | yes | Free-text persona prompt; quoted block. |
| `temperature`, `top_p`, `context_window` | no | Per-member overrides of `defaults`. |
| `memory_limit`, `cpu_limit`, `gpus` | no | Per-member resource overrides. |
| `can_write_files` | no | Default `true`; set to `false` to forbid this member from creating files. |
| `extra_system` | no | Free-form text appended to the rendered system prompt. |
| `ollama_url` | no | Connect to an existing Ollama instance directly; skips Docker. |
| `backend` | no | `"ollama"` (default) or `"openai_compat"` — overrides `defaults.backend`. |
| `api_base` | no | Base URL for the OpenAI-compat API (required when `backend: openai_compat`). |
| `api_key` | no | API key; supports `"env:VAR"` to read from an environment variable. |
| `context_strategy` | no | Per-member override of context management strategy. |
| `context_budget` | no | Per-member override of context budget. |
| `tools` | no | Qualified tool patterns this member may use (e.g. `[web/web_search, code/run_python]`). |
| `max_tool_rounds` | no | Per-member override of the tool-round limit. |
| `tool_timeout` | no | Per-member override of the per-tool execution timeout (seconds, default 300). |
| `tool_mode` | no | Per-member override: `"text"` or `"native"` (default inherits from `defaults.tool_mode`). |
| `extra_context` | no | Member-specific Markdown context files (override `defaults.extra_context`). |
| `keep_alive` | no | Per-member override for Ollama model retention (e.g. `"5m"`, `"-1"`). Inherits from `defaults.keep_alive` when absent. |

---


## The collaboration protocol

Every member receives a system prompt that includes a small,
deterministic protocol so the orchestrator can parse replies reliably:

* **Address a teammate**: prefix a section with `@<member>:`.
* **Write or overwrite a file in the shared workspace**: emit a fenced
  block with an `file:` info-string, e.g.

  ````
  ```file:manuscript/manuscript.md
  # Title
  ...
  ```
  ````

  The orchestrator atomically writes the body to that path under
  `<workspace>/shared/`.  Path-traversal attempts (`..`) are rejected.
* **Private workspace**: each member has `/private` inside its container
  (mapped to `runs/<name>/members/<member>/` on the host) for personal
  scratch files, drafts, and notes that are not shared with the team.
  The list of files currently in `/private` is shown at the top of each
  of the member's turn prompts.
* **Declare the goal achieved**: end the reply with a line containing
  exactly `[[TEAM_DONE]]`.  Workflows interpret this as "stop now".
* **Manager workflow**: end the reply with `NEXT: @<member>` to nominate
  who speaks next.
* **Review-loop workflow**: the reviewer emits `APPROVED` (configurable)
  when the deliverable is ready.

---


## Predefined persona library

Writing a good persona from scratch takes time.  `team` ships with
**16 ready-made personas** spanning academic research, software engineering,
and general-purpose roles.  Each persona lives in its own YAML file under
`personas/` at the root of this repository — making them easy to read,
edit, and contribute back to the project.

### How personas are stored

```
personas/
├── pi.yaml            # Principal Investigator
├── postdoc.yaml       # Postdoctoral Researcher
├── phd.yaml           # PhD Student
├── reviewer.yaml      # Critical Reviewer
├── statistician.yaml  # Statistician
├── bioinformatician.yaml
├── ml_researcher.yaml
├── architect.yaml
├── engineer.yaml
├── qa.yaml
├── devops.yaml
├── tech_writer.yaml
├── analyst.yaml
├── writer.yaml
├── manager.yaml
└── ethicist.yaml
```

Each file follows the same simple format:

```yaml
role: Principal Investigator
description: Lab director — sets research direction, evaluates results, writes grants.
persona: |
  You are a tenured Principal Investigator at a research university.
  Your role is to set and guard the scientific direction of the project.
  ...
```

The filename stem (e.g. `pi` from `pi.yaml`) becomes the `@`-key used in team
YAML files.

### Available personas

| Key | Role | Description |
| --- | --- | --- |
| `@pi` | Principal Investigator | Lab director — sets research direction, evaluates results, writes grants. |
| `@postdoc` | Postdoctoral Researcher | Senior researcher — deep expertise, drives experiments and analysis. |
| `@phd` | PhD Student | Junior researcher — literature review, baseline experiments, drafting. |
| `@reviewer` | Critical Reviewer | Peer-review skeptic — challenges assumptions, finds weaknesses. |
| `@statistician` | Statistician | Statistical methodologist — study design, power, inference correctness. |
| `@bioinformatician` | Bioinformatician | Omics data specialist — pipelines, databases, variant/sequence analysis. |
| `@ml_researcher` | Machine Learning Researcher | ML specialist — model design, training, evaluation, ablations. |
| `@architect` | Software Architect | System designer — API contracts, scalability, tech decisions. |
| `@engineer` | Software Engineer | Implementer — writes production-quality code, debugs, reviews PRs. |
| `@qa` | QA Engineer | Quality assurance — test strategy, edge cases, regression detection. |
| `@devops` | DevOps / SRE | Infrastructure and reliability — CI/CD, monitoring, deployment. |
| `@tech_writer` | Technical Writer | Documentation specialist — clarity, structure, audience-appropriate prose. |
| `@analyst` | Data Analyst | Data explorer — EDA, visualisation, dashboards, business insights. |
| `@writer` | Science Writer | Communicator — translates technical findings into compelling narratives. |
| `@manager` | Project Manager | Coordinator — milestones, blockers, stakeholder communication. |
| `@ethicist` | AI / Research Ethicist | Ethics and compliance — bias, fairness, privacy, responsible use. |

Browse the library from the terminal:

```bash
team personas              # list all personas with key, role, description
team personas pi           # print the full persona text for @pi
team personas engineer     # print the full persona text for @engineer
```

### Using a persona in YAML

Set `persona` to `@<key>` instead of writing a persona block:

```yaml
members:
  - name: alice
    model: llama3.1:70b
    persona: "@pi"              # role is set to "Principal Investigator" automatically
  - name: bob
    model: llama3.1:8b
    persona: "@phd"             # role is "PhD Student"
  - name: carol
    model: qwen2.5:7b
    persona: "@reviewer"        # role is "Critical Reviewer"
```

You can override the default role while keeping the library persona text:

```yaml
  - name: alice
    model: llama3.1:70b
    persona: "@pi"
    role: "Lab Director"        # custom title; persona text stays the same
```

You can also mix library personas with fully custom ones in the same team:

```yaml
members:
  - name: alice
    model: llama3.1:70b
    persona: "@pi"
  - name: custom
    role: Domain Expert
    model: llama3.1:8b
    persona: |
      You are a specialist in protein crystallography with 20 years of
      experimental experience. You validate all structural claims against
      PDB data.
```

### Adding your own personas

**Option 1 — contribute to the built-in library** (share with everyone):

Drop a `.yaml` file into the `personas/` directory at the repo root and submit
a pull request.  The file name becomes the `@`-key.

**Option 2 — project-local personas** (private to your setup):

Point `TEAM_PERSONA_DIR` at any directory; files there are loaded *in addition
to* the built-in library and take precedence over built-in keys with the same
name:

```bash
export TEAM_PERSONA_DIR=~/.team/personas
```

Then add files like `~/.team/personas/clinician.yaml`:

```yaml
role: Clinical Research Collaborator
description: Translates findings into clinical context and regulatory language.
persona: |
  You are a physician-scientist with expertise in clinical trial design.
  You translate pre-clinical findings into clinical hypotheses, identify
  regulatory hurdles (FDA, EMA) early, and ensure the team's outputs are
  framed for a clinical audience.
```

Any team YAML can now use `persona: "@clinician"` once the env var is set.

---


## Workflows

### `round_robin`

Every member speaks in declaration order.  Repeat for `max_rounds` full
rounds, or until a member emits `[[TEAM_DONE]]`.  Useful for brainstorms
and small symmetric teams.

### `manager`

A designated `manager` member opens the work, then after every other
member's turn the manager is asked again to evaluate progress and
nominate the next speaker via `NEXT: @<member>`.  The manager can also
take the floor itself, or end the run with `[[TEAM_DONE]]`.

### `review_loop`

A `producer` writes the first draft.  A `reviewer` critiques it; the
producer revises; repeat until the reviewer emits `APPROVED` (or
`max_rounds` revisions are reached).  When approved, the producer is
given one final turn to finalise and is expected to end with
`[[TEAM_DONE]]`.  Ideal for any "make a deliverable, then iterate until
acceptable" workflow (papers, design docs, code).

### `sequential_chain`

Members form a **pipeline**: the first member runs with the default
prompt, then each subsequent member receives the previous member's full
reply as its explicit prompt.  At the end of a round the chain wraps
around, so the first member of round N+1 receives the last member of
round N's output.

Use this when the work is a transformation series — for example:

* drafter → editor → translator → formatter
* researcher → summariser → chart-generator

Optional `prompt_template` controls how the handoff is framed; it can
use the `{prev_speaker}` and `{prev_content}` placeholders:

```yaml
workflow:
  type: sequential_chain
  max_rounds: 2
  prompt_template: |
    @{prev_speaker} produced the following output.
    Your task is to refine and improve it:

    {prev_content}
```

### `debate`

Two opposing members argue a proposition for N rounds, then a judge
member delivers a verdict.

```yaml
workflow:
  type: debate
  rounds: 3          # pro/con exchange rounds before the judge speaks (default: 3)
  pro: alice         # member arguing in favour
  con: bob           # member arguing against
  judge: carol       # member delivering the final verdict
```

1. The **pro** member makes an opening statement.
2. The **con** member rebuts.
3. Steps 1–2 repeat for `rounds` rounds.
4. The **judge** receives the full exchange and delivers a verdict.
5. Any member can end early by emitting `[[TEAM_DONE]]`.

### `parallel_review`

Like `review_loop` but all reviewers read the deliverable **at the same time**
(using a thread pool), so the total review wall-time is bounded by the
*slowest* reviewer, not the sum of all reviewers.  A designated **synthesizer**
then consolidates the parallel reviews into one prioritised verdict, and the
**producer** revises.

```yaml
workflow:
  type: parallel_review
  max_rounds: 4            # max revision cycles before stopping
  producer: writer         # who creates and revises the deliverable
  reviewers:               # 2 or more members who review in parallel
    - methods_reviewer
    - stats_reviewer
    - clarity_reviewer
  synthesizer: editor      # consolidates the parallel reviews (may equal producer)
  approve_token: APPROVED  # optional; default is "APPROVED"
```

**Flow per revision cycle:**

1. All reviewers are dispatched simultaneously; each receives the same
   transcript snapshot and produces its review independently.
2. Reviews are appended to the transcript in declaration order.
3. The **synthesizer** reads all reviews and emits a consolidated verdict
   (or `APPROVED` when no further changes are needed).
4. If approved, the producer finalises and emits `[[TEAM_DONE]]`.
5. Otherwise the producer addresses the feedback and the cycle repeats.

> **Thread-safety note:** Reviewer turns are truly parallel LLM calls.
> Each reviewer reads the transcript (read-only during the parallel window)
> and calls its own model.  Reviewers should not use file-writing tools
> during their review turns to avoid concurrent workspace writes.

---

### `parallel`

All members speak **simultaneously** in every round.  Unlike `parallel_review`
(which has a fixed producer → reviewers → synthesizer structure), `parallel`
is fully symmetric: every declared member runs at the same time, every round.

Each member receives the same transcript snapshot at the start of the round —
it cannot see what another member wrote *in the current round*, only in
previous rounds.  After all threads complete, turns are appended in member
declaration order so the transcript is deterministic and `--resume` works.

```yaml
workflow:
  type: parallel
  max_rounds: 4
```

**When to use `parallel`**

- Independent expert panels — each member evaluates the problem from its own
  perspective and writes its findings simultaneously.
- Embarrassingly parallel tasks — member A generates candidate A, member B
  generates candidate B; a later sequential step (or `sequential_chain`) picks
  the best.
- Speed-critical brainstorming where sequential dialogue would be too slow.

**Rendering**

The CLI shows a `⚡ parallel` separator banner before the round starts, then
renders each member's completed panel (with full content, file-write list, and
colour) when the round finishes — no token-by-token streaming during the
parallel window.

> **Thread-safety note:** Members read the transcript concurrently (safe) and
> write to the shared workspace.  Concurrent writes to the *same file path*
> are a race condition.  Design your team so that parallel members produce
> output in disjoint paths (e.g. `member_a/output.txt` vs `member_b/output.txt`).

---


## Workspaces and artifacts

For team `<name>` with `workspace: ./runs/<name>` you get:

```
runs/<name>/
├── transcript.jsonl       # one JSON object per turn
├── shared/                # mounted as /workspace inside every container
│   └── <files written by members>
├── checkpoints/           # automatic point-in-time snapshots (one per live turn)
│   ├── 0001_alice_20240501T120000/
│   ├── 0002_bob_20240501T120145/
│   └── ...
└── members/
    ├── pi/                # mounted as /private inside the pi container
    ├── postdoc/
    └── ...
```

* `shared/` is the canonical place for deliverables and is visible to
  every member at every turn.
* `members/<name>/` is the **private workspace** for that member.  Its
  contents are listed in the member's turn prompt under *"Files in your
  private workspace (/private)"*, so the member can reference its own
  previous work, intermediate files, or notes across turns.  Other members
  cannot see these files.
* `transcript.jsonl` is appended to as the run progresses; one record per
  turn, with `speaker`, `role`, `content`, `files_written`, and
  `timestamp` fields.

`team transcript <file>` renders the transcript human-readably.

---


## Containers, isolation, and root

Each member runs in **its own container** with the following properties:

| property | value | rationale |
| --- | --- | --- |
| Image | `ollama/ollama:latest` (overridable) | Standard Ollama runtime. |
| User inside | **root** | Members have full root *inside their own filesystem*, satisfying "root inside the container" without granting host root. |
| Network | per-team Docker bridge `team-<name>-net`, isolated from other teams and from your host services | Members can only reach each other through the orchestrator, not directly. |
| Port exposure | `127.0.0.1:<random>:11434` | Each member's Ollama API is reachable only from the host loopback by the orchestrator. |
| Model cache | per-member named volume `team-<name>-<member>-models` | Members do *not* share model storage. |
| Mounts | shared workspace at `/workspace`, private workspace at `/private` | Conventional file-exchange surface. |
| Restart policy | `unless-stopped` | Survives daemon restarts during long runs. |
| Resource caps | `memory_limit`, `cpu_limit` honoured if set | Keep large models from starving the host. |

Containers are **not** run with `--privileged` and do not get any host
device access by default; root is confined to the container's mount and
PID namespaces.  You can pass GPUs explicitly via `gpus` (see below).

---


## GPU support

Set `gpus` either globally (under `defaults`) or per-member:

```yaml
defaults:
  gpus: all                # all visible GPUs

members:
  - name: pi
    gpus: [0]              # only GPU 0
  - name: postdoc
    gpus: none             # CPU only
```

Requires the NVIDIA Container Toolkit on the host.  Passed through to
Docker via device requests; non-NVIDIA setups can leave `gpus: none`.

### Apple Silicon / no-Docker Ollama

Docker Desktop on **macOS** runs a Linux VM that cannot access the host's
GPU (neither NVIDIA nor Apple Metal).  Using `gpus: all` there produces:

```
could not select device driver "nvidia" with capabilities [[gpu]]
```

There are two escape hatches:

#### Option A — CPU-only containers (`--no-gpu`)

Pass `--no-gpu` to `team up` or `team run`.  All containers are started
without GPU device requests and fall back to CPU inference inside Docker.
No YAML change required, but inference will be slow on large models.

```bash
team run myteam.yaml --no-gpu
team up  myteam.yaml --no-gpu
```

#### Option B — Native host Ollama with Metal (recommended for Apple Silicon)

Install [Ollama for macOS](https://ollama.com) natively.  The native app
uses **Apple Metal** for GPU acceleration and is dramatically faster than
CPU-only Docker containers.  Then tell `team` to bypass Docker entirely and
connect all members to it:

**Via CLI flag** (no YAML change):

```bash
# Default URL is http://localhost:11434
team run myteam.yaml --host-ollama http://localhost:11434
team up  myteam.yaml --host-ollama http://localhost:11434
```

**Via YAML** (permanent):

```yaml
defaults:
  ollama_url: http://localhost:11434   # all members skip Docker
```

When `defaults.ollama_url` is set (or `--host-ollama` is passed), no Ollama
containers are started; the orchestrator connects directly to the given URL.
Per-member `ollama_url` overrides the default for individual members.

> **`team check` will report a `FAIL`** on macOS when GPU is requested
> without an `ollama_url` configured, and will guide you to one of the two
> options above.

---


## OpenAI-compatible backends

By default every member runs Ollama in a Docker container.  You can instead
point any member at any **OpenAI-compatible API** — LM Studio, vLLM, llama.cpp
server, the real OpenAI API, Anthropic (via a LiteLLM proxy), etc. — without
Docker.

```yaml
defaults:
  backend: openai_compat
  api_base: http://localhost:1234/v1   # LM Studio
  api_key: env:OPENAI_API_KEY          # or a literal key

members:
  - name: lead
    role: Tech Lead
    model: gpt-4o                      # model name sent to the API
    persona: ...
  - name: worker
    role: Engineer
    model: llama-3.1-8b-instruct
    backend: ollama                    # this member still uses Docker
    persona: ...
```

The `backend` and `api_base` fields can be set globally in `defaults` or
overridden per-member.

| field | meaning |
| --- | --- |
| `backend` | `"ollama"` (default) or `"openai_compat"` |
| `api_base` | Base URL of the OpenAI-compat API (e.g. `https://api.openai.com/v1`) |
| `api_key` | API key; use `"env:VAR"` to read from environment at runtime |

When `backend: openai_compat` is set, no Docker container is started for
that member — the orchestrator calls the remote API directly.  The `model`
field is passed as-is to the API.

---


## Remote / no-Docker Ollama

If you already have an Ollama server running (locally or on a remote
machine), you can skip Docker for individual members by setting `ollama_url`:

```yaml
members:
  - name: researcher
    role: Researcher
    model: llama3.1:70b
    ollama_url: http://192.168.1.10:11434  # existing Ollama instance
    persona: ...
```

To route **all** members to the same Ollama instance, set it in `defaults`
or pass `--host-ollama` on the command line (see
[Apple Silicon / no-Docker](#apple-silicon--no-docker-ollama)):

```yaml
defaults:
  ollama_url: http://localhost:11434
```

No container is started for any member that has an effective `ollama_url`
(per-member or from `defaults`); the orchestrator connects directly to the
given URL.  The model must already be pulled on that server (or Ollama's
automatic pull will fetch it on first use).

---


## Custom Ollama image

`docker/Dockerfile.ollama` is an optional, slightly-augmented image that
adds `python3`, `git`, `jq`, `curl`, and friends on top of
`ollama/ollama:latest` for members that want richer in-container
tooling.  Build it once and reference it from any team:

```bash
docker build -f docker/Dockerfile.ollama -t team/ollama:latest docker/
```

```yaml
defaults:
  ollama_image: team/ollama:latest
```

The default `ollama/ollama:latest` is fine for most uses.

---


## Context window management

By default the orchestrator passes the full transcript to every member
every turn.  For long-running teams this can exceed a model's context
window, causing silent truncation or errors.  Configure a strategy to
keep the context manageable:

```yaml
defaults:
  context_strategy: sliding_window   # none | sliding_window | truncate | summarize
  context_budget: 20                 # max turns (sliding_window) or ~token budget (truncate/summarize)
```

| strategy | behaviour |
| --- | --- |
| `none` (default) | Full transcript always sent. |
| `sliding_window` | Only the last `context_budget` turns are sent. |
| `truncate` | Oldest turns are dropped until the estimated token count fits within `context_budget`. A note is prepended explaining that earlier turns were omitted. |
| `summarize` | The oldest turns are compressed into a concise bullet-point digest by calling the member's own LLM (at temperature 0.2). The digest is prepended under a *"Summary of N earlier turn(s)"* heading; the most-recent turns are kept verbatim. 80 % of `context_budget` is reserved for recent turns, 20 % for the digest. Falls back to a plain omission notice if the summarization call fails. |

Override per member:

```yaml
members:
  - name: reviewer
    context_strategy: sliding_window
    context_budget: 10    # this member sees only the last 10 turns
```

---


## Model retention (`keep_alive`)

By default, `team` sets Ollama's `keep_alive` to `"-1"` on every chat request, which tells Ollama to keep the model loaded in RAM indefinitely.  Without this, Ollama's built-in default evicts a model after 5 minutes of inactivity — a problem for large models (tens of gigabytes) that must repeatedly load and unload between turns.

```yaml
defaults:
  keep_alive: "-1"   # keep every model loaded for the duration of the run (default)

members:
  - name: summarizer
    model: llama3.2:3b
    keep_alive: "5m"   # lightweight model — OK to evict after 5 minutes of idle
    ...
```

| Value | Behaviour |
| --- | --- |
| `"-1"` | Keep the model loaded until Ollama stops or another model claim evicts it. **Recommended for team runs.** |
| `"5m"`, `"1h"`, … | Evict after the given idle period (Ollama duration string). |
| `"0"` | Unload immediately after each request (maximises GPU headroom at the cost of reload latency). |

`keep_alive` is an Ollama-only parameter.  When the `openai_compat` backend is used it is silently ignored.

---


## CLI reference

```text
team init        [PATH]               Write a starter team YAML.
team new         [PATH]               Interactive wizard to create a new team YAML.
team validate    <team.yaml>          Parse and validate the YAML.
team check       <team.yaml>          Run preflight checks (no Docker started).
team up          <team.yaml>          Start containers, pull models.
                 [--no-gpu] [--host-ollama URL]
team status      <team.yaml>          Show container status per member.
team logs        <team.yaml>          Tail per-member Ollama logs.
                 [--member NAME] [--tail N]
team run         <team.yaml>          Up + run workflow + (down).
                 [--no-up] [--keep-up] [--resume] [--no-stream] [--interactive]
                 [--no-gpu] [--host-ollama URL]
team transcript  <team.yaml>          Render the persisted transcript.
team export      <team.yaml>          Export transcript + artifacts to a report.
                 [--format markdown|html|json] [--output PATH] [--no-artifacts]
team checkpoints <team.yaml>          List all workspace checkpoints.
team restore     <team.yaml> <ID>     Restore the shared workspace to a checkpoint.
team down        <team.yaml>          Stop & remove containers (and volumes).
                 [--purge]
```

Common flags:

* `-v / --verbose` — debug-level logging.
* `--prepare-timeout SECONDS` (on `up`/`run`) — how long to wait for each
  member's Ollama daemon to become ready and its model to finish pulling
  (default 600).

---


## Interactive wizard

`team new` launches a guided wizard that asks you a series of questions
and writes a validated YAML:

```bash
team new my-team.yaml
```

The wizard prompts for:

* Team name and goal
* Number of members, and for each: name, role, model, persona
* Workflow type and max rounds
* Workspace path

The output is a fully-formed, validated YAML ready to use with `team run`.

---


## Pre-flight checks

Before starting containers, verify that the environment is ready with
`team check`:

```bash
team check my-team.yaml
```

The command checks:

| Check | What it tests |
|---|---|
| Workspace writable | Can create the workspace directory and write files to it |
| Disk space | Reports available GB; warns if below **5 GB** |
| Docker daemon | Docker daemon reachable, version ≥ 20.10, Ollama image present |
| GPU availability | Runs `nvidia-smi` when any member requests GPUs; warns if not found |

Exit code is `0` when all checks pass (warnings allowed), `1` when any
check fails.  Failures are shown with a red ✗ and warnings with a yellow ⚠.

---


## Streaming output

By default `team run` streams each member's reply **token-by-token** to the
terminal as it is generated.  You see a header like `@alice (Lead)` followed
by the reply appearing live — no waiting for the full response.

To disable streaming (e.g. for CI or when redirecting output to a file):

```bash
team run my-team.yaml --no-stream
```

With `--no-stream` the full reply is printed at once after each turn
completes.

---


## Per-turn timeout

Set a hard wall-clock deadline (seconds) on how long any single member turn
may take.  If the LLM doesn't finish within the limit, a `TurnTimeoutError`
is raised and the workflow stops.

```yaml
defaults:
  turn_timeout: 120     # 2 minutes for every member by default

members:
  - name: fast_reviewer
    role: Reviewer
    model: qwen2.5:3b
    persona: You review code quickly.
    turn_timeout: 30    # override — this member gets only 30 s
```

Set `turn_timeout: 0` (or leave it absent) to disable timeouts entirely.

**Implementation details**

The member's `take_turn()` is executed in a `ThreadPoolExecutor` thread and
`future.result(timeout=…)` enforces the deadline.  If the timeout fires the
thread is abandoned (it will eventually finish and be garbage-collected), but
the calling workflow raises `TurnTimeoutError` immediately.

---


## LLM retry with backoff

`team` automatically retries LLM calls that fail due to transient infrastructure errors — connection refused, timeouts, and HTTP 5xx responses from the server — using **exponential backoff**.

```yaml
defaults:
  max_retries: 3       # attempts per call (default: 3; 0 = no retries)
  retry_backoff: 2.0   # backoff base in seconds (wait = backoff ** attempt)

members:
  - name: alice
    max_retries: 5     # per-member override
    retry_backoff: 1.5
```

### How it works

| Scenario | Behaviour |
| --- | --- |
| Connection refused / timeout | Retried up to `max_retries` times. |
| HTTP 5xx (server error) | Retried — the server never processed the request. |
| HTTP 4xx (client error) | **Not retried** — a bad model name or malformed request won't self-heal. |
| Partial streaming response | **Not retried** — the caller already received tokens; replaying would produce duplicates. |

The wait between attempts is `retry_backoff ** attempt` seconds (attempt 0 → 1 s, attempt 1 → 2 s, attempt 2 → 4 s for the default `retry_backoff=2.0`).

### When all retries are exhausted

`LLMRetryExhaustedError` (a subclass of `OllamaError`) is raised.  The CLI catches it and prints a red error panel instead of crashing, preserving any transcript written so far.

---


## Resuming an interrupted run

If a run is interrupted (crash, timeout, Ctrl-C) you can pick up exactly
where it left off without re-running the turns that already completed:

```bash
team run my-team.yaml --resume
```

`--resume` loads the existing `transcript.jsonl`, replays every already-
completed turn instantly (no LLM call), and then continues the workflow
live from the first missing turn.

* Containers are restarted (or re-used) as normal; models are not re-pulled
  if their cache volumes still exist.
* Combine with `--no-up` if your containers are already running from a
  previous `team up`.
* If the transcript doesn't exist or is empty, `--resume` is a no-op and
  the run starts fresh.
* If the previous run completed, resuming is a harmless no-op: the workflow
  will detect `[[TEAM_DONE]]` in the first replayed turn and exit immediately.

---


## Human-in-the-loop intervention

You can inject new directives into a running team at any time without
stopping or restarting.  Two mechanisms are available:

### Interactive mode (foreground runs)

Pass `--interactive` to `team run`.  After every workflow round completes
you are prompted for an optional directive.  Press **Enter** with no text to
let the run continue, or type instructions and press **Enter** to have them
injected before the next round:

```bash
team run my-team.yaml --interactive
```

```text
── round 1/4 complete ──
Enter a directive for the team (or press Enter to continue): Focus only on the auth module for now.
↳ directive injected
```

### File-based injection (background / CI runs)

At any point during a run you can write a plain-text file called
`inject.txt` into the workspace directory:

```bash
echo "Switch to Python 3.12 syntax only." > ./runs/my-team/inject.txt
```

Before the **next member turn** begins, the orchestrator checks for this
file.  If it exists, the content is read, the file is deleted, and the
directive is appended to the transcript as a `@human (director)` turn.
All members see it in their next turn's conversation context.

The file is consumed once and automatically removed.  Drop a new file to
inject again at any later point.

### What the team sees

Both mechanisms produce the same type of transcript entry:

```text
--- Turn N | @human | director ---
<your directive here>
```

The entry is visible to every member in their next turn prompt, just like
any other speaker's turn.

---


## Agent mode and tool use

Members can act as **agents**: they may call external tools, then receive
the tool's output and continue reasoning — all within the same logical turn.
Two invocation modes are supported:

| Mode | How it works |
| --- | --- |
| `text` (default) | Member emits fenced `tool:` blocks in its reply; orchestrator parses and executes them. Works with any model. |
| `native` | Uses the LLM's **function-calling API** (Ollama `tools` parameter / OpenAI function calling). Requires a compatible model (Llama 3.1+, Qwen 2.5, GPT-4 family, etc.). |

### Enabling tools

Tools are named `server/tool` (or `server/*` to enable a whole server):

```yaml
defaults:
  tools: [web/web_search, code/run_python]  # enable globally
  max_tool_rounds: 10                       # max tool-call rounds per turn (default: 10)
  tool_timeout: 300                         # seconds per tool execution (default: 300)
  tool_mode: text                           # "text" (default) or "native"

members:
  - name: researcher
    tools: [web/*, workspace/read_file]     # per-member override
    tool_mode: native                       # this member uses function-calling API
  - name: data_scientist
    tools: [code/*, workspace/*]
```

The 25 built-in tools live on 8 reserved servers: `code` (run_python,
run_bash), `web` (web_search, read_url), `workspace` (read_file, write_file,
append_file, list_files), `memory`, `beliefs`, `decisions`, `federation`, and
`expert`.

### Tool invocation syntax — `text` mode

A member invokes a tool by emitting a fenced block whose info-string is the
tool's wire name (`server__tool`) and whose body is a JSON object of arguments:

````
```tool:web__web_search
{"query": "IPCC AR6 key findings 2024"}
```
````

For a tool with a single required string argument (e.g. `code__run_python`), the
raw body is also accepted verbatim — handy for small models:

````
```tool:code__run_python
print("hello")
```
````

### Tool invocation — `native` mode

In native mode the model receives **JSON Schema** definitions for all
enabled tools and returns structured `tool_calls` objects (OpenAI/Ollama
function-calling format) instead of text fenced blocks.  The orchestrator
executes the tools and passes results back via `tool` role messages — no
text parsing required.

Every tool — built-in or from an external MCP server — advertises a JSON Schema
that is passed to the model automatically.

> **Model requirements**: native mode requires a model that supports
> function calling.  For Ollama, use `llama3.1:8b` or newer, `qwen2.5:7b`,
> `mistral-nemo`, etc.  For OpenAI-compat backends, any GPT-4 / Claude
> model works.  If you pass native mode to a model that ignores the `tools`
> parameter, it will fall back to producing a text reply (no tool calls).

````
```tool:code__run_python
import pandas as pd
df = pd.read_csv('data.csv')
print(df.describe())
```
````

````
```tool:read_file
path: analysis/results.json
```
````

````
```tool:write_file
path: output/summary.md
---
# Summary

This file was written by the agent.
```
````

````
```tool:append_file
path: logs/run.log
---
[step 3] analysis complete.
```
````

````
```tool:list_files
pattern: *.py
```
````

After each tool block the orchestrator executes the tool, injects the result
back into the conversation, and asks the member to continue.  Once the member
produces a reply with no tool blocks, that reply is recorded in the
transcript as usual.

### Available built-in tools

Tools are enabled by qualified `server/tool` name (or `server/*`).

| server | tool | description |
| --- | --- | --- |
| `code` | `run_python` | Execute Python code; cwd is the shared workspace directory. |
| `code` | `run_bash` | Execute a bash command; cwd is the shared workspace directory. |
| `web` | `web_search` | Search the web via the DuckDuckGo instant-answer API (no key required). |
| `web` | `read_url` | Fetch and return the plain-text content of a URL. |
| `workspace` | `read_file` | Read a file from the shared workspace by relative path. |
| `workspace` | `write_file` | Write (create or overwrite) a file in the shared workspace. |
| `workspace` | `append_file` | Append text to a file in the shared workspace. |
| `workspace` | `list_files` | List files in the shared workspace with an optional glob filter. |
| `memory` | `remember` | Store a memory in the member's **persistent cross-session** memory store. |
| `memory` | `recall` | Search the member's persistent memory by keyword. |
| `memory` | `forget` | Delete a memory by key from the persistent store. |
| `memory` | `list_memories` | List stored memories (optionally filtered by tag). |
| `beliefs` | `assert_belief` | Add a claim to the team's **shared belief board** with confidence score. |
| `beliefs` | `contest_belief` | Contest an existing belief (moves it to contested status). |
| `beliefs` | `accept_belief` | Cast an accept vote for an existing belief. |
| `beliefs` | `list_beliefs` | List the shared belief board (optionally filtered by status). |
| `federation` | `delegate_task` | Delegate a sub-task to a remote bridge server and wait for results. Use `peer` for named peers or `url` for direct addressing. |
| `federation` | `list_peers` | List all configured peer teams and their live health status (pending/running counts). |
| `federation` | `broadcast_task` | Fan out the same goal to multiple peer teams concurrently and collect all results. |
| `federation` | `cancel_remote_task` | Cancel a queued or running task on a remote bridge server by task ID. |
| `federation` | `query_registry` | Query a team registry to discover teams matching capability tags or a keyword. |
| `federation` | `sync_beliefs` | Synchronize the team belief board with a remote team cluster (pull, push, or both). |
| `decisions` | `log_decision` | Append a timestamped decision entry to `decisions.md` in the shared workspace. |
| `decisions` | `read_decisions` | Read the full decision log (`decisions.md`) from the shared workspace. |
| `expert` | `delegate_to_expert` | Send a prompt to an external cloud LLM (OpenAI, Anthropic, Google) for expert assistance. |

**Tool arguments**

In `text` mode a tool block's body is a JSON object of arguments. For example
`workspace/write_file` takes `path` and `content`:

````
```tool:workspace__write_file
{"path": "report/intro.md", "content": "# Introduction\n..."}
```
````

Paths are relative to the shared workspace root; parent directories are created
automatically. `list_files` takes an optional `pattern` (e.g.
`{"pattern": "**/*.py"}`); an empty body lists all files. In `native` mode the
model supplies the same arguments through the function-calling API.

### Security note

`run_python` and `run_bash` execute code on the **host machine** with the
privileges of the `team` process.  Only enable these tools for members whose
prompts you trust.

### Expert delegation — `delegate_to_expert`

When a task is too complex for the local model assigned to a member, that
member can **delegate the sub-problem** to a subscription-based cloud LLM
(ChatGPT, Claude, Gemini) and receive the answer as a tool result.  The
member remains responsible for the turn — it incorporates the expert's reply
into its own response, so the team structure and role assignments are preserved.

The cloud model is **not** a team member.  It has no access to the
transcript, the shared workspace, or any other team state — only the prompt
text you explicitly send.

#### Setup

Export the API key for the provider(s) you want to use **on the host** before
running `team`:

```bash
export OPENAI_API_KEY=sk-…          # for provider: openai
export ANTHROPIC_API_KEY=sk-ant-…   # for provider: anthropic
export GOOGLE_API_KEY=AIza…         # for provider: google
```

Enable the tool for a member in the YAML:

```yaml
members:
  - name: analyst
    model: llama3.2:3b
    tools: [expert/delegate_to_expert, workspace/read_file, workspace/write_file]
```

#### Usage

The tool body is a JSON object with `provider` and `prompt` (plus optional
`model`, `max_tokens`, `temperature`):

````
```tool:expert__delegate_to_expert
{"provider": "openai", "model": "gpt-4o", "max_tokens": 4096,
 "prompt": "You are a statistics expert. Identify any violations of linear regression assumptions in the residuals below and suggest remedies. Residuals: ..."}
```
````

**Single-line prompt**:

````
```tool:delegate_to_expert
provider: anthropic
model: claude-opus-4-5
prompt: What is the time complexity of Dijkstra's algorithm with a binary heap?
```
````

| field | required | default | description |
| --- | --- | --- | --- |
| `provider` | ✓ | — | `openai`, `anthropic`, or `google` |
| `model` | | provider default | Model name accepted by the provider API |
| `prompt` | ✓ | — | Prompt text to send to the expert model |
| `max_tokens` | | `2048` | Maximum tokens in the response |
| `temperature` | | `0.2` | Sampling temperature 0–2 |

**Provider defaults**: `gpt-4o` (OpenAI), `claude-opus-4-5` (Anthropic),
`gemini-1.5-pro` (Google).

> **Privacy**: the prompt text is sent to the external API.  Do not include
> sensitive data unless your data-handling agreement with the provider permits it.
> Only enable `delegate_to_expert` for members that may handle the data appropriately.

### Full system access and package installation

Agents have **full, unrestricted access to the host system** — the same
privileges as the user who runs the `team` process.  This is intentional:
agents should be able to do anything a human researcher or engineer can do.

In particular, agents can install software at will:

````
```tool:code__run_bash
{"command": "pip install scikit-learn seaborn --quiet"}
```
````

````
```tool:code__run_bash
{"command": "apt-get install -y ffmpeg"}
```
````

````
```tool:code__run_python
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "biopython"], check=True)
import Bio
print(Bio.__version__)
```
````

When a tool invocation takes longer than expected (e.g. downloading a large
package), increase the `tool_timeout` in your YAML:

```yaml
defaults:
  tool_timeout: 600   # 10 minutes — safe for most installs
```

The default `tool_timeout` is **300 seconds** (5 minutes), which covers the
vast majority of `pip install` and `apt-get` operations on a normal network
connection.

### How it works

**Text mode** (`tool_mode: text`):
```
member turn:
  1. LLM called with system prompt + conversation context
  2. If reply contains tool: fenced blocks → execute each tool
  3. Tool results injected as a follow-up user message
  4. LLM called again (no streaming; repeats up to max_tool_rounds)
  5. If no tool blocks in reply → reply recorded in transcript
```

**Native mode** (`tool_mode: native`):
```
member turn:
  1. LLM called with JSON Schema tool definitions in the "tools" parameter
  2. If response contains tool_calls → execute each named tool using args_to_body()
  3. Each result injected as a "tool" role message
  4. LLM called again (repeats up to max_tool_rounds)
  5. When LLM returns text (no tool_calls) → reply recorded in transcript
```

Token usage from all tool-call rounds is accumulated and reported in the
[token usage summary](#token-usage-tracking).

### Streaming display

When streaming is enabled (`team run` without `--no-stream`), tool calls
are displayed inline:

```text
@researcher (Research Lead)
I'll search for recent data on this topic.

  🔧 tool: web_search  query: climate change 2024 report
     ↳ **Climate Change** A programming language. - Flooding in coastal…
Based on the search, the key findings are…
```

---

### Connecting MCP servers

The built-in tool set is a starting point. `team` speaks the
[Model Context Protocol](https://modelcontextprotocol.io), so you can connect
**any** MCP server — the thousands of community servers, a vendor's hosted
server, or one you write yourself — and its tools become available to members
exactly like built-ins. Declare servers under `mcp_servers:`; enable their
tools with `server/tool` patterns.

```yaml
mcp_servers:
  github:                                   # external stdio server
    transport: stdio
    command: npx
    args: ["-y", "@modelcontextprotocol/server-github"]
    env: { GITHUB_PERSONAL_ACCESS_TOKEN: "env:GITHUB_TOKEN" }   # env:VAR expansion
  kb:                                       # remote Streamable HTTP server
    transport: http
    url: https://kb.example.com/mcp
    headers: { Authorization: "env:KB_TOKEN" }
  helpers:                                  # a server script in your repo
    transport: stdio
    command: python
    args: ["examples/mcp/team_helpers_server.py"]

members:
  - name: researcher
    tools: [web/*, workspace/read_file, github/search_repositories, helpers/*]
```

Server names must match `^[a-z][a-z0-9-]*$` and may not collide with the 8
reserved built-in servers.

#### Writing your own server

Any MCP server works, so the simplest path is a small
[FastMCP](https://github.com/modelcontextprotocol/python-sdk) stdio script:

```python
# servers/lab_tools.py
import os
from pathlib import Path
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("lab")
WORKSPACE = Path(os.environ["TEAM_WORKSPACE"])   # injected by the orchestrator

@mcp.tool()
def normalize_counts(path: str, method: str = "tpm") -> str:
    """Normalize a counts matrix from the shared workspace."""
    ...
    return "wrote normalized.csv"

if __name__ == "__main__":
    mcp.run()
```

Stdio servers run as a subprocess of `team` and receive the shared workspace
path via the `TEAM_WORKSPACE` environment variable (the run transcript is one
level up at `TEAM_WORKSPACE/../transcript.jsonl`). The function signature and
docstring become the tool's schema automatically, so it is a first-class typed
tool in both `text` and `native` mode. The same file works unmodified in any
other MCP client (Claude Desktop, Cursor, the MCP Inspector).

`examples/mcp/team_helpers_server.py` is a complete example bundling the
team-process helpers (`task_add`/`task_done`/`task_list`, `search_transcript`,
`progress_snapshot`, `request_critique`/`pick_critique`/`list_critiques`).

#### Packaged servers via entry points

A pip-installable extension can register in-process servers under the
`team.mcp_servers` entry-point group; reference them with
`transport: entry_point`:

```yaml
mcp_servers:
  bioblend:
    transport: entry_point
    entry_point: bioblend        # a name registered by an installed package
```

`team forge <name>` scaffolds a `team-*` extension package with a `servers/`
directory pre-wired to this entry-point group.

### Standing context — `extra_context`

For guidelines, checklists, templates, and domain rules that should always be
visible (rather than looked up via a tool), list Markdown files under
`extra_context:`. Their content is injected verbatim into each member's system
prompt on every turn. Paths are relative to the team YAML's directory.

```yaml
defaults:
  extra_context:
    - context/review_checklist.md
    - context/escalation_rules.md

members:
  - name: reviewer
    extra_context: [context/reviewer_playbook.md]   # overrides the default list
```

The `examples/context/` directory contains ready-to-use files
(`review_checklist.md`, `escalation_rules.md`, `decision_record_format.md`).

## Shared institutional context

When a workspace contains a `context.md` file at its root, `team` injects its
content into **every** member's turn context automatically — no per-member
configuration required.

This is the right place for knowledge that applies to all members equally:
lab conventions, dataset descriptions, domain terminology, naming standards,
relevant prior work, or any background a new team member would need to read
on day one.

**Creating the context file:**

```bash
cat > ./runs/my-team/context.md << 'EOF'
# Lab context

This project analyses the TCGA-BRCA cohort (1,142 samples, 38 features).

## Naming conventions
- All feature files use `snake_case` column names.
- Model outputs go in `results/`.

## Domain notes
- Use log2 CPM normalisation for expression data.
- Primary endpoint is 5-year overall survival (OS5).
EOF
```

The file is read from disk **on every turn** so you can update it while a
run is in progress (e.g. to correct a mistake or add a new constraint).
If the file is absent, the section is silently omitted.
The content is truncated at 8 192 characters if the file is very large.

---


## Decision log

Members with the `log_decision` tool enabled can record structured, timestamped
decisions in a shared `decisions.md` file inside the workspace.  Any member
can later call `read_decisions` to review the accumulated rationale before
making related choices.

**Enabling the tools:**

```yaml
defaults:
  tools: [log_decision, read_decisions]   # add to any existing tool list
```

**Logging a decision:**

````
```tool:log_decision
title: Chose pandas over polars for data wrangling
rationale: Polars ecosystem is too immature; pandas is already a project dependency.
alternatives: polars, dask, vaex
```
````

The entry is appended to `decisions.md` in the shared workspace:

```markdown
## Decision: Chose pandas over polars for data wrangling
**Date:** 2024-07-15T10:32:44Z  
**By:** @data_scientist  

**Rationale:** Polars ecosystem is too immature; pandas is already a project dependency.

**Alternatives considered:** polars, dask, vaex

---
```

**Reading the decision log:**

````
```tool:read_decisions
```
````

Returns the full `decisions.md` content so members can consult previous
decisions when facing related choices.

---


## Structured JSON output

By default members reply in free-form text.  When you need machine-readable
output — e.g. an extractor member whose results are consumed by downstream
code — set `output_format: json` on that member.

```yaml
members:
  - name: extractor
    role: Data extractor
    model: llama3.1:8b
    persona: You extract structured data from documents.
    output_format: json
    output_schema:                     # optional — validates the reply
      type: object
      required: [entities, summary]
      properties:
        entities:
          type: array
          items: {type: string}
        summary:
          type: string
```

**What happens**

1. The system prompt gains an `## Output format` section instructing the model
   to reply with valid JSON only.
2. After the LLM replies, `team` calls `json.loads()` on the content.
3. If parsing fails (or schema validation fails when `output_schema` is set),
   the orchestrator sends a correction prompt and retries up to **3 times**.
4. The parsed object is stored in `TurnResult.json_output` and is accessible
   from custom workflows or post-run code.
5. Schema validation requires `pip install jsonschema`; without it the schema
   check is skipped silently.

> **Note:** `output_format` is per-member only — it is not available as a
> team-wide `defaults` key.

---


## Conditional routing

Enable dynamic, branching conversations where each member's output determines who speaks next — building state-machine-like workflows without any code.

```yaml
workflow:
  type: conditional
  start: writer       # optional; defaults to the first listed member
  max_rounds: 20

members:
  - name: writer
    model: llama3
    persona: You are a technical writer.
    role: Writer
    routes:
      - if_contains: "NEEDS_REVISION"
        next: editor
      - if_match: "APPROVED|LGTM"
        next: publisher
      - default: reviewer    # fallback when nothing else matches

  - name: editor
    model: llama3
    persona: You are an editor.
    role: Editor
    routes:
      - if_contains: "DONE"
        next: publisher
      - default: writer      # loop back for another draft

  - name: reviewer
    model: llama3
    persona: You are a reviewer.
    role: Reviewer
    routes:
      - default: writer

  - name: publisher          # terminal node — no routes needed
    model: llama3
    persona: You are a publisher.
    role: Publisher
```

### Route rules

Rules are evaluated **top-to-bottom**; the first match wins.

| Key | Behaviour |
| --- | --- |
| `if_contains: "TEXT"` | Case-insensitive substring search in the member's last reply. |
| `if_match: "REGEX"` | Case-insensitive `re.search` against the member's last reply. |
| `default: member` | Unconditional fallback; fires when no other rule matches. |

A member with **no `routes`** falls back to the standard round-robin next-speaker logic.

### Workflow end conditions

The workflow stops when:
* any member outputs `[[TEAM_DONE]]`, or
* the total turn count reaches `max_rounds`.

---


## Token budget

Prevent runaway costs by capping how many tokens a member may consume across all turns in a single run.

```yaml
defaults:
  token_budget: 5000   # max prompt+completion tokens per member per run

members:
  - name: alice
    token_budget: 10000  # per-member override
```

When a member's cumulative token usage reaches the budget before their next turn, `TokenBudgetError` is raised and the run stops gracefully. The transcript and any workspace files written so far are preserved, and `team run --resume` with a higher budget can continue from where it left off.

> **Note:** Replayed turns (from `--resume`) do **not** count toward the budget.

### Budget resolution

| Setting | Effective budget |
| --- | --- |
| `token_budget` in `defaults` only | Applied to every member. |
| `token_budget` in a specific member | Overrides the `defaults` value for that member only. |
| Neither set | No limit — member runs until the workflow ends. |

---


## Per-agent persistent memory

In a real research lab, scientists remember what worked and what failed —
across months of experiments.  `team` gives each agent a **private,
persistent memory store** backed by SQLite that survives between completely
separate `team run` invocations.

```
Session 1 (January): alice uses remember to store "AlphaFold3 RMSD 1.2 Å"
Session 2 (February): alice uses recall to surface that result and build on it
```

This is what separates `team` from all other orchestration frameworks: your
agents actually **accumulate knowledge over time**.

### Enabling memory

Add a `memory:` section to your team YAML:

```yaml
memory:
  enabled: true
  inject_recent: 5    # memories injected into each turn's context (default: 5)
  store: ~/.team/memory   # optional; defaults to <workspace>/memory/
```

Enable memory tools for each member:

```yaml
members:
  - name: alice
    tools: [code/run_python, memory/*]
```

### Memory tools

All memory tools use a `key:` / header + `---` / value body format:

**`remember`** — store a cross-session memory:

````
```tool:remember
key: protein_folding_baseline_2025
tags: results, methods
importance: 0.9
---
AlphaFold3 outperforms RoseTTAFold on monomers (RMSD 1.2 vs 2.1 Å, n=1 000).
Dataset: PDB validation set, tested January 2025.
```
````

**`recall`** — full-text search across all memories:

````
```tool:recall
query: protein folding
limit: 5
```
````

Returns a ranked list of matching memories (by importance then recency).

**`forget`** — delete a memory by key:

````
```tool:forget
key: protein_folding_baseline_2025
```
````

**`list_memories`** — browse all memories (optionally by tag):

````
```tool:list_memories
tag: results
limit: 20
```
````

At the start of every turn, the *n* most recent memories are automatically
injected into the member's context under `## Your persistent memories`.

### Memory config reference

| key | type | default | description |
| --- | --- | --- | --- |
| `enabled` | bool | `false` | Enable persistent memory for all members. |
| `inject_recent` | int | `5` | Number of recent memories to inject into each turn's context. |
| `store` | path | `<workspace>/memory` | Directory that holds the per-member SQLite databases. |

---


## Shared team belief board

In collaborative science, a team's most important output is not files — it is
**what the team collectively knows**.  The `team` belief board formalises this
as a living, structured record of claims with provenance, confidence scores,
and consensus voting.

```
alice asserts: "RNA Pol II is rate-limiting in elongation" (confidence: 85%)
bob accepts → 2/3 votes ≥ threshold → status: ACCEPTED
carol contests with reason: "only tested in HEK293" → status: CONTESTED
```

After a run: `team beliefs myteam.yaml` shows everything the team concluded.

### Enabling the belief board

```yaml
beliefs:
  enabled: true
  consensus_threshold: 0.5   # fraction of members required for acceptance
  inject_limit: 10            # beliefs shown in each member's turn context
```

Enable belief tools for each member:

```yaml
members:
  - name: alice
    tools: [code/run_python, beliefs/*]
```

### Belief tools

**`assert_belief`** — propose a claim with optional evidence:

````
```tool:assert_belief
confidence: 0.85
evidence: RMSD analysis, PDB validation set, n=1 000, January 2025
---
AlphaFold3 is the best available method for monomer structure prediction.
```
````

The member who asserts a belief automatically casts an *accept* vote.  The
returned belief ID (e.g. `a3f2b1c9`) is used in subsequent votes.

**`accept_belief`** — vote to accept:

````
```tool:accept_belief
id: a3f2b1c9
```
````

**`contest_belief`** — move a belief to `contested` status:

````
```tool:contest_belief
id: a3f2b1c9
reason: Dataset is limited to well-studied proteins; may not generalise.
```
````

**`list_beliefs`** — browse the board:

````
```tool:list_beliefs
status: contested
```
````

Valid status values: `pending`, `accepted`, `contested`, `rejected`.  Omit to
list all beliefs.

Beliefs are injected into every member's turn context under
`## Shared team belief board` so the whole team sees the current state before
each turn.

### Inspecting beliefs with team beliefs

```bash
team beliefs myteam.yaml                    # all beliefs
team beliefs myteam.yaml --status accepted  # accepted only
team beliefs myteam.yaml --status contested # contested — needs attention
```

Output example:

```
                  Belief board — team 'my-team'
┏━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━┳━━━━━┳━━━━━━━━━┓
┃ ID     ┃ Status      ┃ Claim                                                   ┃ Confidence ┃ By    ┃ For ┃ Against ┃
┡━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━╇━━━━━╇━━━━━━━━━┩
│ a3f2b1 │ ✓ accepted  │ AlphaFold3 is best for monomer structure prediction.    │       85%  │ @alice│   2 │       0 │
│ 9c1d33 │ ⚡ contested│ The dataset generalises to all protein families.        │       60%  │ @bob  │   1 │       1 │
└────────┴─────────────┴─────────────────────────────────────────────────────────┴────────────┴───────┴─────┴─────────┘
⚡ Some beliefs are contested — review and resolve via accept_belief / contest_belief tools.
```

### Belief config reference

| key | type | default | description |
| --- | --- | --- | --- |
| `enabled` | bool | `false` | Enable the shared belief board. |
| `consensus_threshold` | float | `0.5` | Fraction of members who must accept a belief for it to become `accepted`. |
| `inject_limit` | int | `10` | Maximum number of beliefs injected into each member's turn context. |

---


## Workspace checkpoints

Every time a live member turn is about to execute, the orchestrator
automatically snapshots the current state of the **shared workspace** before
any files are written.  Snapshots are stored under
`<workspace>/checkpoints/` with names that encode the turn index, the
member about to speak, and the timestamp:

```
checkpoints/
├── 0001_alice_20240501T120000/   # state before alice's 1st turn
├── 0003_bob_20240501T120145/     # state before bob's 2nd turn
└── ...
```

If the shared workspace is empty (no files have been produced yet), the
snapshot is silently skipped — there is nothing to back up.

### Listing checkpoints

```bash
team checkpoints my-team.yaml
```

```
┌──────────────────────────────┬──────┬──────────────────────┬─────────────────────┬───────┐
│ ID                           │ Turn │ Before member's turn │ Timestamp           │ Files │
├──────────────────────────────┼──────┼──────────────────────┼─────────────────────┼───────┤
│ 0001_alice_20240501T120000   │    1 │ @alice               │ 2024-05-01 12:00:00 │     3 │
│ 0003_bob_20240501T120145     │    3 │ @bob                 │ 2024-05-01 12:01:45 │     5 │
└──────────────────────────────┴──────┴──────────────────────┴─────────────────────┴───────┘
```

### Restoring a checkpoint

Copy the checkpoint ID from the table and pass it to `team restore`:

```bash
team restore my-team.yaml 0001_alice_20240501T120000
```

```
restored checkpoint 0001_alice_20240501T120000 — 3 file(s) now in the shared workspace.
```

The current contents of `shared/` are replaced with the snapshot.
**This cannot be undone** unless a later checkpoint already captured the
state you are overwriting, so check `team checkpoints` before restoring.

### Use cases

* **Undo a bad turn** — a member produced unwanted file changes; restore the
  checkpoint taken just before that turn.
* **Branch from a known-good state** — restore an earlier checkpoint, edit
  `team.yaml` (e.g. change the goal or persona), and re-run from there.
* **Audit the evolution of the workspace** — inspect any checkpoint
  directory directly; it is a plain copy of `shared/` at that point in time.

---


## Workspace time-travel (`team rollback`)

Every live member turn is preceded by an automatic workspace snapshot (see
[Workspace checkpoints](#workspace-checkpoints)).  When things go wrong you
can roll back the shared workspace to *any prior point in time* and resume
from there — effectively forking the timeline:

```bash
# 1. List all available snapshots
team rollback myteam.yaml

# 2. Restore to a specific checkpoint (with confirmation prompt)
team rollback myteam.yaml --to 0005_alice_20250510T183000

# 3. Skip the confirmation prompt (useful in scripts)
team rollback myteam.yaml --to 0005_alice_20250510T183000 --yes
```

After rolling back, resume the run from the restored state:

```bash
team run myteam.yaml --resume
```

Because the transcript also persists, `--resume` skips all turns already
recorded in it.  To *re-run* from turn 5 with a different approach, truncate
the transcript manually (or delete it and rely entirely on the restored
workspace files).

> `team rollback` is a thin wrapper around the existing
> `CheckpointManager.restore()` logic.  The underlying `team restore`
> command (which requires an exact checkpoint ID argument) remains available
> for scripting.

---


## Token usage tracking

After every `team run` a token usage summary is printed:

```text
┌────────────────────────────────────────────────────┐
│              Token usage (live turns)              │
├──────────┬─────────┬───────────┬───────────────────┤
│ member   │  prompt │ completion│  total            │
├──────────┼─────────┼───────────┼───────────────────┤
│ @lead    │  12 450 │     3 210 │  15 660           │
│ @worker  │   8 120 │     5 890 │  14 010           │
├──────────┼─────────┼───────────┼───────────────────┤
│ total    │  20 570 │     9 100 │  29 670           │
└──────────┴─────────┴───────────┴───────────────────┘
```

Token counts come from the Ollama `/api/chat` `eval_count` /
`prompt_eval_count` fields (for the `ollama` backend) or the OpenAI
`usage` object (for `openai_compat`).  The summary is omitted when all
counts are zero (e.g. pure replay runs or backends that don't report
token usage).

---


## Cost estimation

After every `team run` and `team stats` command, the token-usage table includes an **Est. cost** column with a USD estimate based on the model used by each member.

Local Ollama models always show **$0.00 (local)** since they run on your hardware.  Cloud models (`backend: openai_compat`) are looked up in the built-in pricing table.

### Built-in pricing table

| Provider | Models |
| --- | --- |
| **OpenAI** | `gpt-4.1`, `gpt-4.1-mini`, `gpt-4.1-nano`, `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`, `gpt-4`, `gpt-3.5-turbo`, `o1`, `o1-mini`, `o3`, `o3-mini` |
| **Anthropic** | `claude-opus-4`, `claude-sonnet-4`, `claude-3-5-sonnet`, `claude-3-5-haiku`, `claude-3-opus`, `claude-3-sonnet`, `claude-3-haiku` |
| **Google** | `gemini-2.0-flash`, `gemini-1.5-pro`, `gemini-1.5-flash` |
| **Mistral** | `mistral-large`, `mistral-medium`, `mistral-small`, `codestral` |
| **Meta (cloud-hosted)** | `llama-3.1-405b`, `llama-3.1-70b`, `llama-3.1-8b`, `llama-3-70b`, `llama-3-8b` |

Model names are matched by prefix/substring so versioned names like `gpt-4o-2024-08-06` automatically map to `gpt-4o` pricing.  If a model is not recognised, the cost column shows **?**.

> **Prices are estimates only.** Provider pricing changes over time — update `team/pricing.py` with the latest figures from your provider's pricing page.

---


## Run statistics

`team stats` shows a detailed breakdown of a completed run — turn counts,
token usage per speaker, total duration, and files written — without
needing to start any containers:

```bash
team stats my-team.yaml
```

Example output:

```text
Team: my-team  18 turns · 29 670 tokens · duration 142.3s · 5 file(s) written

┌─────────────────────────────────────────────────────────────────────┐
│               Turns & token usage by speaker                        │
├──────────────┬───────┬───────────────┬──────────────────┬───────────┤
│ Speaker      │ Turns │ Prompt tokens │ Completion tokens│    Total  │
├──────────────┼───────┼───────────────┼──────────────────┼───────────┤
│ @lead        │     5 │        12 450 │            3 210 │    15 660 │
│ @orchestrator│     1 │             0 │                0 │         0 │
│ @worker      │    12 │         8 120 │            5 890 │    14 010 │
├──────────────┼───────┼───────────────┼──────────────────┼───────────┤
│ total        │    18 │        20 570 │            9 100 │    29 670 │
└──────────────┴───────┴───────────────┴──────────────────┴───────────┘
```

The `Transcript.stats()` method in `team/bus.py` is also part of the
public Python API:

```python
from team.bus import Transcript
from team.config import load_team

cfg = load_team("my-team.yaml")
t = Transcript(persist_path=cfg.workspace / "transcript.jsonl", resume=True)
s = t.stats()
print(s["total_turns"], s["duration_seconds"])
```

---


## Exporting a run report

After a run you can bundle the full transcript and every produced artifact
into a single shareable document:

```bash
team export my-team.yaml                          # Markdown (default)
team export my-team.yaml --format html            # self-contained HTML (dark-mode aware)
team export my-team.yaml --format json            # machine-readable JSON
team export my-team.yaml --output ~/Desktop/run.md
team export my-team.yaml --no-artifacts           # omit workspace files (faster, smaller)
```

The report includes:
* Team name, goal, members, and workflow settings.
* Every member turn with speaker, role, content, and files written.
* **Token usage & estimated cost table** — per member and totals.
* Full contents of all files produced in the shared workspace (omit with `--no-artifacts`).

Output path defaults to `<workspace>/report.md` / `.html` / `.json`.

**Format details:**

| Format | Description |
| --- | --- |
| `markdown` | Single `.md` file with transcript, token table, and fenced artifact blocks. |
| `html` | Self-contained `.html` — embedded CSS, no external deps, respects `prefers-color-scheme: dark`. |
| `json` | Structured JSON (`format_version: 1`) with `team`, `stats`, `token_usage`, `turns`, and `artifacts` keys — suitable for post-processing. |

---


## `team replay` — interactive transcript browser

After a run completes, `team replay` lets you step through the saved
transcript turn-by-turn in an interactive terminal viewer — like a
debugger for a past run.  No LLM calls, no Docker, no network — it
works entirely from the persisted `transcript.jsonl` file.

```
team replay myteam.yaml                     # start at turn 0
team replay myteam.yaml --from 5            # start at turn 5
team replay myteam.yaml --speaker alice     # jump to alice's first turn
```

### Navigation keybindings

| Key | Action |
| --- | --- |
| `→` / `n` / Space / Enter | Advance to the next turn |
| `←` / `p` / `b` | Go back one turn |
| `g` | Prompt for a turn number and jump directly to it |
| `f` | Prompt for a speaker name and jump to their next turn |
| `s` | Toggle the stats summary panel (token totals, turn counts) |
| `q` / Esc | Quit |

### Non-interactive mode

When stdin is not a TTY (e.g. a CI pipeline or a pipe), `team replay`
prints all turns sequentially — the same rich panel rendering used by
`team transcript` — and exits immediately.  This makes it safe to use
in scripts:

```bash
team replay myteam.yaml | head -100
```

### Options

| Option | Default | Description |
| --- | --- | --- |
| `--from N` | `0` | Start at turn N (0-based). |
| `--speaker NAME` | — | Jump to the first turn by NAME at startup. |

---


## Automated testing with `team test`

`team test` runs the team and then validates a set of assertions defined in the
`tests:` section of the team YAML.  This makes it easy to build a repeatable
test suite for your team in CI.

```yaml
tests:
  - name: creates hello.py
    type: file_exists
    path: hello.py

  - name: script contains print
    type: file_contains
    path: hello.py
    text: "print"

  - name: no error messages
    type: file_not_contains
    path: report.txt
    text: "ERROR"

  - name: results is valid JSON
    type: json_valid
    path: results.json

  - name: results matches schema
    type: json_schema
    path: results.json
    schema:
      type: object
      required: [entities, summary]

  - name: any member mentioned Python
    type: transcript_contains
    text: "Python"

  - name: developer specifically mentioned Python
    type: transcript_contains
    speaker: developer
    text: "Python"

  - name: exactly 4 member turns
    type: transcript_count
    count: 4
```

```
team test myteam.yaml               # run the team, then assert
team test myteam.yaml --no-run      # assert against an existing run
team test myteam.yaml --max-rounds 2 --goal "quick smoke test"
```

Exits with code **0** if all assertions pass, **1** if any fail (suitable for
CI gates).

### Assertion reference

| Type | Required fields | Description |
| --- | --- | --- |
| `file_exists` | `path` | File must exist in the shared workspace. |
| `file_not_exists` | `path` | File must *not* exist. |
| `file_contains` | `path`, `text` | File content must contain the substring. |
| `file_not_contains` | `path`, `text` | File content must *not* contain the substring. |
| `json_valid` | `path` | File must be parseable JSON. |
| `json_schema` | `path`, `schema` | File must be valid JSON matching the JSON Schema. |
| `transcript_contains` | `text` | At least one turn must contain the text. Add `speaker` to restrict to one member. |
| `transcript_count` | `count` | Exact number of member turns (excludes `orchestrator`/`human`). |

All `path` values are relative to the **shared workspace** directory
(`<workspace>/shared/`).

---


## Multi-team pipelines

A *pipeline* lets you chain multiple team runs together so that the output of one team — its shared workspace files and a transcript summary — is automatically injected into the next team's context.

### Pipeline YAML

Create a `pipeline.yaml` alongside your team files:

```yaml
name: research-and-write
description: Research a topic, then write a publication-ready paper.
workspace: ./runs/research-and-write   # optional; default is ./runs/<name>

stages:
  - id: research
    team: ./teams/researcher.yaml

  - id: writing
    team: ./teams/writer.yaml
    depends_on: [research]          # wait for research to complete
    inject_files: true              # copy research's shared/ files here
    inject_context: true            # write context.md from research output
    goal_override: |                # {stage_id.summary} templates available
      Write a publication-ready paper based on the research below.

      {research.summary}
```

### Running a pipeline

```bash
team pipeline pipeline.yaml
```

Preview the execution plan without running anything:

```bash
team pipeline pipeline.yaml --dry-run
```

### Stage fields

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `id` | string | *(required)* | Unique stage identifier used in `depends_on` and goal templates. |
| `team` | path | *(required)* | Path to the team YAML file (relative to the pipeline file). |
| `depends_on` | list of IDs | `[]` | Stages that must complete before this stage runs. |
| `inject_files` | bool | `false` | Copy every file from upstream stages' `shared/` directories into this stage's `shared/` directory before the team starts. |
| `inject_context` | bool | `false` | Write a `context.md` file into this stage's workspace summarising upstream stages' output. Members pick it up automatically. |
| `goal_override` | string | — | Replace the team YAML's `goal` for this pipeline run. Supports `{stage_id.summary}` template substitution. |

### How data flows

Each stage runs inside its own sub-workspace: `<pipeline.workspace>/<stage.id>/`. At the end of every stage the runner extracts:

- **Summary** — the last five member turns from the transcript, concatenated.
- **Artifacts** — all files in `shared/`, keyed by relative path.

When the next stage has `inject_files: true`, artifact files are copied verbatim into the destination stage's `shared/` directory before its team starts. When `inject_context: true`, a `context.md` is written at the stage workspace root with the summaries and file lists from all upstream stages.

### Goal templates

`goal_override` is a Python `str.format()` template. Each upstream stage result is available as `{stage_id.summary}`:

```yaml
goal_override: |
  Review the following research and identify gaps.

  Research output:
  {research.summary}

  Initial draft:
  {writing.summary}
```

---


## Cross-team collaboration (bridge)

`team` clusters running on **different machines**, operated by **different
people or organisations**, can collaborate on common goals through the bridge
protocol.  One cluster delegates a sub-task to a remote cluster; the remote
cluster runs its full team workflow and returns the results — including all
files it produced.  The exchange can repeat over multiple turns, just like a
real inter-laboratory collaboration.

### How it works

```
Lab A cluster (local)                       Lab B cluster (remote)
┌─────────────────────────────────────┐     ┌──────────────────────────────────┐
│  Orchestrator A                     │     │  team serve lab-b.yaml           │
│  members: pi, analyst               │     │  BridgeServer (port 7001)        │
│                                     │     │                                  │
│  @pi uses delegate_task tool ───────┼─────┼──► POST /tasks                   │
│                                     │     │    ┌──────────────────────────┐  │
│                                     │     │    │ Orchestrator B           │  │
│                                     │     │    │ members: coder, reviewer │  │
│                                     │     │    │ runs full workflow       │  │
│                                     │     │    └──────────────────────────┘  │
│  result written to workspace ◄──────┼─────┼─── GET /tasks/{id}  (complete)   │
│  injected into transcript           │     │    files + summary returned      │
└─────────────────────────────────────┘     └──────────────────────────────────┘
```

1. **Lab B** exposes its cluster by running `team serve`.
2. **Lab A's** agents use the `delegate_task` built-in tool, specifying Lab
   B's URL (or its peer name — see below), a goal, optional context, and
   optional workspace files to send.
3. The bridge server receives the task, writes the sent files into a fresh
   sub-workspace, and runs Lab B's full team workflow with the delegated goal.
4. When Lab B's workflow finishes, the server returns a summary and all
   produced files.
5. The `delegate_task` tool writes the received files into Lab A's shared
   workspace and returns the summary to the agent — all within a single tool
   call round.
6. Lab A's agents incorporate the results and can delegate again if needed.

### Exposing a team as a bridge server

```bash
# On Lab B's machine — makes the team reachable from the network
team serve lab-b.yaml --port 7001
```

Output:

```text
bridge server started — team lab-b listening on port 7001
max concurrent tasks: 1 · workspace: ./runs/lab-b/bridge_workspaces
Press Ctrl-C to stop.
```

Each incoming task is run in an isolated sub-workspace under
`<workspace>/bridge_workspaces/<task-id>/` so concurrent tasks never
interfere.  Press **Ctrl-C** to gracefully shut down.

### Delegating work from another team

Lab A's agents use the **`delegate_task`** built-in tool.  Enable it in the
YAML like any other tool:

```yaml
defaults:
  tools: [delegate_task, list_peers, broadcast_task, cancel_remote_task, read_file, write_file]
```

Tool invocation syntax inside a member's reply:

````
```tool:delegate_task
url: http://lab-b.example.com:7001
goal: Perform survival analysis on the BRCA cohort.
context: |
  We completed pre-processing.  The cleaned dataset is in
  data/preprocessed.csv (1 142 samples, 38 features, event column: "os_event").
files: data/preprocessed.csv, data/metadata.json
timeout: 600
```
````

Instead of a raw URL, you can use a **named peer** (see [Named peer registry](#named-peer-registry)):

````
```tool:delegate_task
peer: lab-b
goal: Perform survival analysis on the BRCA cohort.
```
````

| field | required | description |
| --- | --- | --- |
| `url` | ✓ (or `peer`) | Base URL of the remote `team serve` endpoint. |
| `peer` | ✓ (or `url`) | Named peer from `bridge.peers` (resolved to URL at call time). |
| `goal` | ✓ | What the remote team should accomplish.  Becomes their workflow goal. |
| `context` | — | Free-text background that the remote team receives alongside the goal. |
| `files` | — | Comma-separated local workspace paths to send with the task. |
| `timeout` | — | Seconds to wait for the remote team to finish (default: 600). |

When the tool returns, any files the remote team produced are written into
Lab A's local workspace, ready for subsequent tool calls (`read_file`,
`run_python`, etc.).

### Named peer registry

Instead of hard-coding URLs in every tool call, declare known peers in
`bridge.peers`:

```yaml
# lab-a.yaml
bridge:
  secret: "shared-secret"
  peers:
    lab-b: http://lab-b.example.com:7001
    ml-cluster: http://gpu01.example.com:7002
```

Members can now use `peer: lab-b` in `delegate_task`, `broadcast_task`, and
`cancel_remote_task`.  The `list_peers` tool reports live health for all
configured peers before you commit to a delegation:

````
```tool:list_peers
```
````

```text
Configured peers:
  lab-b: http://lab-b.example.com:7001  [ok · pending=0 · running=1]
  ml-cluster: http://gpu01.example.com:7002  [unreachable: connection refused]
```

### Broadcasting to multiple teams

`broadcast_task` fans out the same goal to several peers concurrently and
collects all results.  Ideal for ensemble processing (send to N specialist
teams, compare answers) or embarrassingly parallel work:

````
```tool:broadcast_task
peers: lab-b, lab-c, lab-d
goal: Independently verify the survival analysis results.
context: Our primary result is in analysis/survival_primary.csv
files: analysis/survival_primary.csv
timeout: 600
```
````

Each peer's result is returned under a `<peer-name>/…` path in the local
workspace so files from different peers never overwrite each other.

### Cancelling a remote task

If a delegated task is no longer needed (e.g. you found the answer from
another peer), cancel it to free up the remote team's concurrency slot:

````
```tool:cancel_remote_task
peer: lab-b
task_id: <UUID returned by delegate_task>
```
````

Returns `Cancelled: <task_id>` on success.  Returns an error if the task is
not found or has already reached a terminal state.

### Server HTTP API reference

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/tasks` | Submit a new task; returns `{"task_id": "…"}` |
| `GET` | `/tasks` | List all tasks (add `?status=pending\|running\|complete\|error\|cancelled` to filter) |
| `GET` | `/tasks/{id}` | Poll the status/result of a specific task |
| `DELETE` | `/tasks/{id}` | Cancel a queued or running task |
| `GET` | `/capabilities` | Advertise team name, models, personas, tools, MCP servers, and version |
| `GET` | `/health` | Quick health check with pending/running counts |

### Bridge config reference

Add a `bridge:` section to your YAML to configure the server behaviour:

```yaml
bridge:
  listen_port: 7001           # default port for `team serve` (default: 7000)
  max_concurrent_tasks: 2    # allow up to 2 simultaneous remote tasks (default: 1)
  secret: "change-me"        # shared secret for HMAC-SHA256 authentication
  task_ttl_seconds: 3600     # evict completed/errored tasks after this many seconds (default: 3600)
  peers:                     # named peers this team can delegate to
    lab-b: http://lab-b.example.com:7001
    ml-cluster: http://gpu01.example.com:7002
```

The `--port` flag on `team serve` overrides `listen_port` at runtime.

### Security — HMAC-SHA256 shared secret

Every bridge request is authenticated with a **shared secret** known only to
the two collaborating labs.  Both sides must set the **same value** under
`bridge.secret` in their respective team YAML files.

```yaml
# lab-a.yaml
bridge:
  secret: "super-secret-lab-key-change-me"

# lab-b.yaml
bridge:
  listen_port: 7001
  secret: "super-secret-lab-key-change-me"
```

The client signs every outgoing request with
`HMAC-SHA256(secret, "{unix_timestamp}:{raw_body}")` and attaches two headers:

| Header | Description |
|--------|-------------|
| `X-Bridge-Timestamp` | Unix timestamp (integer seconds) |
| `X-Bridge-Signature` | HMAC-SHA256 hex digest |

The server rejects requests that:
* are missing either header → `401 Unauthorized`
* have a timestamp older than **5 minutes** (replay-attack protection) → `401`
* carry an invalid signature → `401`

If `bridge.secret` is **not set** the server accepts all requests (open mode,
backward compatible — use only on fully trusted private networks).

### Additional security considerations

> **The bridge server runs your team's full LLM workflow — including any
> enabled tools such as `run_python` and `run_bash` — for every task it
> receives.**  Always set `bridge.secret`; only expose a bridge server to
> networks you trust.

Practical recommendations:
* Always set a strong, random `bridge.secret` on both sides (treat it like a
  database password).
* Run `team serve` behind a reverse proxy (nginx, Caddy) with TLS if the
  server is reachable from the public internet.
* Restrict the tools available to remote-triggered runs to the minimum
  needed (e.g. disable `run_bash` if the remote goal is purely analytical).
* Set `max_concurrent_tasks: 1` (the default) if your hardware cannot
  safely support parallel model runs.

---


## Team registry (service discovery)

The registry is a lightweight HTTP directory where **running team clusters
advertise their capabilities** — name, bridge URL, speciality tags, models,
and available tools.  Any member (or another team) can then *query* the
registry to find the right specialist cluster and delegate work to it, without
hard-coding the bridge URL in every YAML.

This is the dynamic counterpart to `bridge.peers`: peers are pre-configured
point-to-point links; the registry enables dynamic discovery across an
unlimited number of teams.

### Starting a standalone registry server

```bash
team registry start --port 8000
# optional HMAC secret for write operations:
team registry start --port 8000 --secret mysecret
```

The registry server listens on `0.0.0.0:8000` by default.  Teams that have
not sent a heartbeat within `--entry-ttl` seconds (default: 300) are
automatically evicted.

### Registering a team

**Option A — one-shot registration from the CLI:**

```bash
team registry register myteam.yaml http://registry.example.com:8000
```

This reads the team name, description, tags, models, and tools from the YAML
and posts them to the registry.  Use `--heartbeat 60` to keep the entry alive.

**Option B — auto-registration inside `team serve`:**

Add a `registry:` section to the team YAML:

```yaml
registry:
  url: http://registry.example.com:8000
  auto_register: true          # register automatically on team serve
  heartbeat_interval: 60       # seconds between heartbeats
  tags:
    - biology
    - python
    - statistics
  description: "Computational biology team — survival analysis, RNA-seq, ML."
```

When you run `team serve myteam.yaml`, the bridge server starts and the team
registers itself with the registry.  A background heartbeat thread keeps the
entry alive.  On `Ctrl-C` the server deregisters gracefully.

### Discovering teams

**From the CLI:**

```bash
# list all registered teams
team registry list http://registry.example.com:8000

# find teams that handle biology AND statistics
team registry find http://registry.example.com:8000 --tag biology --tag statistics

# keyword search
team registry find http://registry.example.com:8000 --keyword "survival analysis"
```

**From an agent tool call:**

```tool:query_registry
url: http://registry.example.com:8000
tag: biology
tag: statistics
keyword: survival analysis
```

The tool returns a Markdown list of matching teams with their names, bridge
URLs, tags, and descriptions.  An agent can then immediately delegate work
using `delegate_task` with the URL from the result.

### `registry:` config reference

| Field | Default | Description |
| --- | --- | --- |
| `url` | `null` | URL of the registry server. Required for auto-registration and as the default for `query_registry` tool calls. |
| `auto_register` | `false` | Register this team with the registry when `team serve` starts. |
| `heartbeat_interval` | `60` | Seconds between heartbeat pings to the registry server. |
| `tags` | `[]` | Capability labels advertised to other teams. Can be a YAML list or a comma-separated string. |
| `description` | `""` | Short human-readable description advertised in the registry. |

### Registry HTTP API reference

| Method | Path | Auth required | Description |
| --- | --- | --- | --- |
| `POST` | `/register` | ✓ | Register or update a team entry (JSON body = `RegistryEntry`). |
| `DELETE` | `/teams/<name>` | ✓ | Deregister a team. |
| `POST` | `/heartbeat/<name>` | ✓ | Refresh the TTL for a registered team. Returns `404` if not found. |
| `GET` | `/teams` | — | List all registered teams. Accepts `?tag=X&tag=Y&keyword=Z` query params. |
| `GET` | `/teams/<name>` | — | Get a single team by name. |
| `GET` | `/health` | — | `{"status":"ok","registered":N}`. |

Auth uses the same HMAC-SHA256 scheme as the bridge (headers
`X-Registry-Timestamp` / `X-Registry-Signature`).  Read operations are always
public.

---


## Federated belief board

Two independent ``team`` clusters can share their **collective knowledge** via
the bridge protocol.  When Team A has strong consensus on a set of findings,
it can push them to Team B.  Team B receives the beliefs as *pending*, subjects
them to its own consensus process, and either accepts or contests them.  This
models how real research groups share, challenge, and build on each other's
knowledge.

### How federation works

Each bridge server exposes two new endpoints when `beliefs.enabled: true`:

* **`GET /beliefs[?status=accepted&limit=50]`** — export the team's belief
  board as a JSON bundle (publicly readable).
* **`POST /beliefs/import`** — import an incoming belief bundle (auth-gated).

Belief deduplication is based on a **claim hash** — a SHA-256 digest of the
normalized claim text (lowercased, punctuation stripped, whitespace collapsed).
A belief is skipped if a belief with the same normalized claim already exists
locally, regardless of status.

Imported beliefs carry author attribution in the form
``"<source_team>/<original_author>"`` so members can distinguish external
beliefs from local ones.

### Synchronizing via the `sync_beliefs` tool

Members can call `sync_beliefs` from inside a workflow:

```tool:sync_beliefs
url: http://lab-b.example.com:7001
direction: pull
status: accepted
limit: 20
```

Or push local knowledge outward:

```tool:sync_beliefs
url: http://lab-b.example.com:7001
direction: push
status: accepted
local_team: lab-a
```

Or do both in one call:

```tool:sync_beliefs
peer: lab-b
direction: both
status: accepted
```

The tool returns a human-readable summary:

```
Pulled from lab-b: 3 belief(s) imported as pending, 1 skipped (already known).
Tip: use list_beliefs to review the new pending beliefs and accept_belief /
     contest_belief to vote on them.
```

### Synchronizing from the CLI

```bash
# Pull accepted beliefs from Lab B
team beliefs-sync lab-a.yaml http://lab-b.example.com:7001 \
    --direction pull --status accepted

# Push and pull simultaneously
team beliefs-sync lab-a.yaml http://lab-b.example.com:7001 \
    --direction both

# With HMAC authentication
team beliefs-sync lab-a.yaml http://lab-b.example.com:7001 \
    --direction pull --secret mysharedsecret
```

### `sync_beliefs` tool parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `url` | — | Bridge URL of the remote team (required unless `peer:` is used). |
| `peer` | — | Named peer from `bridge.peers` — resolved to its URL. |
| `direction` | `pull` | `pull`, `push`, or `both`. |
| `status` | (all) | Filter beliefs by status (e.g. `accepted`). |
| `limit` | `50` | Max beliefs to transfer per direction. |
| `local_team` | `local` | Name to use as source when pushing. |

### Bridge server belief endpoints

The `GET /beliefs` and `POST /beliefs/import` endpoints are served by any
bridge server started with `team serve` when the team's `beliefs.json` exists.
No additional configuration is required.

Auth for `POST /beliefs/import` uses the same HMAC-SHA256 scheme as the bridge
(headers `X-Bridge-Timestamp` / `X-Bridge-Signature`).  `GET /beliefs` is
always public.

---


## Examples

Two ready-to-run examples ship with the project:

### `examples/academic_lab.yaml`

A computational-biology lab investigating a survival-analysis hypothesis
on TCGA-BRCA.  Four members:

* **@pi** (`llama3.1:70b`) — Principal Investigator, sets direction.
* **@postdoc** (`llama3.1:8b`) — drafts the manuscript.
* **@data_scientist** (`qwen2.5-coder:7b`) — writes the analysis script.
* **@reviewer** (`llama3.1:8b`) — peer-reviews until satisfied.

Workflow: `review_loop` (postdoc ⇄ reviewer), with the PI and data
scientist contributing files to the shared workspace.

```bash
team run examples/academic_lab.yaml
```

### `examples/software_team.yaml`

A small product team designing/implementing/testing a CLI utility.
Three members; **manager-driven** workflow (the tech lead picks the next
speaker).

```bash
team run examples/software_team.yaml
```

> Tip: with `team validate <file>` you can lint a spec without touching
> Docker — useful in CI.

---


## Architecture overview

```
team/
├── _version.py
├── config.py        # YAML → TeamConfig (dataclasses, validation)
├── ollama_client.py # HTTP clients for Ollama and OpenAI-compat APIs; token usage
├── container.py     # Docker lifecycle: per-team network/volumes/containers
├── workspace.py     # parse `file:` blocks, atomic writes, traversal guard, CheckpointManager
├── bus.py           # transcript with on-disk JSONL persistence and stats()
├── personas.py      # render the system prompt + collaboration protocol + tool section
├── mcp/             # MCP-native tool layer:
│   ├── bus.py       #   MCPToolBus + MemberToolset (background loop, sessions, dispatch)
│   ├── textmode.py  #   text-mode tool block parsing + prompt rendering
│   └── builtin/     #   25 built-in tools as 8 in-process MCP servers
├── memory.py        # AgentMemory: per-agent SQLite-backed persistent cross-session memory
├── beliefs.py       # BeliefBoard: shared JSON-backed team belief board with voting/consensus
├── persona_library.py # lazy loader for personas/ YAML files + TEAM_PERSONA_DIR support
├── member.py        # Member: persona + container runtime + chat client + agentic loop
├── workflows.py     # round_robin / manager / review_loop / sequential_chain / debate
├── orchestrator.py  # ties everything together, drives the workflow
├── bridge.py        # bridge protocol: BridgeTask, BridgeResult, TaskStore
├── bridge_server.py # HTTP bridge server (team serve): accept tasks, run workflows
├── bridge_client.py # HTTP bridge client: submit_task, poll_result, wait_for_result
├── wizard.py        # interactive `team new` wizard
└── cli.py           # `team` command (Click + Rich)
```

Adding a workflow is ~30 lines of Python: write a function
`my_workflow(orch)` and register it in `team/workflows.py::WORKFLOWS`.
The whole surface a workflow needs is `orch.members`, `orch.run_turn(name, prompt=...)`,
and reading `result.declared_done` / `result.content`.

---


## Development

```bash
pip install -e ".[dev]"
pytest -q
```

The unit tests do **not** require Docker or a running Ollama: they
exercise config parsing, the file-block parser/safety guard, transcript
rendering, the persona renderer, and every workflow against a fake
orchestrator.

The bridge **integration** tests (`TestBridgeIntegration` in
`tests/test_bridge.py`) spin up a real in-process HTTP server on
`127.0.0.1`.  They are automatically **skipped** when TCP loopback
connections are unavailable in the test environment (e.g. some
sandboxed CI runners).

CI: `.github/workflows/tests.yml` runs `pytest` on Python 3.10–3.12.

---


## Troubleshooting

* **`docker.errors.DockerException: ... permission denied`** — your user
  is not in the `docker` group.  `sudo usermod -aG docker $USER` and
  re-login.
* **Model pull is slow / times out** — bump `defaults.pull_timeout` (or
  the `--prepare-timeout` CLI flag).  First-time pulls of a 70B model
  can take a long time.
* **Out of GPU memory** — pin a smaller model to the heavy roles, or set
  `gpus: none` for some members so they run on CPU.
* **A member ignores the `file:` protocol** — try a more capable model
  for that role; smaller models sometimes need an `extra_system` hint
  reiterating "always emit deliverables in `\`\`\`file:...\`\`\` blocks".
* **Containers won't stop** — `team down --purge <team.yaml>` force-
  removes containers and per-member model volumes.

---


## License

MIT — see [LICENSE](LICENSE).

