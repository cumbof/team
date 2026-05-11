# team

> **Orchestrate a cluster of containerized local LLMs — each with its own
> persona, role, and goal — that collaborate until the work is done.**

---

> [!NOTE]
> **AI-assisted development notice**
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
- [Workflows](#workflows)
- [Workspaces and artifacts](#workspaces-and-artifacts)
- [Containers, isolation, and root](#containers-isolation-and-root)
- [GPU support](#gpu-support)
- [CLI reference](#cli-reference)
- [OpenAI-compatible backends](#openai-compatible-backends)
- [Remote / no-Docker Ollama](#remote--no-docker-ollama)
- [Context window management](#context-window-management)
- [Agent mode and tool use](#agent-mode-and-tool-use)
  - [Built-in tools](#built-in-tools)
  - [Custom skill plugins](#custom-skill-plugins)
- [Token usage tracking](#token-usage-tracking)
- [Run statistics](#run-statistics)
- [Cross-team collaboration (bridge)](#cross-team-collaboration-bridge)
  - [How it works](#how-it-works-1)
  - [Exposing a team as a bridge server](#exposing-a-team-as-a-bridge-server)
  - [Delegating work from another team](#delegating-work-from-another-team)
  - [Bridge config reference](#bridge-config-reference)
  - [Security considerations](#security-considerations)
- [Per-agent persistent memory](#per-agent-persistent-memory)
  - [Enabling memory](#enabling-memory)
  - [Memory tools](#memory-tools)
  - [Memory config reference](#memory-config-reference)
- [Shared team belief board](#shared-team-belief-board)
  - [Enabling the belief board](#enabling-the-belief-board)
  - [Belief tools](#belief-tools)
  - [Inspecting beliefs with team beliefs](#inspecting-beliefs-with-team-beliefs)
  - [Belief config reference](#belief-config-reference)
- [Workspace time-travel (`team rollback`)](#workspace-time-travel-team-rollback)
- [Interactive wizard](#interactive-wizard)
- [Workflow visualization](#workflow-visualization)
- [Custom Ollama image](#custom-ollama-image)
- [Streaming output](#streaming-output)
- [Retry and back-off](#retry-and-back-off)
- [Pre-flight checks](#pre-flight-checks)
- [Exporting a run report](#exporting-a-run-report)
- [Resuming an interrupted run](#resuming-an-interrupted-run)
- [Workspace checkpoints](#workspace-checkpoints)
- [Human-in-the-loop intervention](#human-in-the-loop-intervention)
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
                 ┌────────────────── orchestrator (host) ──────────────────┐
                 │                                                          │
                 │   transcript.jsonl     shared workspace (./runs/<team>)  │
                 │        ▲                       ▲                         │
                 │        │ append every turn     │ files written by members│
                 └────┬───┴────────────┬──────────┴─────────────┬───────────┘
                      │                │                        │
                      ▼                ▼                        ▼
       ┌──────────────────┐  ┌──────────────────┐    ┌──────────────────┐
       │ container: pi    │  │ container: postdoc│    │ container: ...  │
       │ ollama serve     │  │ ollama serve     │    │                  │
       │ model: 70B       │  │ model: 8B        │    │                  │
       │ /workspace (ro+) │  │ /workspace (ro+) │    │ /workspace (ro+) │
       │ /private         │  │ /private         │    │ /private         │
       └──────────────────┘  └──────────────────┘    └──────────────────┘
                       \\           |            //
                        \\          |           //
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
| `tools` | list | `[]` | Built-in tools enabled for all members by default. |
| `max_tool_rounds` | int | `10` | Maximum agentic tool-call rounds per member turn. |
| `tool_timeout` | int | `30` | Seconds budget per individual tool execution. |
| `skills` | list | `[]` | Skill plugin sources (local paths or remote URLs) available to all members. |

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
| `tools` | no | List of tool names this member may use (e.g. `[web_search, run_python]`). |
| `max_tool_rounds` | no | Per-member override of the tool-round limit. |
| `tool_timeout` | no | Per-member override of the per-tool execution timeout (seconds). |
| `skills` | no | Member-specific skill sources merged with `defaults.skills`. |

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

---

## CLI reference

```text
team init        [PATH]               Write a starter team YAML.
team new         [PATH]               Interactive wizard to create a new team YAML.
team validate    <team.yaml>          Parse and validate the YAML.
team check       <team.yaml>          Run preflight checks (no Docker started).
team visualize   <team.yaml>          Print an ASCII or Mermaid diagram of the workflow.
                 [--format ascii|mermaid]
team up          <team.yaml>          Start containers, pull models.
team status      <team.yaml>          Show container status per member.
team logs        <team.yaml>          Tail per-member Ollama logs.
                 [--member NAME] [--tail N]
team run         <team.yaml>          Up + run workflow + (down).
                 [--no-up] [--keep-up] [--resume] [--no-stream] [--interactive]
team transcript  <team.yaml>          Render the persisted transcript.
team export      <team.yaml>          Export transcript + artifacts to a report.
                 [--format markdown|html] [--output PATH]
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

## Retry and back-off

When an Ollama request fails due to a transient network problem or a 5xx
server error, `team` retries automatically with exponential back-off before
giving up.  Configure it in `defaults`:

```yaml
defaults:
  max_retries: 3        # total extra attempts after the first (default: 3)
  retry_backoff: 2.0    # wait = backoff ** attempt → 1 s, 2 s, 4 s … (default: 2.0)
```

| Condition | Retried? |
|---|---|
| `requests.ConnectionError` / `Timeout` | ✓ yes |
| HTTP 5xx (server error) | ✓ yes |
| HTTP 4xx (client error, bad model name, …) | ✗ no — fails immediately |
| Empty response body | ✗ no — fails immediately |

For streaming turns, retries only happen if no tokens have been yielded yet
(a partial stream cannot be safely replayed).

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

## Exporting a run report

After a run you can bundle the full transcript and every produced artifact
into a single shareable document:

```bash
team export my-team.yaml                          # Markdown (default)
team export my-team.yaml --format html            # self-contained HTML
team export my-team.yaml --output ~/Desktop/run.md
```

The report includes:
* Team name, goal, members, and workflow settings.
* Every member turn with speaker, role, content, and files written.
* Full contents of all files produced in the shared workspace.

The default output path is `<workspace>/report.md` (or `.html`).  The
HTML variant is a fully self-contained file with embedded CSS — no
external dependencies required.

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
┌──────────────────────────────┬──────┬─────────────────────┬─────────────────────┬───────┐
│ ID                           │ Turn │ Before member's turn │ Timestamp           │ Files │
├──────────────────────────────┼──────┼─────────────────────┼─────────────────────┼───────┤
│ 0001_alice_20240501T120000   │    1 │ @alice               │ 2024-05-01 12:00:00 │     3 │
│ 0003_bob_20240501T120145     │    3 │ @bob                 │ 2024-05-01 12:01:45 │     5 │
└──────────────────────────────┴──────┴─────────────────────┴─────────────────────┴───────┘
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

No container is started for that member; the orchestrator connects directly
to the given URL.  The model must already be pulled on that server (or
Ollama's automatic pull will fetch it on first use).

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
| `summarize` | Same as `truncate` (future: will use a lightweight model to summarise omitted turns). |

Override per member:

```yaml
members:
  - name: reviewer
    context_strategy: sliding_window
    context_budget: 10    # this member sees only the last 10 turns
```

---

## Agent mode and tool use

Members can act as **agents**: they may call external tools by emitting a
special fenced code block in their reply, then receive the tool's output and
continue reasoning — all within the same logical turn.

### Enabling tools

```yaml
defaults:
  tools: [web_search, run_python]  # enable globally
  max_tool_rounds: 10              # max tool-call rounds per turn (default: 10)
  tool_timeout: 30                 # seconds per tool execution (default: 30)

members:
  - name: researcher
    tools: [web_search, read_url]  # per-member override
  - name: data_scientist
    tools: [run_python, run_bash, read_file, write_file, append_file, list_files]
```

### Tool invocation syntax

A member invokes a tool by emitting a fenced block with a `tool:<name>`
info-string:

````
```tool:web_search
query: IPCC AR6 key findings 2024
```
````

````
```tool:run_python
import pandas as pd
df = pd.read_csv('/workspace/shared/data.csv')
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

| tool | description |
| --- | --- |
| `run_python` | Execute Python code; cwd is the shared workspace directory. |
| `run_bash` | Execute a bash command; cwd is the shared workspace directory. |
| `web_search` | Search the web via the DuckDuckGo instant-answer API (no key required). |
| `read_url` | Fetch and return the plain-text content of a URL. |
| `read_file` | Read a file from the shared workspace by relative path. |
| `write_file` | Write (create or overwrite) a file in the shared workspace. |
| `append_file` | Append text to a file in the shared workspace. |
| `list_files` | List files in the shared workspace with an optional glob filter. |
| `remember` | Store a memory in the member's **persistent cross-session** memory store. |
| `recall` | Search the member's persistent memory by keyword. |
| `forget` | Delete a memory by key from the persistent store. |
| `list_memories` | List stored memories (optionally filtered by tag). |
| `assert_belief` | Add a claim to the team's **shared belief board** with confidence score. |
| `contest_belief` | Contest an existing belief (moves it to contested status). |
| `accept_belief` | Cast an accept vote for an existing belief. |
| `list_beliefs` | List the shared belief board (optionally filtered by status). |
| `delegate_task` | Delegate a sub-task to a remote bridge server and wait for results. |

**`write_file` and `append_file` body format**

Both tools use a two-part body separated by a `---` line:

```
path: relative/path/to/file.txt
---
File content goes here.
Multiple lines are fine.
```

The path is relative to the shared workspace root.  Parent directories are
created automatically.  `write_file` replaces any existing content;
`append_file` adds to the end of the file (creating it if it does not exist).

**`list_files` body format**

The body is optional.  If omitted, all workspace files are listed.  Use a
`pattern:` key to filter by glob pattern:

```
pattern: **/*.py
```

### Security note

`run_python` and `run_bash` execute code on the **host machine** with the
privileges of the `team` process.  Only enable these tools for members whose
prompts you trust.

### How it works

```
member turn:
  1. LLM called with system prompt + conversation context
  2. If reply contains tool blocks → execute each tool
  3. Tool results injected as a follow-up user message
  4. LLM called again (no streaming; repeats up to max_tool_rounds)
  5. If no tool blocks in reply → reply recorded in transcript
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

### Custom skill plugins

The built-in tool set is a starting point.  You can extend it with any
Python file — local or fetched from a URL — and make those tools
available to any member.  This gives agents effectively **unlimited**
capabilities depending on what skills you provide.

#### Skill file format

A skill file must expose tools in one of two formats:

**Single-tool format** (`TOOL_NAME` + `execute`):

```python
# skills/my_calculator.py
TOOL_NAME = "my_calculator"
TOOL_DESCRIPTION = "Evaluate a Python arithmetic expression."

def execute(body, *, workspace_path=None, timeout=30, **kwargs):
    try:
        return str(eval(body.strip(), {"__builtins__": {}}, {}))
    except Exception as exc:
        return f"ERROR: {exc}"
```

**Multi-tool format** (`TOOLS` dict + optional `TOOL_DESCRIPTIONS`):

```python
# skills/db_tools.py
import sqlite3

def _query(body, *, workspace_path=None, **kwargs):
    db_path = workspace_path / "data.sqlite"
    conn = sqlite3.connect(db_path)
    rows = conn.execute(body.strip()).fetchall()
    conn.close()
    return "\n".join(str(r) for r in rows)

def _schema(body, *, workspace_path=None, **kwargs):
    db_path = workspace_path / "data.sqlite"
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT name, sql FROM sqlite_master WHERE type='table'").fetchall()
    conn.close()
    return "\n".join(f"{r[0]}: {r[1]}" for r in rows)

TOOLS = {"sql_query": _query, "sql_schema": _schema}
TOOL_DESCRIPTIONS = {
    "sql_query":  "Run an SQL SELECT on the shared SQLite database.",
    "sql_schema": "Return the schema of all tables in the shared SQLite database.",
}
```

Both formats can coexist in the same file.

#### Configuring skills

Add skill sources under `defaults.skills` (inherited by all members) or
`members[*].skills` (member-specific, merged with defaults on top):

```yaml
defaults:
  skills:
    - path: ./skills/my_calculator.py     # local path (relative to CWD)
    - path: ./skills/db_tools.py
    - url: https://example.com/skill.py   # remote URL (see security note below)
      checksum: sha256:e3b0c44298fc…      # optional integrity check
    - ./skills/shorthand.py               # plain string = auto-detect local/remote

  tools: [web_search, my_calculator, sql_query, sql_schema]  # opt-in by name

members:
  - name: analyst
    tools: [sql_query, sql_schema, run_python]   # member-specific tool set
    skills:
      - ./skills/analyst_helpers.py              # member-specific extra skill
```

Tool names from skills are used exactly like built-in tool names everywhere
(`tools:` lists, `tool:` fenced blocks, system prompts).

#### Checksum verification

For any skill (local or remote) you can supply a checksum to verify
integrity before execution:

```yaml
skills:
  - url: https://example.com/skill.py
    checksum: sha256:<hex-digest>
  - path: ./skills/local.py
    checksum: sha256:<hex-digest>
```

Supported algorithms: any name accepted by Python's `hashlib` (e.g.
`sha256`, `sha512`, `md5`).  `team` raises an error and refuses to load
the skill if the digest does not match.

#### Security

> **Remote skills execute arbitrary Python code on the host machine with
> the privileges of the `team` process.**  Treat a remote skill URL with
> the same caution as `curl URL | python`.  Always use `checksum:` for
> remote skills in production.

Local skills (from your own filesystem) are as trustworthy as any other
code you run; they are loaded in the same security context as `run_python`.

---

## Token usage tracking

After every `team run` a token usage summary is printed:

```text
┌────────────────────────────────────────────────────┐
│              Token usage (live turns)               │
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
│  Orchestrator A                      │     │  team serve lab-b.yaml           │
│  members: pi, analyst               │     │  BridgeServer (port 7001)        │
│                                     │     │                                  │
│  @pi uses delegate_task tool ───────┼─────┼──► POST /tasks                  │
│                                     │     │    ┌──────────────────────────┐  │
│                                     │     │    │ Orchestrator B            │  │
│                                     │     │    │ members: coder, reviewer  │  │
│                                     │     │    │ runs full workflow        │  │
│                                     │     │    └──────────────────────────┘  │
│  result written to workspace ◄──────┼─────┼─── GET /tasks/{id}  (complete)  │
│  injected into transcript           │     │    files + summary returned      │
└─────────────────────────────────────┘     └──────────────────────────────────┘
```

1. **Lab B** exposes its cluster by running `team serve`.
2. **Lab A's** agents use the `delegate_task` built-in tool, specifying Lab
   B's URL, a goal, optional context, and optional workspace files to send.
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
  tools: [delegate_task, read_file, write_file]
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

| field | required | description |
| --- | --- | --- |
| `url` | ✓ | Base URL of the remote `team serve` endpoint. |
| `goal` | ✓ | What the remote team should accomplish.  Becomes their workflow goal. |
| `context` | — | Free-text background that the remote team receives alongside the goal. |
| `files` | — | Comma-separated local workspace paths to send with the task. |
| `timeout` | — | Seconds to wait for the remote team to finish (default: 600). |

When the tool returns, any files the remote team produced are written into
Lab A's local workspace, ready for subsequent tool calls (`read_file`,
`run_python`, etc.).

### Bridge config reference

Add a `bridge:` section to your YAML to configure the server behaviour:

```yaml
bridge:
  listen_port: 7001          # default port for `team serve` (default: 7000)
  max_concurrent_tasks: 2   # allow up to 2 simultaneous remote tasks (default: 1)
```

The `--port` flag on `team serve` overrides `listen_port` at runtime.

### Security considerations

> **The bridge server runs your team's full LLM workflow — including any
> enabled tools such as `run_python` and `run_bash` — for every task it
> receives.**  Only expose a bridge server to networks you trust.

Practical recommendations:

* Run `team serve` behind a reverse proxy (nginx, Caddy) with TLS and
  authentication if the server is reachable from the public internet.
* Restrict the tools available to remote-triggered runs to the minimum
  needed (e.g. disable `run_bash` if the remote goal is purely analytical).
* Set `max_concurrent_tasks: 1` (the default) if your hardware cannot
  safely support parallel model runs.

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
    tools: [run_python, remember, recall, forget, list_memories]
```

### Memory tools

All memory tools use a `key:` / header + `---` / value body format:

**`remember`** — store a cross-session memory:

```
```tool:remember
key: protein_folding_baseline_2025
tags: results, methods
importance: 0.9
---
AlphaFold3 outperforms RoseTTAFold on monomers (RMSD 1.2 vs 2.1 Å, n=1 000).
Dataset: PDB validation set, tested January 2025.
```
```

**`recall`** — full-text search across all memories:

```
```tool:recall
query: protein folding
limit: 5
```
```

Returns a ranked list of matching memories (by importance then recency).

**`forget`** — delete a memory by key:

```
```tool:forget
key: protein_folding_baseline_2025
```
```

**`list_memories`** — browse all memories (optionally by tag):

```
```tool:list_memories
tag: results
limit: 20
```
```

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
    tools: [run_python, assert_belief, contest_belief, accept_belief, list_beliefs]
```

### Belief tools

**`assert_belief`** — propose a claim with optional evidence:

```
```tool:assert_belief
confidence: 0.85
evidence: RMSD analysis, PDB validation set, n=1 000, January 2025
---
AlphaFold3 is the best available method for monomer structure prediction.
```
```

The member who asserts a belief automatically casts an *accept* vote.  The
returned belief ID (e.g. `a3f2b1c9`) is used in subsequent votes.

**`accept_belief`** — vote to accept:

```
```tool:accept_belief
id: a3f2b1c9
```
```

**`contest_belief`** — move a belief to `contested` status:

```
```tool:contest_belief
id: a3f2b1c9
reason: Dataset is limited to well-studied proteins; may not generalise.
```
```

**`list_beliefs`** — browse the board:

```
```tool:list_beliefs
status: contested
```
```

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
┏━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━┳━━━━━┳━━━━━━━━━┓
┃ ID     ┃ Status     ┃ Claim                                                    ┃ Confidence ┃ By    ┃ For ┃ Against ┃
┡━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━╇━━━━━╇━━━━━━━━━┩
│ a3f2b1 │ ✓ accepted │ AlphaFold3 is best for monomer structure prediction.     │       85%  │ @alice│   2 │       0 │
│ 9c1d33 │ ⚡ contested│ The dataset generalises to all protein families.        │       60%  │ @bob  │   1 │       1 │
└────────┴────────────┴──────────────────────────────────────────────────────────┴────────────┴───────┴─────┴─────────┘
⚡ Some beliefs are contested — review and resolve via accept_belief / contest_belief tools.
```

### Belief config reference

| key | type | default | description |
| --- | --- | --- | --- |
| `enabled` | bool | `false` | Enable the shared belief board. |
| `consensus_threshold` | float | `0.5` | Fraction of members who must accept a belief for it to become `accepted`. |
| `inject_limit` | int | `10` | Maximum number of beliefs injected into each member's turn context. |

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

## Workflow visualization

`team visualize` renders an ASCII or Mermaid flowchart of a team's
workflow.  Useful for documentation, code review, and reasoning about
large team configs:

```bash
team visualize my-team.yaml               # ASCII (default)
team visualize my-team.yaml --format mermaid
```

ASCII example for a `review_loop` team:

```
  ┌────────────────────────────────────────────────────┐
  │         review_loop (max 4 rounds)                  │
  │                                                    │
  │  @postdoc  ──draft──►  @reviewer                  │
  │     ▲                       │                     │
  │     └───── revise ──────────┘                     │
  │                             │                     │
  │                         APPROVED ──► [[DONE]]      │
  └────────────────────────────────────────────────────┘
```

Mermaid output can be pasted directly into GitHub Markdown or rendered
with any Mermaid-compatible tool.

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
├── tools.py         # built-in agent tools: run_python, run_bash, web_search, read_url, read_file, write_file, append_file, list_files, delegate_task, remember, recall, forget, list_memories, assert_belief, contest_belief, accept_belief, list_beliefs
├── skills.py        # skill plugin loader: local files and remote URLs → tool registry
├── memory.py        # AgentMemory: per-agent SQLite-backed persistent cross-session memory
├── beliefs.py       # BeliefBoard: shared JSON-backed team belief board with voting/consensus
├── member.py        # Member: persona + container runtime + chat client + agentic loop
├── workflows.py     # round_robin / manager / review_loop / sequential_chain / debate
├── orchestrator.py  # ties everything together, drives the workflow
├── bridge.py        # bridge protocol: BridgeTask, BridgeResult, TaskStore
├── bridge_server.py # HTTP bridge server (team serve): accept tasks, run workflows
├── bridge_client.py # HTTP bridge client: submit_task, poll_result, wait_for_result
├── visualize.py     # ASCII and Mermaid diagram renderer
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
