# team

> **Orchestrate a cluster of containerized local LLMs — each with its own
> persona, role, and goal — that collaborate until the work is done.**

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
- [Custom Ollama image](#custom-ollama-image)
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

### `workflow`

```yaml
workflow:
  type: review_loop
  max_rounds: 4
  producer: postdoc
  reviewer: reviewer
  approve_token: APPROVED   # only review_loop; default "APPROVED"
  manager: tech_lead        # only when type=manager
```

| `type` | extra options |
| --- | --- |
| `round_robin` | none |
| `manager` | `manager: <member name>` |
| `review_loop` | `producer: <member>`, `reviewer: <member>`, optional `approve_token` |

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

---

## Workspaces and artifacts

For team `<name>` with `workspace: ./runs/<name>` you get:

```
runs/<name>/
├── transcript.jsonl       # one JSON object per turn
├── shared/                # mounted as /workspace inside every container
│   └── <files written by members>
└── members/
    ├── pi/                # mounted as /private inside the pi container
    ├── postdoc/
    └── ...
```

* `shared/` is the canonical place for deliverables and is visible to
  every member at every turn.
* `members/<name>/` is private scratch space for that member.
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
team init       [PATH]                Write a starter team YAML.
team validate   <team.yaml>           Parse and validate the YAML.
team up         <team.yaml>           Start containers, pull models.
team status     <team.yaml>           Show container status per member.
team logs       <team.yaml> [--member NAME] [--tail N]
                                       Tail per-member Ollama logs.
team run        <team.yaml> [--no-up] [--keep-up] [--resume] [--no-stream]
                                       Up + run workflow + (down).
team transcript <team.yaml>           Render the persisted transcript.
team export     <team.yaml> [--format markdown|html] [--output PATH]
                                       Export transcript + artifacts to a report.
team down       <team.yaml> [--purge] Stop & remove containers (and volumes).
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
├── ollama_client.py # tiny HTTP client for Ollama (ping/pull/chat)
├── container.py     # Docker lifecycle: per-team network/volumes/containers
├── workspace.py     # parse `file:` blocks, atomic writes, traversal guard
├── bus.py           # transcript with on-disk JSONL persistence
├── personas.py      # render the system prompt + collaboration protocol
├── member.py        # Member: persona + container runtime + chat client
├── workflows.py     # round_robin / manager / review_loop schedulers
├── orchestrator.py  # ties everything together, drives the workflow
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
