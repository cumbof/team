"""Configuration model for a *team*.

A team is described by a YAML file with the following top-level shape::

    name: academic-lab
    goal: |
      Investigate whether <hypothesis>. Produce a publication-ready
      manuscript with figures, tables, and a reproducible analysis.

    workspace: ./runs/academic-lab     # host directory shared with members
    workflow:
      type: review_loop                # round_robin | manager | review_loop
      max_rounds: 8
      # workflow-specific options live here

    defaults:
      ollama_image: ollama/ollama:latest
      context_window: 8192
      temperature: 0.4
      memory_limit: "12g"
      cpu_limit: 4.0
      gpus: "all"                      # "all", "none", or a list of indices

    members:
      - name: pi
        role: Principal Investigator
        model: llama3.1:70b
        persona: |
          You are the PI of a computational biology lab. ...
        temperature: 0.3
      - name: postdoc
        role: PostDoc
        model: llama3.1:8b
        persona: ...
      ...

The :func:`load_team` function returns a fully populated :class:`TeamConfig`
that the rest of the package consumes.  All defaults are filled in here so
downstream code never has to special-case missing fields.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class TeamConfigError(ValueError):
    """Raised when a team YAML file is malformed."""


# --------------------------------------------------------------------------- #
# Dataclasses
# --------------------------------------------------------------------------- #


# Member names must start with a lowercase letter and contain only
# lowercase alphanumerics, hyphens, and underscores.  This mirrors Docker
# container naming rules and keeps names safe for use in file paths.
_NAME_RE = re.compile(r"^[a-z][a-z0-9_-]{0,30}$")


@dataclass
class Defaults:
    ollama_image: str = "ollama/ollama:latest"
    context_window: int = 8192
    temperature: float = 0.4
    top_p: float = 0.9
    memory_limit: str | None = None
    cpu_limit: float | None = None
    gpus: str | list[int] = "none"
    pull_timeout: int = 1800
    request_timeout: int = 600
    max_retries: int = 3
    retry_backoff: float = 2.0
    # F1: backend selection
    backend: str = "ollama"   # "ollama" | "openai_compat"
    api_key: str | None = None
    # F2: context management
    context_strategy: str = "none"  # "none" | "sliding_window" | "truncate" | "summarize"
    context_budget: int = 0         # >0: max turns (sliding_window) or approx token budget
    # F4: agent tool use
    tools: list[str] = field(default_factory=list)  # built-in tool names enabled by default
    max_tool_rounds: int = 10  # max agentic tool-call rounds per member turn
    tool_timeout: int = 300    # seconds budget per individual tool execution (generous for pip/apt)
    # F4 skills: external tool plugins (local paths or remote URLs)
    skills: list[Any] = field(default_factory=list)
    # Host Ollama: if set, all members use this URL instead of Docker containers.
    # Per-member ollama_url overrides this value.  Useful for Apple Silicon / CPU-only
    # hosts where GPU passthrough to Docker is not available.
    ollama_url: str | None = None


@dataclass
class WorkflowConfig:
    type: str = "round_robin"
    max_rounds: int = 6
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class MemberConfig:
    name: str
    role: str
    model: str
    persona: str
    temperature: float | None = None
    top_p: float | None = None
    context_window: int | None = None
    memory_limit: str | None = None
    cpu_limit: float | None = None
    gpus: str | list[int] | None = None
    can_write_files: bool = True
    extra_system: str | None = None  # appended to the rendered system prompt
    # F10: skip Docker and connect to an existing Ollama instance
    ollama_url: str | None = None
    # F1: OpenAI-compatible backend
    backend: str | None = None       # overrides defaults.backend; "ollama" | "openai_compat"
    api_base: str | None = None      # base URL for openai_compat (required when backend=openai_compat)
    api_key: str | None = None       # overrides defaults.api_key; supports "env:VAR"
    # F2: per-member context management overrides
    context_strategy: str | None = None
    context_budget: int | None = None
    # F4: per-member agent tool-use
    tools: list[str] | None = None     # None = inherit from defaults; [] = disable tools
    max_tool_rounds: int | None = None  # None = inherit from defaults
    tool_timeout: int | None = None     # None = inherit from defaults
    skills: list[Any] | None = None     # None = inherit from defaults; [] = no skills


@dataclass
class BridgeConfig:
    """Optional bridge server settings (used by ``team serve``)."""
    listen_port: int = 7000
    max_concurrent_tasks: int = 1


@dataclass
class MemoryConfig:
    """Optional per-agent persistent cross-session memory settings."""
    enabled: bool = False
    inject_recent: int = 5    # number of recent memories injected into each turn's system context
    store: str | None = None  # path to the memory store directory; defaults to <workspace>/memory


@dataclass
class BeliefConfig:
    """Optional shared team belief board settings."""
    enabled: bool = False
    consensus_threshold: float = 0.5  # fraction of members required to accept a belief
    inject_limit: int = 10            # max beliefs shown in each turn's context


@dataclass
class TeamConfig:
    name: str
    goal: str
    workspace: Path
    workflow: WorkflowConfig
    defaults: Defaults
    members: list[MemberConfig]
    source_path: Path | None = None
    bridge: BridgeConfig = field(default_factory=BridgeConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    beliefs: BeliefConfig = field(default_factory=BeliefConfig)

    # Convenience -------------------------------------------------------- #

    def member(self, name: str) -> MemberConfig:
        for m in self.members:
            if m.name == name:
                return m
        raise KeyError(f"No member named {name!r}")

    def member_names(self) -> list[str]:
        return [m.name for m in self.members]


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


def _require(d: dict, key: str, ctx: str) -> Any:
    if key not in d:
        raise TeamConfigError(f"{ctx}: missing required field {key!r}")
    return d[key]


def _check_name(name: str, ctx: str) -> None:
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise TeamConfigError(
            f"{ctx}: invalid name {name!r}; must match {_NAME_RE.pattern}"
        )


def _parse_bridge(data: dict) -> BridgeConfig:
    if not data:
        return BridgeConfig()
    b = BridgeConfig()
    listen_port = data.get("listen_port")
    if listen_port is not None:
        b.listen_port = int(listen_port)
    max_conc = data.get("max_concurrent_tasks")
    if max_conc is not None:
        b.max_concurrent_tasks = int(max_conc)
    return b


def _parse_memory(data: dict) -> MemoryConfig:
    if not data:
        return MemoryConfig()
    m = MemoryConfig()
    if "enabled" in data:
        m.enabled = bool(data["enabled"])
    if "inject_recent" in data:
        m.inject_recent = int(data["inject_recent"])
    if "store" in data and data["store"] is not None:
        m.store = str(data["store"])
    return m


def _parse_beliefs(data: dict) -> BeliefConfig:
    if not data:
        return BeliefConfig()
    b = BeliefConfig()
    if "enabled" in data:
        b.enabled = bool(data["enabled"])
    if "consensus_threshold" in data:
        b.consensus_threshold = float(data["consensus_threshold"])
    if "inject_limit" in data:
        b.inject_limit = int(data["inject_limit"])
    return b


def _parse_defaults(data: dict) -> Defaults:
    d = Defaults()
    for k, v in (data or {}).items():
        if not hasattr(d, k):
            raise TeamConfigError(f"defaults: unknown key {k!r}")
        # Use setattr to apply every key from the YAML without a long
        # if/elif chain — relies on Defaults being a flat dataclass.
        setattr(d, k, v)
    return d


def _parse_workflow(data: dict) -> WorkflowConfig:
    if not data:
        return WorkflowConfig()
    wf_type = data.get("type", "round_robin")
    valid_types = {"round_robin", "manager", "review_loop", "sequential_chain", "debate", "parallel_review"}
    if wf_type not in valid_types:
        raise TeamConfigError(
            f"workflow.type={wf_type!r} is not one of "
            + " | ".join(sorted(valid_types))
        )
    opts = {k: v for k, v in data.items() if k not in {"type", "max_rounds"}}
    return WorkflowConfig(
        type=wf_type,
        max_rounds=int(data.get("max_rounds", 6)),
        options=opts,
    )


def _resolve_persona(raw_persona: str, raw_role: str | None, ctx: str) -> tuple[str, str]:
    """Expand a library shorthand (``@key``) into ``(role, persona_text)``.

    When *raw_persona* starts with ``@`` the remainder is looked up in the
    :mod:`team.persona_library`.  The caller may still override the role by
    supplying *raw_role*; if omitted the library's default role is used.

    Returns ``(role, persona_text)``.  Raises :class:`TeamConfigError` if the
    key is not found.
    """
    if not raw_persona.startswith("@"):
        if raw_role is None:
            raise TeamConfigError(f"{ctx}: missing required field 'role'")
        return raw_role, raw_persona

    key = raw_persona[1:].strip()
    from team.persona_library import PERSONAS  # local import to avoid circulars
    if key not in PERSONAS:
        available = ", ".join(sorted(PERSONAS))
        raise TeamConfigError(
            f"{ctx}: unknown persona library key {key!r}. "
            f"Available: {available}"
        )
    lib_role, lib_persona = PERSONAS[key]["role"], PERSONAS[key]["persona"]
    return raw_role if raw_role is not None else lib_role, lib_persona


def _parse_member(data: dict, defaults: Defaults) -> MemberConfig:
    ctx = f"members[{data.get('name', '?')!r}]"
    name = _require(data, "name", ctx)
    _check_name(name, ctx)
    raw_persona = _require(data, "persona", ctx)
    role, persona = _resolve_persona(raw_persona, data.get("role"), ctx)
    return MemberConfig(
        name=name,
        role=role,
        model=_require(data, "model", ctx),
        persona=persona,
        temperature=data.get("temperature"),
        top_p=data.get("top_p"),
        context_window=data.get("context_window"),
        memory_limit=data.get("memory_limit"),
        cpu_limit=data.get("cpu_limit"),
        gpus=data.get("gpus"),
        can_write_files=bool(data.get("can_write_files", True)),
        extra_system=data.get("extra_system"),
        ollama_url=data.get("ollama_url"),
        backend=data.get("backend"),
        api_base=data.get("api_base"),
        api_key=data.get("api_key"),
        context_strategy=data.get("context_strategy"),
        context_budget=data.get("context_budget"),
        tools=data.get("tools"),
        max_tool_rounds=data.get("max_tool_rounds"),
        tool_timeout=data.get("tool_timeout"),
        skills=data.get("skills"),
    )


def load_team(path: str | os.PathLike) -> TeamConfig:
    """Load and validate a team YAML file."""
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise TeamConfigError(f"team file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise TeamConfigError("top-level YAML must be a mapping")

    name = _require(raw, "name", "team")
    _check_name(name, "team.name")
    goal = _require(raw, "goal", "team")
    workspace = Path(raw.get("workspace") or f"./runs/{name}").expanduser().resolve()
    defaults = _parse_defaults(raw.get("defaults", {}))
    workflow = _parse_workflow(raw.get("workflow", {}))
    bridge = _parse_bridge(raw.get("bridge", {}))
    memory = _parse_memory(raw.get("memory", {}))
    beliefs = _parse_beliefs(raw.get("beliefs", {}))

    members_raw = _require(raw, "members", "team")
    if not isinstance(members_raw, list) or not members_raw:
        raise TeamConfigError("team.members must be a non-empty list")
    members = [_parse_member(m, defaults) for m in members_raw]
    seen: set[str] = set()
    for m in members:
        if m.name in seen:
            raise TeamConfigError(f"duplicate member name: {m.name!r}")
        seen.add(m.name)

    if workflow.type == "manager":
        manager = workflow.options.get("manager")
        if manager is None:
            raise TeamConfigError("workflow.manager must be set for type=manager")
        if manager not in seen:
            raise TeamConfigError(f"workflow.manager={manager!r} is not a member")

    if workflow.type == "review_loop":
        producer = workflow.options.get("producer")
        reviewer = workflow.options.get("reviewer")
        if producer not in seen or reviewer not in seen:
            raise TeamConfigError(
                "workflow.producer and workflow.reviewer must reference members"
            )

    if workflow.type == "debate":
        pro = workflow.options.get("pro")
        con = workflow.options.get("con")
        judge = workflow.options.get("judge")
        if not pro or not con or not judge:
            raise TeamConfigError(
                "workflow type=debate requires pro, con, and judge options"
            )
        for role_name in (pro, con, judge):
            if role_name not in seen:
                raise TeamConfigError(
                    f"debate role {role_name!r} is not a member"
                )

    if workflow.type == "parallel_review":
        producer = workflow.options.get("producer")
        reviewers = workflow.options.get("reviewers") or []
        synthesizer = workflow.options.get("synthesizer", producer)
        if not producer:
            raise TeamConfigError(
                "workflow type=parallel_review requires a producer option"
            )
        if not isinstance(reviewers, list) or len(reviewers) < 2:
            raise TeamConfigError(
                "workflow type=parallel_review requires reviewers: [name1, name2, ...]"
                " with at least 2 members"
            )
        for role_name in [producer, synthesizer] + list(reviewers):
            if role_name not in seen:
                raise TeamConfigError(
                    f"parallel_review member {role_name!r} is not a declared member"
                )

    return TeamConfig(
        name=name,
        goal=goal,
        workspace=workspace,
        workflow=workflow,
        defaults=defaults,
        members=members,
        source_path=p,
        bridge=bridge,
        memory=memory,
        beliefs=beliefs,
    )


def resolve_member_setting(
    member: MemberConfig, defaults: Defaults, key: str
) -> Any:
    """Return the member-level value for ``key``, falling back to defaults.

    Member-level settings (e.g. ``temperature``) override the team-wide
    defaults.  If neither is set the function returns ``None``.
    """
    val = getattr(member, key, None)
    if val is None:
        val = getattr(defaults, key, None)
    return val
