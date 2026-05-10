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


@dataclass
class TeamConfig:
    name: str
    goal: str
    workspace: Path
    workflow: WorkflowConfig
    defaults: Defaults
    members: list[MemberConfig]
    source_path: Path | None = None

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
    valid_types = {"round_robin", "manager", "review_loop", "sequential_chain", "debate"}
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


def _parse_member(data: dict, defaults: Defaults) -> MemberConfig:
    ctx = f"members[{data.get('name', '?')!r}]"
    name = _require(data, "name", ctx)
    _check_name(name, ctx)
    return MemberConfig(
        name=name,
        role=_require(data, "role", ctx),
        model=_require(data, "model", ctx),
        persona=_require(data, "persona", ctx),
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

    return TeamConfig(
        name=name,
        goal=goal,
        workspace=workspace,
        workflow=workflow,
        defaults=defaults,
        members=members,
        source_path=p,
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
