"""Multi-team pipeline orchestration.

A *pipeline* chains several :class:`~team.config.TeamConfig` runs in sequence
(or parallel, when stages have no mutual dependency) so that the output of one
team — its shared workspace files and a transcript summary — is automatically
injected into the next team's context.

Pipeline YAML example::

    name: research-and-write
    description: Research a topic then write a publication-ready paper.
    workspace: ./runs/research-and-write   # optional; default is ./runs/<name>

    stages:
      - id: research
        team: ./teams/researcher.yaml

      - id: writing
        team: ./teams/writer.yaml
        depends_on: [research]
        inject_files: true          # copy upstream shared/ files to this stage
        inject_context: true        # write context.md from upstream summaries
        goal_override: |            # {stage_id.summary} template available
          Write a publication-ready paper based on the research below.

          {research.summary}
"""

from __future__ import annotations

import dataclasses
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any, Callable

import yaml

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class PipelineStageConfig:
    """Configuration for one stage of a pipeline."""

    id: str
    team_file: Path
    depends_on: list[str] = dataclasses.field(default_factory=list)
    inject_files: bool = False
    inject_context: bool = False
    goal_override: str | None = None


@dataclasses.dataclass
class PipelineConfig:
    """Parsed representation of a pipeline YAML file."""

    name: str
    description: str
    workspace: Path
    stages: list[PipelineStageConfig]
    source_path: Path | None = None


@dataclasses.dataclass
class StageResult:
    """Outcome produced by a completed pipeline stage."""

    stage_id: str
    success: bool
    summary: str
    artifacts: dict[str, str]  # relative path → text content
    workspace: Path
    duration_seconds: float
    error: str | None = None

    # Template proxy attribute used in goal_override substitution.
    @property
    def _proxy(self) -> "_StageProxy":
        return _StageProxy(self)


class PipelineError(RuntimeError):
    """Raised when a pipeline stage fails."""


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #


class _StageProxy:
    """Lightweight namespace so ``{stage_id.summary}`` works in format strings."""

    def __init__(self, result: StageResult) -> None:
        self.summary = result.summary
        self.artifacts = result.artifacts

    def __repr__(self) -> str:
        return f"<stage summary ({len(self.summary)} chars)>"


def _extract_summary(workspace: Path, max_turns: int = 5) -> str:
    """Read the transcript and return the last *max_turns* member turns as text."""
    transcript_path = workspace / "transcript.jsonl"
    if not transcript_path.is_file():
        return ""
    turns = []
    for raw in transcript_path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            t = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if t.get("speaker") not in ("orchestrator", "human"):
            turns.append(t)
    summary_turns = turns[-max_turns:]
    parts = []
    for t in summary_turns:
        speaker = t.get("speaker", "?")
        content = (t.get("content") or "").strip()[:800]
        parts.append(f"@{speaker}: {content}")
    return "\n\n".join(parts)


def _collect_artifacts(workspace: Path) -> dict[str, str]:
    """Return {relative_path: text_content} for every file in shared/."""
    shared = workspace / "shared"
    result: dict[str, str] = {}
    if not shared.is_dir():
        return result
    for p in sorted(shared.rglob("*")):
        if p.is_file():
            rel = str(p.relative_to(shared))
            try:
                result[rel] = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                pass
    return result


def _toposort(stages: list[PipelineStageConfig]) -> list[str]:
    """Return stage IDs in a valid execution order (Kahn's algorithm)."""
    ids = {s.id for s in stages}
    for s in stages:
        for dep in s.depends_on:
            if dep not in ids:
                raise PipelineError(
                    f"Stage '{s.id}' depends on unknown stage '{dep}'"
                )
    in_degree: dict[str, int] = {s.id: len(s.depends_on) for s in stages}
    dependents: dict[str, list[str]] = {s.id: [] for s in stages}
    for s in stages:
        for dep in s.depends_on:
            dependents[dep].append(s.id)

    queue = [sid for sid, deg in in_degree.items() if deg == 0]
    order: list[str] = []
    while queue:
        queue.sort()  # deterministic ordering for stages with equal depth
        node = queue.pop(0)
        order.append(node)
        for child in dependents[node]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    if len(order) != len(stages):
        cycle = sorted(sid for sid, deg in in_degree.items() if deg > 0)
        raise PipelineError(
            f"Dependency cycle detected among pipeline stages: {cycle}"
        )
    return order


# --------------------------------------------------------------------------- #
# Pipeline runner
# --------------------------------------------------------------------------- #


class PipelineRunner:
    """Executes a :class:`PipelineConfig`, running each stage in topological order.

    Parameters
    ----------
    cfg:
        The parsed pipeline configuration.
    run_team_fn:
        Injectable function ``(team_cfg) -> None`` that runs a team to
        completion.  Defaults to the real Orchestrator-based runner.
        Override in tests to avoid Docker.
    console:
        Optional Rich console for progress output.
    """

    def __init__(
        self,
        cfg: PipelineConfig,
        run_team_fn: Callable | None = None,
        console: Any = None,
    ) -> None:
        self.cfg = cfg
        self._run_team_fn = run_team_fn or _default_run_team
        self._console = console
        self._results: dict[str, StageResult] = {}

    # ------------------------------------------------------------------ #
    # Public interface
    # ------------------------------------------------------------------ #

    def run(self) -> list[StageResult]:
        """Run all stages in dependency order.

        Returns a list of :class:`StageResult` in execution order.
        Raises :class:`PipelineError` if any stage fails.
        """
        order = _toposort(self.cfg.stages)
        stage_by_id = {s.id: s for s in self.cfg.stages}

        for stage_id in order:
            stage = stage_by_id[stage_id]
            self._print(f"[pipeline] starting stage '{stage_id}' …")
            result = self._run_stage(stage)
            self._results[stage_id] = result
            if not result.success:
                raise PipelineError(
                    f"Stage '{stage_id}' failed: {result.error}"
                )
            self._print(
                f"[pipeline] stage '{stage_id}' done "
                f"({result.duration_seconds:.1f}s, "
                f"{len(result.artifacts)} artifact(s))"
            )

        return list(self._results.values())

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _run_stage(self, stage: PipelineStageConfig) -> StageResult:
        """Run a single stage and return its result."""
        from team.config import load_team

        stage_workspace = self.cfg.workspace / stage.id
        stage_workspace.mkdir(parents=True, exist_ok=True)

        # Inject upstream outputs before loading the team config so that any
        # context.md we write is present when members start rendering prompts.
        self._inject_upstream(stage, stage_workspace)

        team_cfg = load_team(stage.team_file)
        # Override workspace to the pipeline-managed directory.
        team_cfg = dataclasses.replace(team_cfg, workspace=stage_workspace)

        # Apply goal_override with template substitution.
        if stage.goal_override:
            proxies = {
                r.stage_id: r._proxy for r in self._results.values()
            }
            try:
                team_cfg = dataclasses.replace(
                    team_cfg,
                    goal=stage.goal_override.format(**proxies),
                )
            except (KeyError, AttributeError) as exc:
                raise PipelineError(
                    f"Stage '{stage.id}' goal_override template error: {exc}"
                ) from exc

        t0 = time.monotonic()
        error: str | None = None
        try:
            self._run_team_fn(team_cfg)
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
            log.exception("Stage '%s' raised an exception", stage.id)

        duration = time.monotonic() - t0
        summary = _extract_summary(stage_workspace)
        artifacts = _collect_artifacts(stage_workspace)

        return StageResult(
            stage_id=stage.id,
            success=error is None,
            summary=summary,
            artifacts=artifacts,
            workspace=stage_workspace,
            duration_seconds=duration,
            error=error,
        )

    def _inject_upstream(
        self, stage: PipelineStageConfig, stage_workspace: Path
    ) -> None:
        """Copy files and write context from every upstream stage."""
        upstream = [
            self._results[dep_id]
            for dep_id in stage.depends_on
            if dep_id in self._results
        ]
        if not upstream:
            return

        if stage.inject_files:
            shared_dst = stage_workspace / "shared"
            shared_dst.mkdir(parents=True, exist_ok=True)
            for result in upstream:
                shared_src = result.workspace / "shared"
                if shared_src.is_dir():
                    for src_file in sorted(shared_src.rglob("*")):
                        if src_file.is_file():
                            rel = src_file.relative_to(shared_src)
                            dst_file = shared_dst / rel
                            dst_file.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(src_file, dst_file)

        if stage.inject_context:
            parts: list[str] = ["# Pipeline context from previous stages\n"]
            for result in upstream:
                if result.summary:
                    parts.append(
                        f"## Stage: {result.stage_id}\n\n{result.summary}\n"
                    )
                if result.artifacts:
                    file_list = "\n".join(f"- {f}" for f in sorted(result.artifacts))
                    parts.append(
                        f"### Artifacts produced by {result.stage_id}\n\n{file_list}\n"
                    )
            ctx_path = stage_workspace / "context.md"
            ctx_path.write_text("\n".join(parts), encoding="utf-8")

    def _print(self, msg: str) -> None:
        if self._console is not None:
            self._console.print(msg)
        else:
            log.info(msg)


# --------------------------------------------------------------------------- #
# Default team runner (production)
# --------------------------------------------------------------------------- #


def _default_run_team(team_cfg: Any) -> None:
    """Run a team to completion using the real Orchestrator."""
    from team.orchestrator import Orchestrator

    orch = Orchestrator(team_cfg)
    try:
        orch.up()
        orch.run()
    finally:
        orch.down()


# --------------------------------------------------------------------------- #
# YAML loader
# --------------------------------------------------------------------------- #


def load_pipeline(path: str | Path) -> PipelineConfig:
    """Parse a pipeline YAML file into a :class:`PipelineConfig`.

    Parameters
    ----------
    path:
        Filesystem path to the ``pipeline.yaml`` file.

    Returns
    -------
    PipelineConfig
        Fully validated pipeline configuration.

    Raises
    ------
    PipelineError
        If the YAML is malformed or required fields are missing.
    """
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise PipelineError(f"Pipeline file not found: {p}")

    try:
        raw: dict = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise PipelineError(f"YAML parse error in {p}: {exc}") from exc

    if not isinstance(raw, dict):
        raise PipelineError(f"Pipeline YAML must be a mapping, got {type(raw).__name__}")

    name: str = raw.get("name") or ""
    if not name:
        raise PipelineError("Pipeline YAML must have a 'name' field")

    description: str = raw.get("description") or ""

    raw_ws = raw.get("workspace")
    if raw_ws:
        workspace = Path(raw_ws).expanduser()
        if not workspace.is_absolute():
            workspace = (p.parent / workspace).resolve()
    else:
        workspace = (p.parent / "runs" / name).resolve()

    raw_stages = raw.get("stages")
    if not raw_stages or not isinstance(raw_stages, list):
        raise PipelineError("Pipeline YAML must have a non-empty 'stages' list")

    stages: list[PipelineStageConfig] = []
    seen_ids: set[str] = set()
    for i, s in enumerate(raw_stages):
        if not isinstance(s, dict):
            raise PipelineError(f"Stage {i} must be a mapping")
        stage_id = s.get("id") or ""
        if not stage_id:
            raise PipelineError(f"Stage {i} is missing required field 'id'")
        if stage_id in seen_ids:
            raise PipelineError(f"Duplicate stage id '{stage_id}'")
        seen_ids.add(stage_id)

        team_file_raw = s.get("team") or ""
        if not team_file_raw:
            raise PipelineError(f"Stage '{stage_id}' is missing required field 'team'")
        team_file = Path(team_file_raw).expanduser()
        if not team_file.is_absolute():
            team_file = (p.parent / team_file).resolve()

        depends_on = s.get("depends_on") or []
        if isinstance(depends_on, str):
            depends_on = [depends_on]

        stages.append(
            PipelineStageConfig(
                id=stage_id,
                team_file=team_file,
                depends_on=list(depends_on),
                inject_files=bool(s.get("inject_files", False)),
                inject_context=bool(s.get("inject_context", False)),
                goal_override=s.get("goal_override") or None,
            )
        )

    return PipelineConfig(
        name=name,
        description=description,
        workspace=workspace,
        stages=stages,
        source_path=p,
    )
