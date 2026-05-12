"""Tests for the multi-team pipeline feature (team/pipeline.py)."""

from __future__ import annotations

import dataclasses
import json
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from team.pipeline import (
    PipelineConfig,
    PipelineError,
    PipelineRunner,
    PipelineStageConfig,
    StageResult,
    _collect_artifacts,
    _extract_summary,
    _toposort,
    load_pipeline,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _make_stage(
    stage_id: str,
    team_file: str | Path = "team.yaml",
    depends_on: list[str] | None = None,
    inject_files: bool = False,
    inject_context: bool = False,
    goal_override: str | None = None,
) -> PipelineStageConfig:
    return PipelineStageConfig(
        id=stage_id,
        team_file=Path(team_file),
        depends_on=depends_on or [],
        inject_files=inject_files,
        inject_context=inject_context,
        goal_override=goal_override,
    )


def _make_result(
    stage_id: str,
    workspace: Path,
    summary: str = "",
    artifacts: dict | None = None,
    success: bool = True,
) -> StageResult:
    return StageResult(
        stage_id=stage_id,
        success=success,
        summary=summary,
        artifacts=artifacts or {},
        workspace=workspace,
        duration_seconds=1.0,
    )


def _write_transcript(workspace: Path, turns: list[dict]) -> None:
    t_path = workspace / "transcript.jsonl"
    t_path.write_text(
        "\n".join(json.dumps(t) for t in turns) + "\n", encoding="utf-8"
    )


def _write_pipeline_yaml(path: Path, content: str) -> None:
    path.write_text(textwrap.dedent(content), encoding="utf-8")


# --------------------------------------------------------------------------- #
# _toposort
# --------------------------------------------------------------------------- #


class TestToposort:
    def test_single_stage_no_deps(self):
        stages = [_make_stage("a")]
        assert _toposort(stages) == ["a"]

    def test_linear_chain(self):
        stages = [
            _make_stage("a"),
            _make_stage("b", depends_on=["a"]),
            _make_stage("c", depends_on=["b"]),
        ]
        order = _toposort(stages)
        assert order.index("a") < order.index("b") < order.index("c")

    def test_diamond_dependency(self):
        stages = [
            _make_stage("root"),
            _make_stage("left", depends_on=["root"]),
            _make_stage("right", depends_on=["root"]),
            _make_stage("merge", depends_on=["left", "right"]),
        ]
        order = _toposort(stages)
        assert order[0] == "root"
        assert order[-1] == "merge"

    def test_independent_stages_sorted_alphabetically(self):
        stages = [_make_stage("c"), _make_stage("a"), _make_stage("b")]
        assert _toposort(stages) == ["a", "b", "c"]

    def test_unknown_dependency_raises(self):
        stages = [_make_stage("b", depends_on=["nonexistent"])]
        with pytest.raises(PipelineError, match="unknown stage 'nonexistent'"):
            _toposort(stages)

    def test_cycle_raises(self):
        stages = [
            _make_stage("a", depends_on=["b"]),
            _make_stage("b", depends_on=["a"]),
        ]
        with pytest.raises(PipelineError, match="cycle"):
            _toposort(stages)

    def test_self_dependency_cycle(self):
        stages = [_make_stage("a", depends_on=["a"])]
        with pytest.raises(PipelineError):
            _toposort(stages)

    def test_empty_list_returns_empty(self):
        assert _toposort([]) == []


# --------------------------------------------------------------------------- #
# _extract_summary
# --------------------------------------------------------------------------- #


class TestExtractSummary:
    def test_no_transcript_returns_empty(self, tmp_path):
        assert _extract_summary(tmp_path) == ""

    def test_skips_orchestrator_turns(self, tmp_path):
        _write_transcript(
            tmp_path,
            [
                {"speaker": "orchestrator", "content": "start"},
                {"speaker": "alice", "content": "hello"},
            ],
        )
        summary = _extract_summary(tmp_path)
        assert "orchestrator" not in summary
        assert "@alice" in summary

    def test_takes_last_n_turns(self, tmp_path):
        turns = [
            {"speaker": f"member{i}", "content": f"msg {i}"}
            for i in range(10)
        ]
        _write_transcript(tmp_path, turns)
        summary = _extract_summary(tmp_path, max_turns=3)
        lines = [l for l in summary.split("\n\n") if l]
        assert len(lines) == 3
        # Should contain the last 3 turns.
        assert "@member9" in summary
        assert "@member0" not in summary

    def test_truncates_long_content(self, tmp_path):
        long_content = "x" * 2000
        _write_transcript(tmp_path, [{"speaker": "alice", "content": long_content}])
        summary = _extract_summary(tmp_path)
        # Content is truncated at 800 chars in the summary.
        assert len(summary) < 2000

    def test_handles_malformed_lines(self, tmp_path):
        path = tmp_path / "transcript.jsonl"
        path.write_text('{"speaker":"alice","content":"ok"}\nNOT JSON\n', encoding="utf-8")
        summary = _extract_summary(tmp_path)
        assert "@alice" in summary

    def test_empty_transcript_returns_empty(self, tmp_path):
        (tmp_path / "transcript.jsonl").write_text("", encoding="utf-8")
        assert _extract_summary(tmp_path) == ""


# --------------------------------------------------------------------------- #
# _collect_artifacts
# --------------------------------------------------------------------------- #


class TestCollectArtifacts:
    def test_no_shared_dir_returns_empty(self, tmp_path):
        assert _collect_artifacts(tmp_path) == {}

    def test_reads_text_files(self, tmp_path):
        shared = tmp_path / "shared"
        shared.mkdir()
        (shared / "report.md").write_text("# hello", encoding="utf-8")
        artifacts = _collect_artifacts(tmp_path)
        assert artifacts == {"report.md": "# hello"}

    def test_nested_files(self, tmp_path):
        shared = tmp_path / "shared"
        (shared / "sub").mkdir(parents=True)
        (shared / "sub" / "data.csv").write_text("a,b", encoding="utf-8")
        artifacts = _collect_artifacts(tmp_path)
        assert "sub/data.csv" in artifacts

    def test_ignores_subdirectories_as_entries(self, tmp_path):
        shared = tmp_path / "shared"
        (shared / "dir").mkdir(parents=True)
        artifacts = _collect_artifacts(tmp_path)
        assert artifacts == {}


# --------------------------------------------------------------------------- #
# load_pipeline
# --------------------------------------------------------------------------- #


class TestLoadPipeline:
    def test_minimal_valid_pipeline(self, tmp_path):
        yaml_path = tmp_path / "pipeline.yaml"
        _write_pipeline_yaml(
            yaml_path,
            """\
            name: test-pipeline
            stages:
              - id: step1
                team: ./team.yaml
            """,
        )
        cfg = load_pipeline(yaml_path)
        assert cfg.name == "test-pipeline"
        assert len(cfg.stages) == 1
        assert cfg.stages[0].id == "step1"

    def test_default_workspace_uses_name(self, tmp_path):
        yaml_path = tmp_path / "pipeline.yaml"
        _write_pipeline_yaml(
            yaml_path,
            """\
            name: my-pipe
            stages:
              - id: s
                team: ./t.yaml
            """,
        )
        cfg = load_pipeline(yaml_path)
        assert cfg.workspace == (tmp_path / "runs" / "my-pipe").resolve()

    def test_explicit_workspace(self, tmp_path):
        yaml_path = tmp_path / "pipeline.yaml"
        _write_pipeline_yaml(
            yaml_path,
            f"""\
            name: p
            workspace: ./runs/custom
            stages:
              - id: s
                team: ./t.yaml
            """,
        )
        cfg = load_pipeline(yaml_path)
        assert cfg.workspace == (tmp_path / "runs" / "custom").resolve()

    def test_full_stage_config(self, tmp_path):
        yaml_path = tmp_path / "pipeline.yaml"
        _write_pipeline_yaml(
            yaml_path,
            """\
            name: full
            stages:
              - id: stage1
                team: ./a.yaml
              - id: stage2
                team: ./b.yaml
                depends_on: [stage1]
                inject_files: true
                inject_context: true
                goal_override: "Write based on {stage1.summary}"
            """,
        )
        cfg = load_pipeline(yaml_path)
        s2 = cfg.stages[1]
        assert s2.depends_on == ["stage1"]
        assert s2.inject_files is True
        assert s2.inject_context is True
        assert s2.goal_override == "Write based on {stage1.summary}"

    def test_missing_name_raises(self, tmp_path):
        yaml_path = tmp_path / "pipeline.yaml"
        _write_pipeline_yaml(yaml_path, "stages:\n  - id: s\n    team: ./t.yaml\n")
        with pytest.raises(PipelineError, match="name"):
            load_pipeline(yaml_path)

    def test_missing_stages_raises(self, tmp_path):
        yaml_path = tmp_path / "pipeline.yaml"
        _write_pipeline_yaml(yaml_path, "name: p\n")
        with pytest.raises(PipelineError, match="stages"):
            load_pipeline(yaml_path)

    def test_missing_stage_id_raises(self, tmp_path):
        yaml_path = tmp_path / "pipeline.yaml"
        _write_pipeline_yaml(
            yaml_path, "name: p\nstages:\n  - team: ./t.yaml\n"
        )
        with pytest.raises(PipelineError, match="'id'"):
            load_pipeline(yaml_path)

    def test_missing_stage_team_raises(self, tmp_path):
        yaml_path = tmp_path / "pipeline.yaml"
        _write_pipeline_yaml(yaml_path, "name: p\nstages:\n  - id: s\n")
        with pytest.raises(PipelineError, match="'team'"):
            load_pipeline(yaml_path)

    def test_duplicate_stage_id_raises(self, tmp_path):
        yaml_path = tmp_path / "pipeline.yaml"
        _write_pipeline_yaml(
            yaml_path,
            """\
            name: p
            stages:
              - id: s
                team: ./a.yaml
              - id: s
                team: ./b.yaml
            """,
        )
        with pytest.raises(PipelineError, match="Duplicate"):
            load_pipeline(yaml_path)

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(PipelineError, match="not found"):
            load_pipeline(tmp_path / "missing.yaml")

    def test_invalid_yaml_raises(self, tmp_path):
        yaml_path = tmp_path / "pipeline.yaml"
        yaml_path.write_text("name: [\n", encoding="utf-8")
        with pytest.raises(PipelineError, match="YAML parse error"):
            load_pipeline(yaml_path)

    def test_depends_on_as_string(self, tmp_path):
        """depends_on can be given as a bare string instead of a list."""
        yaml_path = tmp_path / "pipeline.yaml"
        _write_pipeline_yaml(
            yaml_path,
            """\
            name: p
            stages:
              - id: a
                team: ./a.yaml
              - id: b
                team: ./b.yaml
                depends_on: a
            """,
        )
        cfg = load_pipeline(yaml_path)
        assert cfg.stages[1].depends_on == ["a"]

    def test_team_path_resolved_relative_to_pipeline_file(self, tmp_path):
        yaml_path = tmp_path / "pipeline.yaml"
        _write_pipeline_yaml(
            yaml_path,
            "name: p\nstages:\n  - id: s\n    team: ./sub/team.yaml\n",
        )
        cfg = load_pipeline(yaml_path)
        assert cfg.stages[0].team_file == (tmp_path / "sub" / "team.yaml").resolve()

    def test_source_path_set(self, tmp_path):
        yaml_path = tmp_path / "pipeline.yaml"
        _write_pipeline_yaml(
            yaml_path,
            "name: p\nstages:\n  - id: s\n    team: ./t.yaml\n",
        )
        cfg = load_pipeline(yaml_path)
        assert cfg.source_path == yaml_path.resolve()


# --------------------------------------------------------------------------- #
# PipelineRunner
# --------------------------------------------------------------------------- #


def _make_minimal_team_yaml(tmp_path: Path, name: str = "test-team") -> Path:
    """Write a minimal team YAML that load_team() will accept."""
    p = tmp_path / f"{name}.yaml"
    p.write_text(
        textwrap.dedent(
            f"""\
            name: {name}
            goal: Do something useful.
            members:
              - name: agent
                model: llama3
                role: assistant
                persona: A helpful assistant.
            workflow:
              type: round_robin
              max_rounds: 1
            """
        ),
        encoding="utf-8",
    )
    return p


class TestPipelineRunner:
    """Tests for PipelineRunner using a mocked run_team_fn (no Docker)."""

    def _noop_run(self, team_cfg):
        """No-op team runner for tests; creates a minimal workspace layout."""
        (team_cfg.workspace / "shared").mkdir(parents=True, exist_ok=True)

    def _noop_run_with_transcript(self, turns: list[dict]):
        def runner(team_cfg):
            ws = team_cfg.workspace
            ws.mkdir(parents=True, exist_ok=True)
            (ws / "shared").mkdir(exist_ok=True)
            _write_transcript(ws, turns)
        return runner

    def _noop_run_with_artifact(self, filename: str, content: str):
        def runner(team_cfg):
            ws = team_cfg.workspace
            shared = ws / "shared"
            shared.mkdir(parents=True, exist_ok=True)
            (shared / filename).write_text(content, encoding="utf-8")
        return runner

    def test_single_stage_success(self, tmp_path):
        team_yaml = _make_minimal_team_yaml(tmp_path)
        cfg = PipelineConfig(
            name="test",
            description="",
            workspace=tmp_path / "runs",
            stages=[_make_stage("step1", team_file=team_yaml)],
        )
        runner = PipelineRunner(cfg, run_team_fn=self._noop_run)
        results = runner.run()
        assert len(results) == 1
        assert results[0].stage_id == "step1"
        assert results[0].success is True

    def test_two_stage_linear_pipeline(self, tmp_path):
        team_yaml = _make_minimal_team_yaml(tmp_path)
        execution_order = []

        def ordered_run(team_cfg):
            execution_order.append(team_cfg.workspace.name)
            team_cfg.workspace.mkdir(parents=True, exist_ok=True)
            (team_cfg.workspace / "shared").mkdir(exist_ok=True)

        cfg = PipelineConfig(
            name="test",
            description="",
            workspace=tmp_path / "runs",
            stages=[
                _make_stage("research", team_file=team_yaml),
                _make_stage("writing", team_file=team_yaml, depends_on=["research"]),
            ],
        )
        runner = PipelineRunner(cfg, run_team_fn=ordered_run)
        results = runner.run()
        assert execution_order == ["research", "writing"]
        assert len(results) == 2

    def test_stage_workspace_is_subdirectory(self, tmp_path):
        team_yaml = _make_minimal_team_yaml(tmp_path)
        seen_workspaces = []

        def record_ws(team_cfg):
            seen_workspaces.append(team_cfg.workspace)
            team_cfg.workspace.mkdir(parents=True, exist_ok=True)
            (team_cfg.workspace / "shared").mkdir(exist_ok=True)

        cfg = PipelineConfig(
            name="test",
            description="",
            workspace=tmp_path / "runs",
            stages=[_make_stage("alpha", team_file=team_yaml)],
        )
        PipelineRunner(cfg, run_team_fn=record_ws).run()
        assert seen_workspaces[0] == tmp_path / "runs" / "alpha"

    def test_goal_override_applied(self, tmp_path):
        team_yaml = _make_minimal_team_yaml(tmp_path)
        seen_goals = []

        def capture_goal(team_cfg):
            seen_goals.append(team_cfg.goal)
            team_cfg.workspace.mkdir(parents=True, exist_ok=True)
            (team_cfg.workspace / "shared").mkdir(exist_ok=True)

        cfg = PipelineConfig(
            name="test",
            description="",
            workspace=tmp_path / "runs",
            stages=[
                _make_stage("s1", team_file=team_yaml),
                _make_stage(
                    "s2",
                    team_file=team_yaml,
                    depends_on=["s1"],
                    goal_override="Custom goal.",
                ),
            ],
        )
        PipelineRunner(cfg, run_team_fn=capture_goal).run()
        assert seen_goals[1] == "Custom goal."

    def test_goal_override_with_template(self, tmp_path):
        """goal_override can reference upstream stage summary via {stage_id.summary}."""
        team_yaml = _make_minimal_team_yaml(tmp_path)
        seen_goals = []

        def runner_with_transcript(team_cfg):
            team_cfg.workspace.mkdir(parents=True, exist_ok=True)
            (team_cfg.workspace / "shared").mkdir(exist_ok=True)
            if team_cfg.workspace.name == "research":
                _write_transcript(
                    team_cfg.workspace,
                    [{"speaker": "alice", "content": "Research complete!"}],
                )
            seen_goals.append(team_cfg.goal)

        cfg = PipelineConfig(
            name="test",
            description="",
            workspace=tmp_path / "runs",
            stages=[
                _make_stage("research", team_file=team_yaml),
                _make_stage(
                    "writing",
                    team_file=team_yaml,
                    depends_on=["research"],
                    goal_override="Write paper. Context: {research.summary}",
                ),
            ],
        )
        PipelineRunner(cfg, run_team_fn=runner_with_transcript).run()
        assert "Research complete!" in seen_goals[1]

    def test_inject_files_copies_artifacts(self, tmp_path):
        team_yaml = _make_minimal_team_yaml(tmp_path)
        runs = tmp_path / "runs"

        def runner_stage1(team_cfg):
            team_cfg.workspace.mkdir(parents=True, exist_ok=True)
            shared = team_cfg.workspace / "shared"
            shared.mkdir(exist_ok=True)
            (shared / "output.txt").write_text("from stage1", encoding="utf-8")

        def runner_stage2(team_cfg):
            # Verify files were copied.
            assert (team_cfg.workspace / "shared" / "output.txt").is_file()

        stages_run = []

        def runner(team_cfg):
            if team_cfg.workspace.name == "stage1":
                runner_stage1(team_cfg)
            else:
                runner_stage2(team_cfg)
            stages_run.append(team_cfg.workspace.name)

        cfg = PipelineConfig(
            name="test",
            description="",
            workspace=runs,
            stages=[
                _make_stage("stage1", team_file=team_yaml),
                _make_stage(
                    "stage2",
                    team_file=team_yaml,
                    depends_on=["stage1"],
                    inject_files=True,
                ),
            ],
        )
        PipelineRunner(cfg, run_team_fn=runner).run()
        assert "stage2" in stages_run  # stage2 ran without assertion error

    def test_inject_context_writes_context_md(self, tmp_path):
        team_yaml = _make_minimal_team_yaml(tmp_path)
        runs = tmp_path / "runs"
        context_md_paths = []

        def runner(team_cfg):
            team_cfg.workspace.mkdir(parents=True, exist_ok=True)
            (team_cfg.workspace / "shared").mkdir(exist_ok=True)
            if team_cfg.workspace.name == "stage1":
                _write_transcript(
                    team_cfg.workspace, [{"speaker": "alice", "content": "Done!"}]
                )
            else:
                ctx = team_cfg.workspace / "context.md"
                if ctx.is_file():
                    context_md_paths.append(ctx.read_text(encoding="utf-8"))

        cfg = PipelineConfig(
            name="test",
            description="",
            workspace=runs,
            stages=[
                _make_stage("stage1", team_file=team_yaml),
                _make_stage(
                    "stage2",
                    team_file=team_yaml,
                    depends_on=["stage1"],
                    inject_context=True,
                ),
            ],
        )
        PipelineRunner(cfg, run_team_fn=runner).run()
        assert context_md_paths, "context.md was not written"
        assert "stage1" in context_md_paths[0]
        assert "Done!" in context_md_paths[0]

    def test_failed_stage_raises_pipeline_error(self, tmp_path):
        team_yaml = _make_minimal_team_yaml(tmp_path)

        def failing_run(team_cfg):
            team_cfg.workspace.mkdir(parents=True, exist_ok=True)
            raise RuntimeError("LLM exploded")

        cfg = PipelineConfig(
            name="test",
            description="",
            workspace=tmp_path / "runs",
            stages=[_make_stage("broken", team_file=team_yaml)],
        )
        with pytest.raises(PipelineError, match="broken"):
            PipelineRunner(cfg, run_team_fn=failing_run).run()

    def test_downstream_stage_not_run_after_failure(self, tmp_path):
        team_yaml = _make_minimal_team_yaml(tmp_path)
        ran = []

        def runner(team_cfg):
            ran.append(team_cfg.workspace.name)
            team_cfg.workspace.mkdir(parents=True, exist_ok=True)
            if team_cfg.workspace.name == "s1":
                raise RuntimeError("fail")

        cfg = PipelineConfig(
            name="test",
            description="",
            workspace=tmp_path / "runs",
            stages=[
                _make_stage("s1", team_file=team_yaml),
                _make_stage("s2", team_file=team_yaml, depends_on=["s1"]),
            ],
        )
        with pytest.raises(PipelineError):
            PipelineRunner(cfg, run_team_fn=runner).run()
        assert "s2" not in ran

    def test_result_artifacts_populated(self, tmp_path):
        team_yaml = _make_minimal_team_yaml(tmp_path)
        runner = self._noop_run_with_artifact("analysis.md", "# result")
        cfg = PipelineConfig(
            name="test",
            description="",
            workspace=tmp_path / "runs",
            stages=[_make_stage("step", team_file=team_yaml)],
        )
        results = PipelineRunner(cfg, run_team_fn=runner).run()
        assert "analysis.md" in results[0].artifacts

    def test_result_summary_populated(self, tmp_path):
        team_yaml = _make_minimal_team_yaml(tmp_path)
        runner = self._noop_run_with_transcript(
            [{"speaker": "alice", "content": "Final answer"}]
        )
        cfg = PipelineConfig(
            name="test",
            description="",
            workspace=tmp_path / "runs",
            stages=[_make_stage("step", team_file=team_yaml)],
        )
        results = PipelineRunner(cfg, run_team_fn=runner).run()
        assert "Final answer" in results[0].summary

    def test_stage_result_duration_positive(self, tmp_path):
        team_yaml = _make_minimal_team_yaml(tmp_path)
        cfg = PipelineConfig(
            name="test",
            description="",
            workspace=tmp_path / "runs",
            stages=[_make_stage("step", team_file=team_yaml)],
        )
        results = PipelineRunner(cfg, run_team_fn=self._noop_run).run()
        assert results[0].duration_seconds >= 0.0


# --------------------------------------------------------------------------- #
# StageResult proxy for template substitution
# --------------------------------------------------------------------------- #


class TestStageProxy:
    def test_summary_accessible(self, tmp_path):
        result = _make_result("s", tmp_path, summary="hello world")
        assert result._proxy.summary == "hello world"

    def test_template_substitution(self, tmp_path):
        result = _make_result("research", tmp_path, summary="topic X covered")
        proxy = result._proxy
        tpl = "Write based on: {research.summary}"
        rendered = tpl.format(research=proxy)
        assert "topic X covered" in rendered


# --------------------------------------------------------------------------- #
# CLI pipeline command
# --------------------------------------------------------------------------- #


class TestPipelineCLI:
    def test_dry_run_validates_and_prints_plan(self, tmp_path):
        from click.testing import CliRunner

        from team.cli import cli

        team_yaml = _make_minimal_team_yaml(tmp_path)
        pipeline_yaml = tmp_path / "pipeline.yaml"
        _write_pipeline_yaml(
            pipeline_yaml,
            f"""\
            name: my-pipe
            stages:
              - id: step
                team: {team_yaml}
            """,
        )
        runner = CliRunner()
        result = runner.invoke(cli, ["pipeline", str(pipeline_yaml), "--dry-run"])
        assert result.exit_code == 0, result.output
        assert "my-pipe" in result.output
        assert "step" in result.output

    def test_pipeline_not_found_exits_nonzero(self, tmp_path):
        from click.testing import CliRunner

        from team.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["pipeline", str(tmp_path / "no.yaml")])
        # Click validates exists=True and exits non-zero before our code runs.
        assert result.exit_code != 0

    def test_invalid_pipeline_yaml_exits_nonzero(self, tmp_path):
        from click.testing import CliRunner

        from team.cli import cli

        pipeline_yaml = tmp_path / "pipeline.yaml"
        pipeline_yaml.write_text("stages: []", encoding="utf-8")
        runner = CliRunner()
        result = runner.invoke(cli, ["pipeline", str(pipeline_yaml), "--dry-run"])
        assert result.exit_code != 0
