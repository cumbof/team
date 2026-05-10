"""Tests for team.checks — preflight check helpers.

Only the non-Docker checks (workspace writability, disk space, version
parsing, GPU detection) are tested here; Docker-dependent checks would
require a running Docker daemon.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from team.checks import (
    CheckResult,
    Status,
    _version_ok,
    check_disk_space,
    check_gpu,
    check_workspace_writable,
)
from team.config import Defaults, MemberConfig, TeamConfig, WorkflowConfig


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _team(tmp_path: Path, *, gpus: str = "none") -> TeamConfig:
    defaults = Defaults(gpus=gpus)
    return TeamConfig(
        name="check-test",
        goal="g",
        workspace=tmp_path / "runs" / "check-test",
        workflow=WorkflowConfig(),
        defaults=defaults,
        members=[MemberConfig(name="a", role="R", model="m", persona="p")],
    )


# --------------------------------------------------------------------------- #
# _version_ok
# --------------------------------------------------------------------------- #


def test_version_ok_passes_equal() -> None:
    assert _version_ok("20.10.5", min_major=20, min_minor=10) is True


def test_version_ok_passes_higher_major() -> None:
    assert _version_ok("24.0.0", min_major=20, min_minor=10) is True


def test_version_ok_fails_older_minor() -> None:
    assert _version_ok("20.09.1", min_major=20, min_minor=10) is False


def test_version_ok_fails_older_major() -> None:
    assert _version_ok("19.03.0", min_major=20, min_minor=10) is False


def test_version_ok_unparseable_assumed_ok() -> None:
    assert _version_ok("unknown", min_major=20, min_minor=10) is True


# --------------------------------------------------------------------------- #
# check_workspace_writable
# --------------------------------------------------------------------------- #


def test_workspace_writable_ok(tmp_path: Path) -> None:
    cfg = _team(tmp_path)
    result = check_workspace_writable(cfg)
    assert result.status == Status.OK


def test_workspace_writable_fail(tmp_path: Path) -> None:
    cfg = _team(tmp_path)
    # Point workspace at a path inside a read-only file (impossible to mkdir)
    cfg.workspace = tmp_path / "file_not_dir"
    cfg.workspace.write_text("block")  # make it a file so mkdir fails
    # Replace workspace with a sub-path of the file (can't create)
    cfg.workspace = cfg.workspace / "subdir"
    result = check_workspace_writable(cfg)
    assert result.status == Status.FAIL


# --------------------------------------------------------------------------- #
# check_disk_space
# --------------------------------------------------------------------------- #


def test_disk_space_ok(tmp_path: Path) -> None:
    cfg = _team(tmp_path)
    cfg.workspace.mkdir(parents=True, exist_ok=True)
    # Use a very small minimum so this always passes on any real filesystem
    result = check_disk_space(cfg, min_gb=0.0)
    assert result.status == Status.OK
    assert "GB free" in result.detail


def test_disk_space_warn_when_below_minimum(tmp_path: Path) -> None:
    cfg = _team(tmp_path)
    cfg.workspace.mkdir(parents=True, exist_ok=True)
    # Require 999 PB — will always warn
    result = check_disk_space(cfg, min_gb=999_000_000.0)
    assert result.status == Status.WARN
    assert "only" in result.detail


# --------------------------------------------------------------------------- #
# check_gpu
# --------------------------------------------------------------------------- #


def test_gpu_check_skipped_when_gpus_none(tmp_path: Path) -> None:
    cfg = _team(tmp_path, gpus="none")
    result = check_gpu(cfg)
    assert result is None


def test_gpu_check_ok_when_nvidia_smi_found(tmp_path: Path) -> None:
    cfg = _team(tmp_path, gpus="all")
    with patch("shutil.which", return_value="/usr/bin/nvidia-smi"):
        result = check_gpu(cfg)
    assert result is not None
    assert result.status == Status.OK


def test_gpu_check_warn_when_nvidia_smi_missing(tmp_path: Path) -> None:
    cfg = _team(tmp_path, gpus="all")
    with patch("shutil.which", return_value=None):
        result = check_gpu(cfg)
    assert result is not None
    assert result.status == Status.WARN


def test_gpu_check_triggered_by_member_override(tmp_path: Path) -> None:
    """GPU check fires if any member requests GPUs, even if defaults say none."""
    cfg = _team(tmp_path, gpus="none")
    cfg.members[0].gpus = "all"
    with patch("shutil.which", return_value=None):
        result = check_gpu(cfg)
    assert result is not None
    assert result.status == Status.WARN
