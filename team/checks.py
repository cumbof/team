"""Preflight checks for a team run.

:func:`run_all_checks` is the main entry point.  It returns a list of
:class:`CheckResult` objects that the CLI renders as a table.

Checks that do not require Docker (workspace write access, disk space) are
always run.  Docker-dependent checks are attempted and report ``FAIL`` if
the Docker daemon is unreachable or the ``docker`` Python package is missing.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from team.config import TeamConfig


# --------------------------------------------------------------------------- #
# Result type
# --------------------------------------------------------------------------- #


class Status(Enum):
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"


@dataclass
class CheckResult:
    name: str
    status: Status
    detail: str


# --------------------------------------------------------------------------- #
# Individual checks
# --------------------------------------------------------------------------- #


def check_workspace_writable(cfg: TeamConfig) -> CheckResult:
    """Verify the workspace directory can be created and written to."""
    ws = cfg.workspace
    try:
        ws.mkdir(parents=True, exist_ok=True)
        probe = ws / ".team_check_probe"
        probe.touch()
        probe.unlink()
        return CheckResult("Workspace writable", Status.OK, str(ws))
    except OSError as exc:
        return CheckResult("Workspace writable", Status.FAIL, str(exc))


def check_disk_space(cfg: TeamConfig, min_gb: float = 5.0) -> CheckResult:
    """Warn if free disk space on the workspace filesystem falls below *min_gb* GB."""
    ws = cfg.workspace
    try:
        ws.mkdir(parents=True, exist_ok=True)
        usage = shutil.disk_usage(ws)
        free_gb = usage.free / 1_073_741_824
        if free_gb >= min_gb:
            return CheckResult(
                "Disk space", Status.OK, f"{free_gb:.1f} GB free"
            )
        return CheckResult(
            "Disk space",
            Status.WARN,
            f"only {free_gb:.1f} GB free (recommended ≥ {min_gb:.0f} GB)",
        )
    except OSError as exc:
        return CheckResult("Disk space", Status.WARN, str(exc))


def check_docker_daemon(cfg: TeamConfig) -> list[CheckResult]:
    """Check Docker daemon reachability, server version, and Ollama image."""
    results: list[CheckResult] = []
    try:
        # Lazy import: the docker package is an optional dependency.
        # We report FAIL rather than crashing at import time if it's missing.
        import docker  # type: ignore[import-untyped]
        from docker.errors import DockerException  # type: ignore[import-untyped]
    except ImportError:
        results.append(
            CheckResult(
                "Docker daemon",
                Status.FAIL,
                "docker Python package not installed (pip install docker)",
            )
        )
        return results

    try:
        client = docker.from_env()
        version_info = client.version()
        server_ver = version_info.get("Version", "unknown")
        # Require Docker ≥ 20.10
        if _version_ok(server_ver, min_major=20, min_minor=10):
            results.append(
                CheckResult("Docker daemon", Status.OK, f"v{server_ver}")
            )
        else:
            results.append(
                CheckResult(
                    "Docker daemon",
                    Status.WARN,
                    f"v{server_ver} (recommended ≥ 20.10)",
                )
            )
    except DockerException as exc:
        results.append(CheckResult("Docker daemon", Status.FAIL, str(exc)))
        return results  # no point checking image if daemon is down

    # Ollama image presence
    image = cfg.defaults.ollama_image
    try:
        client.images.get(image)
        results.append(
            CheckResult(f"Image {image!r}", Status.OK, "present locally")
        )
    except docker.errors.ImageNotFound:
        results.append(
            CheckResult(
                f"Image {image!r}",
                Status.WARN,
                "not pulled locally — will be downloaded on first run",
            )
        )

    return results


def check_gpu(cfg: TeamConfig) -> CheckResult | None:
    """If any member requests GPU access, verify nvidia-smi is available."""
    default_gpus = cfg.defaults.gpus
    member_gpus = [m.gpus for m in cfg.members if m.gpus is not None]
    all_gpus = [default_gpus] + member_gpus
    needs_gpu = any(g not in (None, "none") for g in all_gpus)
    if not needs_gpu:
        return None

    if shutil.which("nvidia-smi"):
        return CheckResult(
            "NVIDIA GPU (nvidia-smi)", Status.OK, "nvidia-smi found"
        )
    return CheckResult(
        "NVIDIA GPU (nvidia-smi)",
        Status.WARN,
        "nvidia-smi not found — GPU acceleration may not work",
    )


# --------------------------------------------------------------------------- #
# Aggregate
# --------------------------------------------------------------------------- #


def run_all_checks(cfg: TeamConfig) -> list[CheckResult]:
    """Run every preflight check for *cfg* and return results in order."""
    results: list[CheckResult] = []
    results.append(check_workspace_writable(cfg))
    results.append(check_disk_space(cfg))
    results.extend(check_docker_daemon(cfg))
    gpu = check_gpu(cfg)
    if gpu is not None:
        results.append(gpu)
    return results


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _version_ok(version_str: str, *, min_major: int, min_minor: int) -> bool:
    match = re.match(r"(\d+)\.(\d+)", version_str)
    if not match:
        # If we can't parse the version string, assume it's OK rather than
        # blocking the user on a version check we can't perform reliably.
        return True
    major, minor = int(match.group(1)), int(match.group(2))
    return (major, minor) >= (min_major, min_minor)
