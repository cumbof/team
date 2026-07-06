"""``code`` server — run_python, run_bash (host code execution)."""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Annotated

from pydantic import Field

from team.mcp.builtin import BuiltinContext
from team.mcp.builtin._util import offload, truncate

log = logging.getLogger(__name__)

#: Valid sandbox identifiers for run_python / run_bash.
VALID_TOOL_SANDBOXES = {"none", "firejail", "bubblewrap"}


def _sandbox_available(name: str) -> bool:
    """Return True if *name* (e.g. ``"firejail"``, ``"bwrap"``) is on PATH."""
    import shutil

    return shutil.which(name) is not None


def build_sandboxed_cmd(
    cmd: list[str], workspace_path: Path | None, sandbox: str
) -> list[str]:
    """Wrap *cmd* with the requested sandbox tool (no-op for ``"none"``).

    Falls back to *cmd* unchanged (with a warning) when the sandbox binary is
    not installed, so the tool still runs — just unsandboxed.
    """
    if sandbox == "none":
        return cmd

    ws = str(workspace_path) if workspace_path else None

    if sandbox == "firejail":
        if not _sandbox_available("firejail"):
            log.warning(
                "tool_sandbox=firejail requested but 'firejail' is not on PATH; "
                "running without sandbox. Install firejail to enable isolation."
            )
            return cmd
        args = ["firejail", "--quiet", "--noprofile", "--private-tmp", "--noroot"]
        if ws:
            args.append(f"--whitelist={ws}")
        args.append("--")
        return args + cmd

    if sandbox == "bubblewrap":
        bwrap = "bwrap"
        if not _sandbox_available(bwrap):
            log.warning(
                "tool_sandbox=bubblewrap requested but 'bwrap' is not on PATH; "
                "running without sandbox. Install bubblewrap to enable isolation."
            )
            return cmd
        args = [bwrap]
        for sys_dir in ("/usr", "/bin", "/sbin", "/lib", "/lib32", "/lib64", "/etc", "/run"):
            p = Path(sys_dir)
            if p.exists():
                args += ["--ro-bind", sys_dir, sys_dir]
        args += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]
        args += ["--tmpfs", "/home", "--tmpfs", "/root"]
        if ws:
            args += ["--bind", ws, ws]
        args += ["--chdir", ws or "/tmp", "--die-with-parent", "--"]
        return args + cmd

    log.warning("Unknown tool_sandbox value %r; running without sandbox.", sandbox)
    return cmd


def build(ctx: BuiltinContext) -> "object":
    from mcp.server.fastmcp import FastMCP

    srv = FastMCP("code")
    workspace_path = ctx.workspace_path
    sandbox = ctx.sandbox
    timeout = ctx.tool_timeout

    @srv.tool()
    @offload
    def run_python(
        code: Annotated[str, Field(description="Python source code to execute.")],
    ) -> str:
        """Execute Python code in the shared workspace directory and return stdout/stderr."""
        src = code.strip()
        env = os.environ.copy()
        cwd = str(workspace_path) if workspace_path and workspace_path.is_dir() else None
        if cwd:
            env["WORKSPACE"] = cwd
        script_dir = (
            workspace_path
            if sandbox != "none" and workspace_path and workspace_path.is_dir()
            else None
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8", dir=script_dir
        ) as fh:
            fh.write(src)
            script = fh.name
        try:
            cmd = build_sandboxed_cmd(["python", script], workspace_path, sandbox)
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout, env=env, cwd=cwd
            )
            out = result.stdout
            if result.stderr:
                out += ("\n" if out else "") + f"STDERR:\n{result.stderr}"
            return truncate(out or "(no output)")
        except subprocess.TimeoutExpired:
            return f"ERROR: execution timed out after {timeout}s"
        except Exception as exc:  # noqa: BLE001
            return f"ERROR: {exc}"
        finally:
            try:
                os.unlink(script)
            except OSError:
                pass

    @srv.tool()
    @offload
    def run_bash(
        command: Annotated[str, Field(description="Bash command to execute.")],
    ) -> str:
        """Execute a bash command in the shared workspace directory and return stdout/stderr."""
        cmd_str = command.strip()
        cwd = str(workspace_path) if workspace_path and workspace_path.is_dir() else None
        try:
            cmd = build_sandboxed_cmd(["bash", "-c", cmd_str], workspace_path, sandbox)
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd
            )
            out = result.stdout
            if result.stderr:
                out += ("\n" if out else "") + f"STDERR:\n{result.stderr}"
            return truncate(out or "(no output)")
        except subprocess.TimeoutExpired:
            return f"ERROR: execution timed out after {timeout}s"
        except Exception as exc:  # noqa: BLE001
            return f"ERROR: {exc}"

    return srv
