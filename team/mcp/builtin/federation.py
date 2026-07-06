"""``federation`` server — cross-team delegation, broadcast, peers, registry, belief sync.

delegate_task, broadcast_task, list_peers, cancel_remote_task, sync_beliefs,
query_registry.  These wrap the existing bridge/registry clients unchanged.
"""

from __future__ import annotations

import concurrent.futures
import logging
from pathlib import Path
from typing import Annotated

from pydantic import Field

from team.mcp.builtin import BuiltinContext
from team.mcp.builtin._util import offload, truncate

log = logging.getLogger(__name__)


def _read_input_files(workspace_path: Path | None, files_str: str) -> dict[str, str]:
    """Read comma-separated workspace-relative files into a name→content map."""
    input_files: dict[str, str] = {}
    if files_str and workspace_path:
        for rel in [f.strip() for f in files_str.split(",") if f.strip()]:
            try:
                target = (workspace_path / rel).resolve()
                target.relative_to(workspace_path.resolve())
                if target.is_file():
                    input_files[rel] = target.read_text(encoding="utf-8", errors="replace")
                else:
                    log.warning("delegate: file %r not found, skipping", rel)
            except (ValueError, OSError) as exc:
                log.warning("delegate: skipping %r: %s", rel, exc)
    return input_files


def _write_output_files(
    workspace_path: Path | None, files: dict[str, str] | None
) -> list[str]:
    """Write remote-returned files into the local workspace; return written paths."""
    written: list[str] = []
    if workspace_path and files:
        for rel, content in files.items():
            try:
                target = (workspace_path / rel).resolve()
                target.relative_to(workspace_path.resolve())
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
                written.append(rel)
            except (ValueError, OSError) as exc:
                log.warning("delegate: could not write %r: %s", rel, exc)
    return written


def build(ctx: BuiltinContext) -> "object":
    from mcp.server.fastmcp import FastMCP

    srv = FastMCP("federation")
    workspace_path = ctx.workspace_path
    member_name = ctx.member_name
    bridge_secret = ctx.bridge_secret
    peers = ctx.peers or {}
    registry_url = ctx.registry_url
    beliefs = ctx.beliefs
    timeout = ctx.tool_timeout

    @srv.tool()
    @offload
    def delegate_task(
        goal: Annotated[str, Field(description="What the remote team should accomplish.")],
        url: Annotated[str, Field(description="Remote bridge server URL (alternative to peer).")] = "",
        peer: Annotated[str, Field(description="Named peer from bridge.peers (alternative to url).")] = "",
        context: Annotated[str, Field(description="Optional background for the remote team.")] = "",
        files: Annotated[str, Field(description="Comma-separated relative paths of files to send.")] = "",
        task_timeout: Annotated[int, Field(description="Seconds to wait for the remote team (default 600).")] = 600,
    ) -> str:
        """Delegate a sub-task to a remote team cluster (team serve) and wait for its results."""
        from team.bridge import BridgeTask
        from team.bridge_client import BridgeClient, BridgeClientError

        target_url = url.strip()
        if not target_url:
            if peer.strip():
                target_url = peers.get(peer.strip(), "")
                if not target_url:
                    return (
                        f"ERROR: peer {peer.strip()!r} is not in the configured peers registry. "
                        f"Add it under bridge.peers in your team YAML."
                    )
            else:
                return "ERROR: provide url: <remote bridge server URL> or peer: <peer name>"
        if not goal.strip():
            return "ERROR: provide goal: <what the remote team should accomplish>"

        input_files = _read_input_files(workspace_path, files)
        task = BridgeTask(goal=goal, context=context, files=input_files, sender="local-team")
        client = BridgeClient(target_url, secret=bridge_secret)
        log.info("delegate_task: submitting to %s (goal: %.60s…)", target_url, goal)
        try:
            task_id = client.submit_task(task)
            result = client.wait_for_result(task_id, timeout=float(task_timeout))
        except BridgeClientError as exc:
            return f"ERROR: bridge communication failed: {exc}"
        if result.status == "error":
            return f"ERROR: remote team failed: {result.error}"
        written = _write_output_files(workspace_path, result.files)
        parts = [f"Remote team completed the task.\n\nSummary:\n{result.summary}"]
        if written:
            parts.append(f"\nFiles received from remote team ({len(written)}):")
            parts.extend(f"  - {p}" for p in written)
        return truncate("\n".join(parts))

    @srv.tool()
    @offload
    def broadcast_task(
        goal: Annotated[str, Field(description="What all remote teams should accomplish.")],
        peers_list: Annotated[str, Field(description="Comma-separated peer names or URLs to broadcast to.")],
        context: Annotated[str, Field(description="Optional background for the remote teams.")] = "",
        files: Annotated[str, Field(description="Comma-separated relative paths of files to send.")] = "",
        task_timeout: Annotated[int, Field(description="Seconds to wait per peer (default 600).")] = 600,
    ) -> str:
        """Submit the same goal to multiple peer teams concurrently and collect all results."""
        from team.bridge import BridgeTask
        from team.bridge_client import BridgeClient, BridgeClientError

        if not goal.strip():
            return "ERROR: provide goal: <what all remote teams should accomplish>"
        if not peers_list.strip():
            return "ERROR: provide peers: <comma-separated peer names or URLs>"

        resolved: dict[str, str] = {}
        for raw in [p.strip() for p in peers_list.split(",") if p.strip()]:
            if raw in peers:
                resolved[raw] = peers[raw]
            elif raw.startswith("http"):
                resolved[raw] = raw
            else:
                return (
                    f"ERROR: {raw!r} is not a known peer name and does not look like a URL. "
                    f"Known peers: {', '.join(peers) or '(none)'}"
                )
        if not resolved:
            return "ERROR: no valid peers to broadcast to"

        input_files = _read_input_files(workspace_path, files)

        def _submit_one(peer_name: str, purl: str):
            try:
                task = BridgeTask(goal=goal, context=context, files=input_files, sender="local-team")
                client = BridgeClient(purl, secret=bridge_secret)
                task_id = client.submit_task(task)
                result = client.wait_for_result(task_id, timeout=float(task_timeout))
                if result.status == "error":
                    return peer_name, "", None, result.error
                return peer_name, result.summary, result.files, None
            except BridgeClientError as exc:
                return peer_name, "", None, str(exc)

        result_parts = [f"Broadcast to {len(resolved)} peer(s) · goal: '{goal[:60]}'\n"]
        all_files: dict[str, str] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(resolved)) as pool:
            futures = {pool.submit(_submit_one, n, u): n for n, u in resolved.items()}
            for future in concurrent.futures.as_completed(futures):
                peer_name, summary, files_out, error = future.result()
                if error:
                    result_parts.append(f"### {peer_name}: ERROR\n{error}\n")
                else:
                    result_parts.append(f"### {peer_name}: complete\n{summary[:500]}\n")
                    if files_out:
                        for rel, content in files_out.items():
                            all_files[f"{peer_name}/{rel}"] = content

        written = _write_output_files(workspace_path, all_files)
        if written:
            result_parts.append(f"\nFiles received ({len(written)} total, namespaced by peer):")
            result_parts.extend(f"  - {p}" for p in written)
        return truncate("\n".join(result_parts))

    @srv.tool()
    @offload
    def list_peers() -> str:
        """List all configured peer teams and their live health status."""
        from team.bridge_client import BridgeClient, BridgeClientError

        if not peers:
            return "No peers configured. Add a peers: section under bridge: in your team YAML."
        lines = ["Configured peers:"]
        for name, url in peers.items():
            try:
                client = BridgeClient(url, secret=bridge_secret)
                h = client.health()
                lines.append(
                    f"  {name}: {url}  [ok · pending={h.get('pending', '?')}"
                    f" · running={h.get('running', '?')}]"
                )
            except BridgeClientError as exc:
                lines.append(f"  {name}: {url}  [unreachable: {exc}]")
        return "\n".join(lines)

    @srv.tool()
    @offload
    def cancel_remote_task(
        task_id: Annotated[str, Field(description="Task UUID to cancel (returned by delegate_task).")],
        url: Annotated[str, Field(description="Remote bridge server URL (alternative to peer).")] = "",
        peer: Annotated[str, Field(description="Named peer from bridge.peers (alternative to url).")] = "",
    ) -> str:
        """Cancel a queued or running task on a remote bridge server by task ID."""
        from team.bridge_client import BridgeClient, BridgeClientError

        target_url = url.strip()
        if not target_url:
            if peer.strip():
                target_url = peers.get(peer.strip(), "")
                if not target_url:
                    return f"ERROR: peer {peer.strip()!r} is not in the configured peers registry."
            else:
                return "ERROR: provide url: <remote bridge server URL> or peer: <peer name>"
        if not task_id.strip():
            return "ERROR: provide task_id: <task UUID to cancel>"
        try:
            client = BridgeClient(target_url, secret=bridge_secret)
            client.cancel_task(task_id.strip())
            return f"Cancelled: {task_id.strip()}"
        except BridgeClientError as exc:
            return f"ERROR: {exc}"

    @srv.tool()
    @offload
    def query_registry(
        url: Annotated[str, Field(description="Registry server URL (optional if registry.url is configured).")] = "",
        tags: Annotated[list[str], Field(description="Required capability tags (AND logic).")] = [],
        keyword: Annotated[str, Field(description="Substring searched across names, descriptions, and tags.")] = "",
        limit: Annotated[int, Field(description="Maximum number of results to return (default 10).")] = 10,
    ) -> str:
        """Query a team registry server to discover teams matching capability tags or a keyword."""
        from team.registry_client import RegistryClient, RegistryClientError

        target_url = url.strip() or registry_url
        if not target_url:
            return "ERROR: provide url: <registry URL> or configure registry.url in the team YAML"
        tag_list = [t.strip() for t in (tags or []) if t.strip()]
        kw = keyword.strip() or None
        try:
            client = RegistryClient(target_url)
            entries = client.query(tags=tag_list or None, keyword=kw)
        except RegistryClientError as exc:
            return f"ERROR: registry query failed: {exc}"
        if not entries:
            filters = []
            if tag_list:
                filters.append(f"tags={tag_list}")
            if kw:
                filters.append(f"keyword={kw!r}")
            desc = " and ".join(filters) if filters else "(no filters)"
            return f"No registered teams match {desc}."
        shown = entries[:limit]
        lines = [
            f"Found {len(entries)} matching team(s)"
            + (f" (showing {len(shown)})" if len(entries) > limit else "")
            + ":"
        ]
        for e in shown:
            tags_str = ", ".join(e.tags) if e.tags else "(none)"
            models_str = ", ".join(e.models) if e.models else "(unknown)"
            lines.append(
                f"\n**{e.name}**  \n"
                f"  URL: {e.url}  \n"
                f"  Description: {e.description or '(none)'}  \n"
                f"  Tags: {tags_str}  \n"
                f"  Models: {models_str}"
            )
        return truncate("\n".join(lines))

    @srv.tool()
    @offload
    def sync_beliefs(
        url: Annotated[str, Field(description="Bridge URL of the remote team.")] = "",
        peer: Annotated[str, Field(description="Named peer (resolved from bridge.peers config).")] = "",
        direction: Annotated[str, Field(description="Sync direction: pull | push | both (default pull).")] = "pull",
        status: Annotated[str, Field(description="Filter beliefs by status (e.g. 'accepted'). Omit for all.")] = "",
        limit: Annotated[int, Field(description="Max beliefs to transfer per direction (default 50).")] = 50,
        local_team: Annotated[str, Field(description="Name to use as source when pushing (default 'local').")] = "local",
    ) -> str:
        """Synchronize the team belief board with a remote team cluster (pull, push, or both)."""
        from team.bridge_client import BridgeClient, BridgeClientError

        raw_url = url.strip()
        if peer.strip():
            raw_url = peers.get(peer.strip()) or raw_url
        if not raw_url:
            return "ERROR: provide url: <bridge URL> or peer: <name>"
        dir_val = (direction or "pull").lower().strip()
        if dir_val not in ("pull", "push", "both"):
            dir_val = "pull"
        status_filter = status.strip() or None

        board = beliefs
        if board is None and workspace_path is not None:
            beliefs_path = workspace_path / "beliefs.json"
            if beliefs_path.exists():
                try:
                    from team.beliefs import BeliefBoard

                    board = BeliefBoard(beliefs_path, [])
                except Exception:  # noqa: BLE001
                    pass
        if board is None:
            return "ERROR: no belief board available — enable beliefs in the team YAML"

        client = BridgeClient(raw_url, secret=bridge_secret)
        lines: list[str] = []

        if dir_val in ("pull", "both"):
            try:
                bundle = client.pull_beliefs(status=status_filter, limit=limit)
            except BridgeClientError as exc:
                lines.append(f"Pull failed: {exc}")
                bundle = None
            if bundle is None:
                lines.append("Remote team does not expose a belief board.")
            else:
                from team.belief_federation import import_beliefs

                imported, skipped = import_beliefs(board, bundle)
                lines.append(
                    f"Pulled from **{bundle.source_team}**: "
                    f"{imported} belief(s) imported as pending, {skipped} skipped (already known)."
                )
                if imported:
                    lines.append(
                        "Tip: use `list_beliefs` to review the new pending beliefs "
                        "and `accept_belief` / `contest_belief` to vote on them."
                    )

        if dir_val in ("push", "both"):
            try:
                from team.belief_federation import export_beliefs

                bundle_out = export_beliefs(
                    board, local_team, raw_url, status=status_filter, limit=limit
                )
                result = client.push_beliefs(bundle_out)
            except BridgeClientError as exc:
                lines.append(f"Push failed: {exc}")
                result = None
            if result is None:
                lines.append("Remote team does not support belief import.")
            else:
                r_imported, r_skipped = result
                lines.append(
                    f"Pushed to remote ({raw_url}): "
                    f"remote imported {r_imported} belief(s), skipped {r_skipped}."
                )

        return "\n".join(lines) if lines else "No sync performed."

    return srv
