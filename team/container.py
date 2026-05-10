"""Docker container management for Ollama-backed members.

Each member of a team is given its own dedicated Ollama container.  The
container:

* Runs the official ``ollama/ollama`` image (overridable per-member).
* Listens on its own randomly-allocated host port.
* Has a per-member named volume for the model cache (so models persist
  across ``team down``/``team up`` cycles and are *not* shared between
  members — each member has root inside its own filesystem, fully
  isolated from the others).
* Has the team's shared workspace mounted under ``/workspace`` and the
  member's private workspace under ``/private``.
* Joins a dedicated team Docker network (one per team) so members are
  isolated from anything else on the host.

The container is **not** privileged on the host, but the default user
inside the Ollama image is root, satisfying the "full root inside the
container" requirement.
"""

from __future__ import annotations

import logging
import socket
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import docker
from docker.errors import APIError, ImageNotFound, NotFound
from docker.models.containers import Container
from docker.models.networks import Network

from team.config import Defaults, MemberConfig, TeamConfig, resolve_member_setting

log = logging.getLogger(__name__)


CONTAINER_LABEL = "team.project"
MEMBER_LABEL = "team.member"


@dataclass
class MemberRuntime:
    member: MemberConfig
    container: Container | None  # None for remote/external Ollama members
    host_port: int | None
    base_url: str  # http://127.0.0.1:<port> or remote URL


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _free_port() -> int:
    # Bind to port 0 to let the OS pick a free ephemeral port, then read it
    # back and release the socket.  The chosen port may theoretically be taken
    # by another process in the brief window before Docker binds it, but in
    # practice this is safe enough for local development.
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _container_name(team: str, member: str) -> str:
    return f"team-{team}-{member}"


def _network_name(team: str) -> str:
    return f"team-{team}-net"


def _volume_name(team: str, member: str) -> str:
    return f"team-{team}-{member}-models"


def _gpu_device_requests(gpus: Any) -> list[dict] | None:
    if not gpus or gpus == "none":
        return None
    if gpus == "all":
        # Count=-1 means "all available GPUs" in the Docker device-requests API.
        return [{"Driver": "nvidia", "Count": -1, "Capabilities": [["gpu"]]}]
    if isinstance(gpus, list):
        # Individual GPU indices, e.g. [0, 1] → DeviceIDs ["0", "1"].
        return [
            {
                "Driver": "nvidia",
                "DeviceIDs": [str(i) for i in gpus],
                "Capabilities": [["gpu"]],
            }
        ]
    return None


# --------------------------------------------------------------------------- #
# Manager
# --------------------------------------------------------------------------- #


class ContainerManager:
    """Bring up / tear down the docker resources backing a team."""

    def __init__(self, team: TeamConfig, client: docker.DockerClient | None = None):
        self.team = team
        self.client = client or docker.from_env()

    # --- network -------------------------------------------------------- #

    def ensure_network(self) -> Network:
        name = _network_name(self.team.name)
        try:
            return self.client.networks.get(name)
        except NotFound:
            log.info("creating network %s", name)
            return self.client.networks.create(
                name,
                driver="bridge",
                labels={CONTAINER_LABEL: self.team.name},
            )

    # --- image / volume ------------------------------------------------- #

    def ensure_image(self, image: str) -> None:
        try:
            self.client.images.get(image)
        except ImageNotFound:
            log.info("pulling image %s", image)
            self.client.images.pull(image)

    def ensure_volume(self, name: str) -> None:
        try:
            self.client.volumes.get(name)
        except NotFound:
            self.client.volumes.create(
                name=name, labels={CONTAINER_LABEL: self.team.name}
            )

    # --- per-member container ------------------------------------------ #

    def _existing(self, member: MemberConfig) -> Container | None:
        try:
            return self.client.containers.get(_container_name(self.team.name, member.name))
        except NotFound:
            return None

    def start_member(self, member: MemberConfig) -> MemberRuntime:
        # Remote Ollama (F10): bypass all Docker management entirely.
        if member.ollama_url:
            log.info(
                "member %s uses remote Ollama at %s (skipping Docker)",
                member.name,
                member.ollama_url,
            )
            return MemberRuntime(
                member=member,
                container=None,
                host_port=None,
                base_url=member.ollama_url,
            )

        # OpenAI-compat backend (F1): also needs no container.
        effective_backend = member.backend or self.team.defaults.backend
        if effective_backend == "openai_compat":
            api_base = member.api_base or ""
            if not api_base:
                raise ValueError(
                    f"member {member.name!r}: backend=openai_compat requires api_base"
                )
            log.info(
                "member %s uses openai_compat backend at %s (skipping Docker)",
                member.name,
                api_base,
            )
            return MemberRuntime(
                member=member,
                container=None,
                host_port=None,
                base_url=api_base,
            )

        defaults = self.team.defaults
        image = defaults.ollama_image
        self.ensure_image(image)
        net = self.ensure_network()
        vol = _volume_name(self.team.name, member.name)
        self.ensure_volume(vol)

        # Create workspace directories on the host before bind-mounting them.
        # Each member gets its own private directory; all members share one
        # shared directory.
        shared = self.team.workspace / "shared"
        private = self.team.workspace / "members" / member.name
        shared.mkdir(parents=True, exist_ok=True)
        private.mkdir(parents=True, exist_ok=True)

        # Reuse a container that already exists (e.g. after `team up` without
        # `team down`).  Just restart it if stopped; no need to recreate it.
        existing = self._existing(member)
        if existing is not None:
            if existing.status != "running":
                existing.start()
            existing.reload()
            host_port = int(
                existing.attrs["NetworkSettings"]["Ports"]["11434/tcp"][0]["HostPort"]
            )
            return MemberRuntime(
                member=member,
                container=existing,
                host_port=host_port,
                base_url=f"http://127.0.0.1:{host_port}",
            )

        host_port = _free_port()
        gpus = resolve_member_setting(member, defaults, "gpus")
        device_requests = _gpu_device_requests(gpus)

        # Build the host-config dict only with keys that have values, because
        # passing mem_limit=None or nano_cpus=None to the Docker SDK causes errors.
        mem_limit = member.memory_limit or defaults.memory_limit
        cpu_limit = member.cpu_limit if member.cpu_limit is not None else defaults.cpu_limit
        host_config: dict[str, Any] = {}
        if mem_limit:
            host_config["mem_limit"] = mem_limit
        if cpu_limit:
            # Docker SDK expresses CPU quota in "nano CPUs" (1 CPU = 1_000_000_000).
            host_config["nano_cpus"] = int(float(cpu_limit) * 1_000_000_000)

        log.info(
            "starting container for member %s (model=%s, port=%d)",
            member.name, member.model, host_port,
        )
        container = self.client.containers.run(
            image=image,
            name=_container_name(self.team.name, member.name),
            detach=True,
            network=net.name,
            # Bind to 127.0.0.1 on the host so the Ollama port is NOT reachable
            # from outside the machine.
            ports={"11434/tcp": ("127.0.0.1", host_port)},
            volumes={
                # Per-member named volume for the Ollama model cache.  Keeping
                # volumes separate prevents one member's model downloads from
                # interfering with another's and allows independent purging.
                vol: {"bind": "/root/.ollama", "mode": "rw"},
                str(shared.resolve()): {"bind": "/workspace", "mode": "rw"},
                str(private.resolve()): {"bind": "/private", "mode": "rw"},
            },
            environment={
                # OLLAMA_HOST must bind on all interfaces inside the container
                # (not just localhost) so the host can reach it via the mapped port.
                "OLLAMA_HOST": "0.0.0.0:11434",
                # Convenience env vars: accessible to any scripts running inside.
                "TEAM_NAME": self.team.name,
                "TEAM_MEMBER": member.name,
                "TEAM_ROLE": member.role,
            },
            labels={
                CONTAINER_LABEL: self.team.name,
                MEMBER_LABEL: member.name,
            },
            device_requests=device_requests,
            # "unless-stopped" survives Docker daemon restarts without restarting
            # on manual `docker stop`, which is how `team down` stops containers.
            restart_policy={"Name": "unless-stopped"},
            **host_config,
        )

        return MemberRuntime(
            member=member,
            container=container,
            host_port=host_port,
            base_url=f"http://127.0.0.1:{host_port}",
        )

    def start_all(self) -> list[MemberRuntime]:
        runtimes: list[MemberRuntime] = []
        for m in self.team.members:
            runtimes.append(self.start_member(m))
        return runtimes

    # --- teardown ------------------------------------------------------- #

    def stop_member(self, member_name: str, *, remove_volume: bool = False) -> None:
        # Skip teardown for members with no Docker container.
        member = next((m for m in self.team.members if m.name == member_name), None)
        if member and (member.ollama_url or (member.backend or self.team.defaults.backend) == "openai_compat"):
            log.debug("member %s has no container, skipping teardown", member_name)
            return
        name = _container_name(self.team.name, member_name)
        try:
            c = self.client.containers.get(name)
            log.info("stopping %s", name)
            try:
                c.stop(timeout=15)
            except APIError:
                pass  # container may already be stopped; remove it anyway
            c.remove(force=True)
        except NotFound:
            pass  # container was never created or was already removed
        if remove_volume:
            try:
                self.client.volumes.get(_volume_name(self.team.name, member_name)).remove(force=True)
            except NotFound:
                pass

    def stop_all(self, *, remove_volumes: bool = False) -> None:
        for m in self.team.members:
            self.stop_member(m.name, remove_volume=remove_volumes)
        try:
            self.client.networks.get(_network_name(self.team.name)).remove()
        except (NotFound, APIError):
            pass

    def status(self) -> list[dict]:
        out: list[dict] = []
        for m in self.team.members:
            if m.ollama_url:
                out.append({
                    "member": m.name,
                    "role": m.role,
                    "model": m.model,
                    "container": "(remote)",
                    "status": "remote",
                })
                continue
            effective_backend = m.backend or self.team.defaults.backend
            if effective_backend == "openai_compat":
                out.append({
                    "member": m.name,
                    "role": m.role,
                    "model": m.model,
                    "container": "(openai_compat)",
                    "status": "external",
                })
                continue
            c = self._existing(m)
            out.append(
                {
                    "member": m.name,
                    "role": m.role,
                    "model": m.model,
                    "container": _container_name(self.team.name, m.name),
                    "status": c.status if c else "absent",
                }
            )
        return out
