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
    container: Container
    host_port: int
    base_url: str  # http://127.0.0.1:<port>


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _free_port() -> int:
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
        return [{"Driver": "nvidia", "Count": -1, "Capabilities": [["gpu"]]}]
    if isinstance(gpus, list):
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
        defaults = self.team.defaults
        image = defaults.ollama_image
        self.ensure_image(image)
        net = self.ensure_network()
        vol = _volume_name(self.team.name, member.name)
        self.ensure_volume(vol)

        # Workspace bind mounts ----------------------------------------- #
        shared = self.team.workspace / "shared"
        private = self.team.workspace / "members" / member.name
        shared.mkdir(parents=True, exist_ok=True)
        private.mkdir(parents=True, exist_ok=True)

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

        mem_limit = member.memory_limit or defaults.memory_limit
        cpu_limit = member.cpu_limit if member.cpu_limit is not None else defaults.cpu_limit
        host_config: dict[str, Any] = {}
        if mem_limit:
            host_config["mem_limit"] = mem_limit
        if cpu_limit:
            # Docker SDK uses nano_cpus (1e9 == 1 CPU)
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
            ports={"11434/tcp": ("127.0.0.1", host_port)},
            volumes={
                vol: {"bind": "/root/.ollama", "mode": "rw"},
                str(shared.resolve()): {"bind": "/workspace", "mode": "rw"},
                str(private.resolve()): {"bind": "/private", "mode": "rw"},
            },
            environment={
                "OLLAMA_HOST": "0.0.0.0:11434",
                "TEAM_NAME": self.team.name,
                "TEAM_MEMBER": member.name,
                "TEAM_ROLE": member.role,
            },
            labels={
                CONTAINER_LABEL: self.team.name,
                MEMBER_LABEL: member.name,
            },
            device_requests=device_requests,
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
        name = _container_name(self.team.name, member_name)
        try:
            c = self.client.containers.get(name)
            log.info("stopping %s", name)
            try:
                c.stop(timeout=15)
            except APIError:
                pass
            c.remove(force=True)
        except NotFound:
            pass
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
