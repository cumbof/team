"""Tests for workflow strategies, using a fake orchestrator.

We don't need real containers or LLMs to verify scheduling behaviour: each
workflow only interacts with the orchestrator via :meth:`run_turn` and the
:attr:`members` mapping.
"""

from dataclasses import dataclass
from typing import Callable

from team.config import (
    Defaults,
    MemberConfig,
    TeamConfig,
    WorkflowConfig,
)
from team.workflows import manager_driven, review_loop, round_robin


@dataclass
class FakeResult:
    content: str
    declared_done: bool = False
    files_written: list = None  # type: ignore[assignment]


class FakeOrch:
    def __init__(self, team: TeamConfig, scripts: dict[str, Callable[[int], FakeResult]]):
        self.team = team
        self.members = {m.name: type("M", (), {"name": m.name})() for m in team.members}
        self.scripts = scripts
        self.calls: list[tuple[str, str | None]] = []
        self._counts: dict[str, int] = {n: 0 for n in self.members}

    def run_turn(self, member_name: str, prompt: str | None = None) -> FakeResult:
        self.calls.append((member_name, prompt))
        i = self._counts[member_name]
        self._counts[member_name] += 1
        return self.scripts[member_name](i)


def _team(workflow: WorkflowConfig, names: list[str]) -> TeamConfig:
    members = [MemberConfig(name=n, role=n.title(), model="m", persona="x") for n in names]
    return TeamConfig(
        name="t",
        goal="g",
        workspace=__import__("pathlib").Path("/tmp/t"),
        workflow=workflow,
        defaults=Defaults(),
        members=members,
    )


# --------------------------------------------------------------------------- #
# round_robin
# --------------------------------------------------------------------------- #


def test_round_robin_full_rounds() -> None:
    team = _team(WorkflowConfig(type="round_robin", max_rounds=3), ["a", "b"])
    scripts = {n: (lambda i: FakeResult(content="ok")) for n in ["a", "b"]}
    orch = FakeOrch(team, scripts)
    round_robin(orch)
    assert [c[0] for c in orch.calls] == ["a", "b", "a", "b", "a", "b"]


def test_round_robin_early_done() -> None:
    team = _team(WorkflowConfig(type="round_robin", max_rounds=5), ["a", "b"])
    scripts = {
        "a": lambda i: FakeResult(content="ok"),
        "b": lambda i: FakeResult(content="bye", declared_done=(i == 0)),
    }
    orch = FakeOrch(team, scripts)
    round_robin(orch)
    assert [c[0] for c in orch.calls] == ["a", "b"]


# --------------------------------------------------------------------------- #
# manager
# --------------------------------------------------------------------------- #


def test_manager_routes_via_next_marker() -> None:
    team = _team(
        WorkflowConfig(type="manager", max_rounds=2, options={"manager": "lead"}),
        ["lead", "eng"],
    )
    # lead bootstraps -> eng; then eng's reply is followed by manager
    # nominating eng again, then manager declares done.
    lead_replies = [
        FakeResult("plan\nNEXT: @eng"),     # bootstrap
        FakeResult("more direction\nNEXT: @eng"),  # after eng turn 1
        FakeResult("done\n[[TEAM_DONE]]", declared_done=True),  # after eng turn 2
    ]
    eng_replies = [FakeResult("did x"), FakeResult("did y")]
    li = iter(lead_replies)
    ei = iter(eng_replies)
    scripts = {"lead": lambda i: next(li), "eng": lambda i: next(ei)}
    orch = FakeOrch(team, scripts)
    manager_driven(orch)
    speakers = [c[0] for c in orch.calls]
    # lead, eng, lead, eng, lead(done)
    assert speakers == ["lead", "eng", "lead", "eng", "lead"]


# --------------------------------------------------------------------------- #
# review_loop
# --------------------------------------------------------------------------- #


def test_review_loop_until_approved() -> None:
    team = _team(
        WorkflowConfig(
            type="review_loop",
            max_rounds=5,
            options={"producer": "p", "reviewer": "r"},
        ),
        ["p", "r"],
    )
    p_replies = [FakeResult("v1"), FakeResult("v2"), FakeResult("final", declared_done=True)]
    r_replies = [FakeResult("needs work"), FakeResult("looks good APPROVED")]
    pi = iter(p_replies)
    ri = iter(r_replies)
    scripts = {"p": lambda i: next(pi), "r": lambda i: next(ri)}
    orch = FakeOrch(team, scripts)
    review_loop(orch)
    speakers = [c[0] for c in orch.calls]
    # initial p, r(rev1)->needs work, p(revise), r(rev2)->approved, p(finalize)
    assert speakers == ["p", "r", "p", "r", "p"]


def test_review_loop_max_rounds_exhausted() -> None:
    team = _team(
        WorkflowConfig(
            type="review_loop",
            max_rounds=2,
            options={"producer": "p", "reviewer": "r"},
        ),
        ["p", "r"],
    )
    scripts = {
        "p": lambda i: FakeResult(f"draft{i}"),
        "r": lambda i: FakeResult(f"reject{i}"),
    }
    orch = FakeOrch(team, scripts)
    review_loop(orch)
    # initial p, then 2 (r, p) revision pairs
    assert [c[0] for c in orch.calls] == ["p", "r", "p", "r", "p"]
