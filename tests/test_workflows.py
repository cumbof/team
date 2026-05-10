"""Tests for workflow strategies, using a fake orchestrator.

We don't need real containers or LLMs to verify scheduling behaviour: each
workflow only interacts with the orchestrator via :meth:`run_turn` and the
:attr:`members` mapping.
"""

from dataclasses import dataclass
from typing import Callable

import pytest

from team.config import (
    Defaults,
    MemberConfig,
    TeamConfig,
    WorkflowConfig,
)
from team.workflows import manager_driven, review_loop, round_robin, sequential_chain, debate


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
        self._on_round_end: Callable[[int], None] | None = None

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


# --------------------------------------------------------------------------- #
# sequential_chain
# --------------------------------------------------------------------------- #


def test_sequential_chain_order_and_prompts() -> None:
    """Members are called in order; each receives the previous reply as prompt."""
    team = _team(WorkflowConfig(type="sequential_chain", max_rounds=1), ["a", "b", "c"])
    received_prompts: dict[str, list] = {"a": [], "b": [], "c": []}
    scripts = {n: (lambda i, name=n: FakeResult(f"output-{name}")) for n in ["a", "b", "c"]}

    class PromptCapture(FakeOrch):
        def run_turn(self, member_name, prompt=None):
            received_prompts[member_name].append(prompt)
            return super().run_turn(member_name, prompt=prompt)

    orch = PromptCapture(team, scripts)
    sequential_chain(orch)

    # All three members spoke once, in order.
    assert [c[0] for c in orch.calls] == ["a", "b", "c"]
    # First member gets no explicit prompt (None).
    assert received_prompts["a"] == [None]
    # Second member's prompt contains @a's output.
    assert "output-a" in received_prompts["b"][0]
    # Third member's prompt contains @b's output.
    assert "output-b" in received_prompts["c"][0]


def test_sequential_chain_wraps_across_rounds() -> None:
    """The chain from the last member of round N feeds the first of round N+1."""
    team = _team(WorkflowConfig(type="sequential_chain", max_rounds=2), ["x", "y"])
    received_prompts: dict[str, list] = {"x": [], "y": []}
    scripts = {n: (lambda i, name=n: FakeResult(f"out-{name}")) for n in ["x", "y"]}

    class PromptCapture(FakeOrch):
        def run_turn(self, member_name, prompt=None):
            received_prompts[member_name].append(prompt)
            return super().run_turn(member_name, prompt=prompt)

    orch = PromptCapture(team, scripts)
    sequential_chain(orch)

    # round1: x(None), y(uses x), round2: x(uses y), y(uses x)
    assert [c[0] for c in orch.calls] == ["x", "y", "x", "y"]
    assert received_prompts["x"][0] is None           # first turn, no prev
    assert "out-y" in received_prompts["x"][1]        # second round receives y's output


def test_sequential_chain_early_done() -> None:
    team = _team(WorkflowConfig(type="sequential_chain", max_rounds=5), ["a", "b"])
    scripts = {
        "a": lambda i: FakeResult("start"),
        "b": lambda i: FakeResult("done", declared_done=True),
    }
    orch = FakeOrch(team, scripts)
    sequential_chain(orch)
    assert [c[0] for c in orch.calls] == ["a", "b"]


def test_sequential_chain_custom_template() -> None:
    """Custom prompt_template is applied to the handoff."""
    wf = WorkflowConfig(
        type="sequential_chain",
        max_rounds=1,
        options={"prompt_template": "INPUT: {prev_content}"},
    )
    team = _team(wf, ["writer", "editor"])
    received_prompts: dict[str, list] = {"writer": [], "editor": []}
    scripts = {
        "writer": lambda i: FakeResult("my draft"),
        "editor": lambda i: FakeResult("refined"),
    }

    class PromptCapture(FakeOrch):
        def run_turn(self, member_name, prompt=None):
            received_prompts[member_name].append(prompt)
            return super().run_turn(member_name, prompt=prompt)

    orch = PromptCapture(team, scripts)
    sequential_chain(orch)
    assert received_prompts["editor"][0] == "INPUT: my draft"


# --------------------------------------------------------------------------- #
# debate (F3)
# --------------------------------------------------------------------------- #


def test_debate_full_flow() -> None:
    """Opening → max_rounds rebuttals per side → closing → judge [[TEAM_DONE]]."""
    team = _team(
        WorkflowConfig(
            type="debate",
            max_rounds=2,
            options={"pro": "pro", "con": "con", "judge": "judge"},
        ),
        ["pro", "con", "judge"],
    )
    scripts = {
        "pro": lambda i: FakeResult("pro turn"),
        "con": lambda i: FakeResult("con turn"),
        "judge": lambda i: FakeResult("verdict [[TEAM_DONE]]", declared_done=True),
    }
    orch = FakeOrch(team, scripts)
    debate(orch)

    speakers = [c[0] for c in orch.calls]
    # opening pro, opening con,
    # rebuttal round 1: pro, con
    # rebuttal round 2: pro, con
    # closing pro, closing con
    # judge
    assert speakers == ["pro", "con", "pro", "con", "pro", "con", "pro", "con", "judge"]


def test_debate_early_done_by_pro() -> None:
    """If pro declares [[TEAM_DONE]] the workflow exits immediately."""
    team = _team(
        WorkflowConfig(
            type="debate",
            max_rounds=3,
            options={"pro": "pro", "con": "con", "judge": "judge"},
        ),
        ["pro", "con", "judge"],
    )
    scripts = {
        "pro": lambda i: FakeResult("done [[TEAM_DONE]]", declared_done=(i == 0)),
        "con": lambda i: FakeResult("nope"),
        "judge": lambda i: FakeResult("verdict"),
    }
    orch = FakeOrch(team, scripts)
    debate(orch)
    # Only pro's opening turn fires before early exit
    assert [c[0] for c in orch.calls] == ["pro"]


def test_debate_config_requires_pro_con_judge(tmp_path) -> None:
    from pathlib import Path
    import textwrap
    from team.config import TeamConfigError, load_team

    p = tmp_path / "t.yaml"
    p.write_text(textwrap.dedent("""
        name: t1
        goal: g
        workflow:
          type: debate
          pro: alice
          con: bob
        members:
          - {name: alice, role: r, model: m, persona: p}
          - {name: bob, role: r, model: m, persona: p}
    """), encoding="utf-8")
    with pytest.raises(TeamConfigError, match="judge"):
        load_team(p)


def test_debate_config_validates_members(tmp_path) -> None:
    from pathlib import Path
    import textwrap
    from team.config import TeamConfigError, load_team

    p = tmp_path / "t.yaml"
    p.write_text(textwrap.dedent("""
        name: t1
        goal: g
        workflow:
          type: debate
          pro: alice
          con: bob
          judge: ghost
        members:
          - {name: alice, role: r, model: m, persona: p}
          - {name: bob, role: r, model: m, persona: p}
    """), encoding="utf-8")
    with pytest.raises(TeamConfigError, match="ghost"):
        load_team(p)

