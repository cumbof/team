"""Tests for token budget (F15).

Covers:
- Config: token_budget in Defaults and MemberConfig (load_team + resolve)
- run_turn(): no budget / under / at-limit / over-limit / replay-bypass / per-member override
- run_parallel_round(): under budget / exhausted member / both exhausted / declaration-order error
- CLI: TokenBudgetError caught gracefully in the run command
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from team.member import TurnResult
from team.orchestrator import Orchestrator, TokenBudgetError


# --------------------------------------------------------------------------- #
# Helpers (shared with test_parallel_workflow pattern)
# --------------------------------------------------------------------------- #


def _make_team(
    tmp_path,
    *,
    member_names=("alpha", "beta"),
    team_token_budget=None,
    member_budgets=None,   # dict name→budget or None
):
    from team.config import Defaults, MemberConfig, TeamConfig, WorkflowConfig

    defaults = Defaults()
    if team_token_budget is not None:
        defaults.token_budget = team_token_budget

    member_budgets = member_budgets or {}
    members = [
        MemberConfig(
            name=n,
            role=n.capitalize(),
            model="m",
            persona=f"You are {n}.",
            token_budget=member_budgets.get(n),
        )
        for n in member_names
    ]
    return TeamConfig(
        name="test-budget",
        goal="budget test",
        workspace=tmp_path / "runs",
        workflow=WorkflowConfig(type="round_robin", max_rounds=3),
        defaults=defaults,
        members=members,
    )


def _make_orch(team):
    mock_cm = MagicMock()
    orch = Orchestrator(team, container_manager=mock_cm)
    orch.workspace = MagicMock()
    orch.workspace.touch = MagicMock()
    orch.checkpoints = MagicMock()
    return orch


def _mock_member(name, prompt_tokens=10, completion_tokens=5, token_budget=None):
    """Mock Member whose take_turn returns fixed token counts."""
    m = MagicMock()
    m.name = name
    m.config = MagicMock()
    m.config.name = name
    m.config.role = name.capitalize()
    m.config.turn_timeout = None
    m.config.token_budget = token_budget
    m.take_turn.return_value = TurnResult(
        content="reply",
        declared_done=False,
        files_written=[],
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    return m


def _attach(orch, *members):
    for m in members:
        orch.members[m.name] = m


def _set_used(orch, name, prompt, completion):
    """Pre-populate the token totals as if previous turns already consumed these."""
    orch._token_totals[name] = {"prompt": prompt, "completion": completion}


# --------------------------------------------------------------------------- #
# Config: token_budget field parsing
# --------------------------------------------------------------------------- #


class TestConfigParsing:
    def test_defaults_has_no_budget_by_default(self, tmp_path):
        from team.config import Defaults
        assert Defaults().token_budget is None

    def test_load_team_budget_in_defaults(self, tmp_path):
        from team.config import load_team
        yaml = (
            "name: myteam\ngoal: test\n"
            f"workspace: {tmp_path/'ws'}\n"
            "defaults:\n  token_budget: 4000\n"
            "members:\n"
            "  - name: alice\n    role: R\n    model: m\n    persona: p\n"
            "workflow:\n  type: round_robin\n  max_rounds: 1\n"
        )
        p = tmp_path / "t.yaml"
        p.write_text(yaml)
        cfg = load_team(p)
        assert cfg.defaults.token_budget == 4000

    def test_load_team_budget_per_member(self, tmp_path):
        from team.config import load_team
        yaml = (
            "name: myteam\ngoal: test\n"
            f"workspace: {tmp_path/'ws'}\n"
            "members:\n"
            "  - name: alice\n    role: R\n    model: m\n    persona: p\n"
            "    token_budget: 2000\n"
            "workflow:\n  type: round_robin\n  max_rounds: 1\n"
        )
        p = tmp_path / "t.yaml"
        p.write_text(yaml)
        cfg = load_team(p)
        assert cfg.member("alice").token_budget == 2000

    def test_member_no_budget_defaults_to_none(self, tmp_path):
        from team.config import load_team
        yaml = (
            "name: myteam\ngoal: test\n"
            f"workspace: {tmp_path/'ws'}\n"
            "members:\n"
            "  - name: alice\n    role: R\n    model: m\n    persona: p\n"
            "workflow:\n  type: round_robin\n  max_rounds: 1\n"
        )
        p = tmp_path / "t.yaml"
        p.write_text(yaml)
        cfg = load_team(p)
        assert cfg.member("alice").token_budget is None

    def test_resolve_member_budget_overrides_default(self, tmp_path):
        from team.config import Defaults, MemberConfig, resolve_member_setting
        defaults = Defaults(token_budget=5000)
        member = MemberConfig(
            name="alice", role="R", model="m", persona="p", token_budget=1000
        )
        val = resolve_member_setting(member, defaults, "token_budget")
        assert val == 1000

    def test_resolve_falls_back_to_defaults(self, tmp_path):
        from team.config import Defaults, MemberConfig, resolve_member_setting
        defaults = Defaults(token_budget=3000)
        member = MemberConfig(name="alice", role="R", model="m", persona="p")
        val = resolve_member_setting(member, defaults, "token_budget")
        assert val == 3000


# --------------------------------------------------------------------------- #
# run_turn: sequential budget checks
# --------------------------------------------------------------------------- #


class TestRunTurnBudget:
    def test_no_budget_runs_normally(self, tmp_path):
        team = _make_team(tmp_path)
        orch = _make_orch(team)
        alpha = _mock_member("alpha")
        _attach(orch, alpha)

        result = orch.run_turn("alpha")
        assert result.content == "reply"
        alpha.take_turn.assert_called_once()

    def test_under_budget_runs(self, tmp_path):
        team = _make_team(tmp_path, team_token_budget=1000)
        orch = _make_orch(team)
        alpha = _mock_member("alpha", prompt_tokens=10, completion_tokens=5)
        _attach(orch, alpha)
        _set_used(orch, "alpha", 500, 200)  # 700 used < 1000

        result = orch.run_turn("alpha")
        assert result.content == "reply"
        alpha.take_turn.assert_called_once()

    def test_exactly_at_budget_raises(self, tmp_path):
        team = _make_team(tmp_path, team_token_budget=1000)
        orch = _make_orch(team)
        alpha = _mock_member("alpha")
        _attach(orch, alpha)
        _set_used(orch, "alpha", 600, 400)  # 1000 = exactly at limit

        with pytest.raises(TokenBudgetError, match="alpha"):
            orch.run_turn("alpha")
        alpha.take_turn.assert_not_called()

    def test_over_budget_raises(self, tmp_path):
        team = _make_team(tmp_path, team_token_budget=500)
        orch = _make_orch(team)
        alpha = _mock_member("alpha")
        _attach(orch, alpha)
        _set_used(orch, "alpha", 400, 200)  # 600 > 500

        with pytest.raises(TokenBudgetError):
            orch.run_turn("alpha")
        alpha.take_turn.assert_not_called()

    def test_error_message_contains_counts(self, tmp_path):
        team = _make_team(tmp_path, team_token_budget=500)
        orch = _make_orch(team)
        alpha = _mock_member("alpha")
        _attach(orch, alpha)
        _set_used(orch, "alpha", 300, 250)  # 550 > 500

        with pytest.raises(TokenBudgetError) as exc_info:
            orch.run_turn("alpha")
        msg = str(exc_info.value)
        assert "550" in msg
        assert "500" in msg

    def test_replay_bypasses_budget_check(self, tmp_path):
        """A replayed turn must not be blocked by an exhausted budget."""
        from team.bus import Turn

        team = _make_team(tmp_path, team_token_budget=100)
        orch = _make_orch(team)
        alpha = _mock_member("alpha")
        _attach(orch, alpha)
        _set_used(orch, "alpha", 100, 0)  # at limit

        # Inject a cached turn for alpha into the replay queue.
        cached = Turn(
            index=0, speaker="alpha", role="Researcher",
            content="cached reply", files_written=[], timestamp=1.0,
        )
        orch._replay_queue = [cached]

        result = orch.run_turn("alpha")
        assert result.content == "cached reply"
        alpha.take_turn.assert_not_called()  # LLM was NOT called

    def test_per_member_budget_overrides_team_default(self, tmp_path):
        team = _make_team(
            tmp_path,
            team_token_budget=500,
            member_budgets={"alpha": 2000},
        )
        orch = _make_orch(team)
        # Mock member must carry the per-member budget so resolve_member_setting works.
        alpha = _mock_member("alpha", token_budget=2000)
        _attach(orch, alpha)
        # alpha has used 1500; team default says 500 but member says 2000 → should run
        _set_used(orch, "alpha", 1000, 500)

        result = orch.run_turn("alpha")
        assert result.content == "reply"
        alpha.take_turn.assert_called_once()

    def test_per_member_budget_lower_than_default_respected(self, tmp_path):
        team = _make_team(
            tmp_path,
            team_token_budget=10000,
            member_budgets={"alpha": 100},
        )
        orch = _make_orch(team)
        alpha = _mock_member("alpha", token_budget=100)
        _attach(orch, alpha)
        _set_used(orch, "alpha", 60, 50)  # 110 > 100 (member budget)

        with pytest.raises(TokenBudgetError):
            orch.run_turn("alpha")

    def test_token_totals_accumulated_after_live_turn(self, tmp_path):
        """Tokens from a live turn are added to _token_totals."""
        team = _make_team(tmp_path)
        orch = _make_orch(team)
        alpha = _mock_member("alpha", prompt_tokens=20, completion_tokens=8)
        _attach(orch, alpha)

        orch.run_turn("alpha")
        totals = orch._token_totals["alpha"]
        assert totals["prompt"] == 20
        assert totals["completion"] == 8

    def test_different_members_have_independent_budgets(self, tmp_path):
        team = _make_team(tmp_path, team_token_budget=200)
        orch = _make_orch(team)
        alpha = _mock_member("alpha")
        beta = _mock_member("beta")
        _attach(orch, alpha, beta)

        _set_used(orch, "alpha", 150, 100)  # 250 > 200 — exhausted
        _set_used(orch, "beta",  50,  30)  # 80 < 200 — fine

        with pytest.raises(TokenBudgetError, match="alpha"):
            orch.run_turn("alpha")

        # beta should still run fine
        result = orch.run_turn("beta")
        assert result.content == "reply"


# --------------------------------------------------------------------------- #
# run_parallel_round: budget checks before dispatch
# --------------------------------------------------------------------------- #


class TestParallelBudget:
    def _make_parallel_team(self, tmp_path, **kwargs):
        from team.config import Defaults, MemberConfig, TeamConfig, WorkflowConfig

        defaults = Defaults()
        if kwargs.get("team_token_budget"):
            defaults.token_budget = kwargs["team_token_budget"]
        member_budgets = kwargs.get("member_budgets", {})
        members = [
            MemberConfig(
                name=n, role=n.capitalize(), model="m", persona=f"You are {n}.",
                token_budget=member_budgets.get(n),
            )
            for n in ("alpha", "beta")
        ]
        return TeamConfig(
            name="par-budget",
            goal="test",
            workspace=tmp_path / "runs",
            workflow=WorkflowConfig(type="parallel", max_rounds=2),
            defaults=defaults,
            members=members,
        )

    def test_all_under_budget_both_run(self, tmp_path):
        team = self._make_parallel_team(tmp_path, team_token_budget=1000)
        orch = _make_orch(team)
        alpha = _mock_member("alpha")
        beta = _mock_member("beta")
        _attach(orch, alpha, beta)
        _set_used(orch, "alpha", 100, 50)
        _set_used(orch, "beta", 200, 80)

        results = orch.run_parallel_round(["alpha", "beta"])
        assert len(results) == 2
        alpha.take_turn.assert_called_once()
        beta.take_turn.assert_called_once()

    def test_exhausted_member_raises(self, tmp_path):
        team = self._make_parallel_team(tmp_path, team_token_budget=300)
        orch = _make_orch(team)
        alpha = _mock_member("alpha")
        beta = _mock_member("beta")
        _attach(orch, alpha, beta)
        _set_used(orch, "alpha", 200, 150)  # 350 > 300 — exhausted
        _set_used(orch, "beta", 50, 30)     # 80 < 300 — fine

        with pytest.raises(TokenBudgetError, match="alpha"):
            orch.run_parallel_round(["alpha", "beta"])
        alpha.take_turn.assert_not_called()
        beta.take_turn.assert_called_once()

    def test_both_exhausted_raises_first_in_declaration_order(self, tmp_path):
        team = self._make_parallel_team(tmp_path, team_token_budget=100)
        orch = _make_orch(team)
        alpha = _mock_member("alpha")
        beta = _mock_member("beta")
        _attach(orch, alpha, beta)
        _set_used(orch, "alpha", 60, 50)   # 110 > 100
        _set_used(orch, "beta", 70, 40)    # 110 > 100

        with pytest.raises(TokenBudgetError, match="alpha"):
            orch.run_parallel_round(["alpha", "beta"])

    def test_exhausted_member_does_not_call_llm(self, tmp_path):
        team = self._make_parallel_team(tmp_path, team_token_budget=50)
        orch = _make_orch(team)
        alpha = _mock_member("alpha")
        beta = _mock_member("beta")
        _attach(orch, alpha, beta)
        _set_used(orch, "alpha", 30, 25)  # 55 > 50 — exhausted
        # beta fine

        try:
            orch.run_parallel_round(["alpha", "beta"])
        except TokenBudgetError:
            pass
        alpha.take_turn.assert_not_called()

    def test_parallel_replay_bypasses_budget(self, tmp_path):
        """A cached turn in the replay queue skips the budget check."""
        from team.bus import Turn

        team = self._make_parallel_team(tmp_path, team_token_budget=10)
        orch = _make_orch(team)
        alpha = _mock_member("alpha")
        beta = _mock_member("beta")
        _attach(orch, alpha, beta)
        _set_used(orch, "alpha", 10, 0)   # at limit
        _set_used(orch, "beta", 10, 0)    # at limit

        # Put a cached turn for alpha at the front of the replay queue.
        cached_alpha = Turn(
            index=0, speaker="alpha", role="R",
            content="cached", files_written=[], timestamp=1.0,
        )
        orch._replay_queue = [cached_alpha]

        # alpha is replayed (bypass budget), beta is live (budget error → raises)
        with pytest.raises(TokenBudgetError, match="beta"):
            orch.run_parallel_round(["alpha", "beta"])
        alpha.take_turn.assert_not_called()


# --------------------------------------------------------------------------- #
# CLI: TokenBudgetError is caught and displayed gracefully
# --------------------------------------------------------------------------- #


def test_run_cli_catches_token_budget_error(tmp_path):
    """team run should print a warning panel and exit 0 when TokenBudgetError is raised."""
    from unittest.mock import MagicMock, patch

    from click.testing import CliRunner

    from team.cli import cli

    ws = tmp_path / "ws"
    ws.mkdir()
    yaml_content = (
        "name: myteam\ngoal: test\n"
        f"workspace: {ws}\n"
        "members:\n"
        "  - name: alice\n    role: R\n    model: m\n    persona: p\n"
        "workflow:\n  type: round_robin\n  max_rounds: 2\n"
    )
    yaml_file = tmp_path / "myteam.yaml"
    yaml_file.write_text(yaml_content)

    def _raise_budget(self):
        raise TokenBudgetError("@alice has exhausted its token budget (100 of 100 tokens used)")

    with (
        patch("team.orchestrator.ContainerManager", return_value=MagicMock()),
        patch.object(Orchestrator, "up", lambda self, **kw: None),
        patch.object(Orchestrator, "down", lambda self, **kw: None),
        patch.object(Orchestrator, "run", _raise_budget),
    ):
        runner = CliRunner()
        result = runner.invoke(cli, ["run", str(yaml_file)])

    assert result.exit_code == 0, result.output
    assert "token budget" in result.output.lower()
