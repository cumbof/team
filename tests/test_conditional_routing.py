"""Tests for conditional routing (workflow type=conditional).

Coverage:
* RouteConfig dataclass construction and config parsing
* _parse_routes() validation
* load_team() validation for conditional workflow
* _eval_routes() matching logic
* conditional_routing() full workflow runs
"""

from __future__ import annotations

import textwrap
from unittest.mock import MagicMock, patch

import pytest

from team.config import (
    RouteConfig,
    TeamConfigError,
    _parse_routes,
    load_team,
)
from team.workflows import _eval_routes, conditional_routing


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _mock_member(name: str, content: str = "ok", declared_done: bool = False) -> MagicMock:
    m = MagicMock()
    m.name = name
    m.config = MagicMock()
    m.config.name = name
    m.config.role = name.capitalize()
    m.config.turn_timeout = None
    m.config.token_budget = None
    m.take_turn.return_value = MagicMock(
        content=content,
        declared_done=declared_done,
        files_written=[],
        prompt_tokens=10,
        completion_tokens=5,
    )
    return m


def _make_route(
    next_member: str,
    if_contains: str | None = None,
    if_match: str | None = None,
    is_default: bool = False,
) -> RouteConfig:
    return RouteConfig(
        next=next_member,
        if_contains=if_contains,
        if_match=if_match,
        is_default=is_default,
    )


def _make_orch(members_dict: dict, workflow_opts: dict | None = None, max_turns: int = 20):
    """Build a minimal mock Orchestrator for workflow tests."""
    orch = MagicMock()
    orch.members = members_dict
    orch._on_round_end = None

    # Build a real-enough TeamConfig so conditional_routing can call orch.team.member()
    team_cfg = MagicMock()
    team_cfg.workflow.max_rounds = max_turns
    team_cfg.workflow.options = workflow_opts or {}

    member_configs = {}
    for name, mock_m in members_dict.items():
        mc = MagicMock()
        mc.name = name
        mc.routes = []
        member_configs[name] = mc
        mock_m.config = mc

    def _member(name):
        return member_configs[name]

    team_cfg.member.side_effect = _member
    orch.team = team_cfg

    # run_turn delegates to the mock member's take_turn result
    def _run_turn(member_name, **kwargs):
        return members_dict[member_name].take_turn()

    orch.run_turn.side_effect = _run_turn

    return orch, member_configs


# --------------------------------------------------------------------------- #
# RouteConfig and _parse_routes
# --------------------------------------------------------------------------- #


class TestParseRoutes:
    def test_empty_list_returns_empty(self):
        assert _parse_routes([], "ctx") == []

    def test_if_contains_route(self):
        routes = _parse_routes([{"if_contains": "DONE", "next": "publisher"}], "ctx")
        assert len(routes) == 1
        assert routes[0].if_contains == "DONE"
        assert routes[0].next == "publisher"
        assert not routes[0].is_default

    def test_if_match_route(self):
        routes = _parse_routes([{"if_match": "APPROVED|LGTM", "next": "deploy"}], "ctx")
        assert routes[0].if_match == "APPROVED|LGTM"
        assert routes[0].next == "deploy"

    def test_default_route(self):
        routes = _parse_routes([{"default": "fallback"}], "ctx")
        assert routes[0].is_default
        assert routes[0].next == "fallback"

    def test_multiple_routes_parsed_in_order(self):
        raw = [
            {"if_contains": "YES", "next": "a"},
            {"if_contains": "NO", "next": "b"},
            {"default": "c"},
        ]
        routes = _parse_routes(raw, "ctx")
        assert [r.next for r in routes] == ["a", "b", "c"]

    def test_missing_next_raises(self):
        with pytest.raises(TeamConfigError, match="next.*default"):
            _parse_routes([{"if_contains": "X"}], "ctx")

    def test_non_default_without_condition_raises(self):
        with pytest.raises(TeamConfigError, match="if_contains.*if_match"):
            _parse_routes([{"next": "somewhere"}], "ctx")

    def test_empty_next_raises(self):
        with pytest.raises(TeamConfigError):
            _parse_routes([{"default": ""}], "ctx")

    def test_non_dict_entry_raises(self):
        with pytest.raises(TeamConfigError, match="must be a mapping"):
            _parse_routes(["bad"], "ctx")

    def test_both_if_contains_and_if_match_allowed(self):
        routes = _parse_routes(
            [{"if_contains": "X", "if_match": "Y", "next": "z"}], "ctx"
        )
        assert routes[0].if_contains == "X"
        assert routes[0].if_match == "Y"


# --------------------------------------------------------------------------- #
# load_team validation for conditional workflow
# --------------------------------------------------------------------------- #


class TestLoadTeamConditional:
    def _write_yaml(self, tmp_path, content: str):
        p = tmp_path / "team.yaml"
        p.write_text(textwrap.dedent(content))
        return p

    def test_valid_conditional_loads(self, tmp_path):
        p = self._write_yaml(
            tmp_path,
            """
            name: myteam
            goal: test
            workflow:
              type: conditional
              start: alice
            members:
              - name: alice
                model: llama3
                persona: You are alice.
                role: Writer
                routes:
                  - if_contains: "DONE"
                    next: bob
                  - default: alice
              - name: bob
                model: llama3
                persona: You are bob.
                role: Reviewer
            """,
        )
        from team.config import load_team

        tc = load_team(p)
        assert tc.workflow.type == "conditional"
        alice = tc.member("alice")
        assert len(alice.routes) == 2
        assert alice.routes[0].if_contains == "DONE"
        assert alice.routes[0].next == "bob"
        assert alice.routes[1].is_default
        assert alice.routes[1].next == "alice"

    def test_invalid_start_raises(self, tmp_path):
        p = self._write_yaml(
            tmp_path,
            """
            name: myteam
            goal: test
            workflow:
              type: conditional
              start: nobody
            members:
              - name: alice
                model: llama3
                persona: You are alice.
                role: Writer
            """,
        )
        with pytest.raises(TeamConfigError, match="start=.*not a declared member"):
            load_team(p)

    def test_route_next_unknown_member_raises(self, tmp_path):
        p = self._write_yaml(
            tmp_path,
            """
            name: myteam
            goal: test
            workflow:
              type: conditional
            members:
              - name: alice
                model: llama3
                persona: You are alice.
                role: Writer
                routes:
                  - if_contains: "X"
                    next: phantom
            """,
        )
        with pytest.raises(TeamConfigError, match="next=.*not a declared member"):
            load_team(p)

    def test_no_routes_is_valid(self, tmp_path):
        p = self._write_yaml(
            tmp_path,
            """
            name: myteam
            goal: test
            workflow:
              type: conditional
            members:
              - name: alice
                model: llama3
                persona: You are alice.
                role: Writer
            """,
        )
        tc = load_team(p)
        assert tc.member("alice").routes == []


# --------------------------------------------------------------------------- #
# _eval_routes
# --------------------------------------------------------------------------- #


class TestEvalRoutes:
    def test_no_routes_returns_none(self):
        assert _eval_routes("anything", []) is None

    def test_if_contains_match(self):
        routes = [_make_route("dest", if_contains="APPROVED")]
        assert _eval_routes("Work APPROVED here", routes) == "dest"

    def test_if_contains_case_insensitive(self):
        routes = [_make_route("dest", if_contains="approved")]
        assert _eval_routes("Work APPROVED here", routes) == "dest"

    def test_if_contains_no_match(self):
        routes = [_make_route("dest", if_contains="APPROVED")]
        assert _eval_routes("REJECTED", routes) is None

    def test_if_match_regex(self):
        routes = [_make_route("dest", if_match=r"APPROVED|LGTM")]
        assert _eval_routes("Looks LGTM to me", routes) == "dest"

    def test_if_match_case_insensitive(self):
        routes = [_make_route("dest", if_match=r"approved")]
        assert _eval_routes("APPROVED", routes) == "dest"

    def test_if_match_no_match(self):
        routes = [_make_route("dest", if_match=r"APPROVED|LGTM")]
        assert _eval_routes("REJECTED", routes) is None

    def test_default_fires_when_nothing_matches(self):
        routes = [
            _make_route("a", if_contains="YES"),
            _make_route("fallback", is_default=True),
        ]
        assert _eval_routes("NO", routes) == "fallback"

    def test_default_does_not_fire_when_rule_matches(self):
        routes = [
            _make_route("winner", if_contains="YES"),
            _make_route("fallback", is_default=True),
        ]
        assert _eval_routes("YES", routes) == "winner"

    def test_first_matching_rule_wins(self):
        routes = [
            _make_route("first", if_contains="X"),
            _make_route("second", if_contains="X"),
        ]
        assert _eval_routes("X", routes) == "first"

    def test_only_default_returns_default(self):
        routes = [_make_route("fallback", is_default=True)]
        assert _eval_routes("anything", routes) == "fallback"

    def test_both_if_contains_and_if_match_both_evaluated(self):
        # Both conditions on the same rule: either can trigger a match.
        routes = [_make_route("dest", if_contains="YES", if_match=r"NO")]
        # if_contains fires first
        assert _eval_routes("YES", routes) == "dest"
        # if_match fires when if_contains misses
        assert _eval_routes("NO", routes) == "dest"


# --------------------------------------------------------------------------- #
# conditional_routing() workflow
# --------------------------------------------------------------------------- #


class TestConditionalRouting:
    def test_routes_to_correct_next_member(self):
        """Writer outputs APPROVED → routes to publisher."""
        writer = _mock_member("writer", content="APPROVED")
        publisher = _mock_member("publisher", content="done", declared_done=True)

        orch, mcs = _make_orch({"writer": writer, "publisher": publisher})
        mcs["writer"].routes = [
            _make_route("publisher", if_contains="APPROVED"),
        ]

        conditional_routing(orch)

        # writer ran once, publisher ran once (and declared done)
        assert writer.take_turn.call_count == 1
        assert publisher.take_turn.call_count == 1

    def test_default_route_used_when_no_match(self):
        """Writer output doesn't match any rule → default route taken."""
        writer = _mock_member("writer", content="nothing special")
        reviewer = _mock_member("reviewer", content="done", declared_done=True)

        orch, mcs = _make_orch({"writer": writer, "reviewer": reviewer})
        mcs["writer"].routes = [
            _make_route("publisher", if_contains="APPROVED"),
            _make_route("reviewer", is_default=True),
        ]

        conditional_routing(orch)

        assert writer.take_turn.call_count == 1
        assert reviewer.take_turn.call_count == 1

    def test_no_routes_falls_back_to_round_robin(self):
        """Member with no routes → round-robin next-member."""
        alice = _mock_member("alice", content="hi")
        bob = _mock_member("bob", content="bye", declared_done=True)

        orch, mcs = _make_orch({"alice": alice, "bob": bob})
        # mcs["alice"].routes is already [] from _make_orch

        conditional_routing(orch)

        assert alice.take_turn.call_count == 1
        assert bob.take_turn.call_count == 1

    def test_declared_done_stops_workflow(self):
        alice = _mock_member("alice", content="[[TEAM_DONE]]", declared_done=True)
        bob = _mock_member("bob", content="hi")

        orch, mcs = _make_orch({"alice": alice, "bob": bob})
        mcs["alice"].routes = [_make_route("bob", is_default=True)]

        conditional_routing(orch)

        assert alice.take_turn.call_count == 1
        assert bob.take_turn.call_count == 0

    def test_max_turns_stops_workflow(self):
        alice = _mock_member("alice", content="hi")

        orch, mcs = _make_orch({"alice": alice}, max_turns=3)
        mcs["alice"].routes = [_make_route("alice", is_default=True)]

        conditional_routing(orch)

        assert alice.take_turn.call_count == 3

    def test_custom_start_member(self):
        """workflow.options['start'] controls the first speaker."""
        alice = _mock_member("alice", content="hi")
        bob = _mock_member("bob", content="done", declared_done=True)

        orch, mcs = _make_orch(
            {"alice": alice, "bob": bob},
            workflow_opts={"start": "bob"},
            max_turns=5,
        )

        conditional_routing(orch)

        # bob speaks first (and declares done immediately)
        assert bob.take_turn.call_count == 1
        assert alice.take_turn.call_count == 0

    def test_on_round_end_called(self):
        """_on_round_end hook fires after each turn."""
        calls = []
        alice = _mock_member("alice", content="hi")
        bob = _mock_member("bob", content="done", declared_done=True)

        orch, mcs = _make_orch({"alice": alice, "bob": bob})
        orch._on_round_end = lambda i: calls.append(i)
        mcs["alice"].routes = [_make_route("bob", is_default=True)]

        conditional_routing(orch)

        # Called after alice's turn (turn 0), not after bob (declared done)
        assert calls == [0]

    def test_regex_route_selection(self):
        writer = _mock_member("writer", content="Status: REJECTED")
        editor = _mock_member("editor", content="done", declared_done=True)
        publisher = _mock_member("publisher", content="hi")

        orch, mcs = _make_orch({"writer": writer, "editor": editor, "publisher": publisher})
        mcs["writer"].routes = [
            _make_route("publisher", if_match=r"APPROVED|LGTM"),
            _make_route("editor", if_match=r"REJECTED|REVISION"),
            _make_route("editor", is_default=True),
        ]

        conditional_routing(orch)

        assert writer.take_turn.call_count == 1
        assert editor.take_turn.call_count == 1
        assert publisher.take_turn.call_count == 0

    def test_loop_back_to_same_member(self):
        """A default route back to self loops until max_turns."""
        alice = _mock_member("alice", content="working...")
        orch, mcs = _make_orch({"alice": alice}, max_turns=4)
        mcs["alice"].routes = [_make_route("alice", is_default=True)]

        conditional_routing(orch)

        assert alice.take_turn.call_count == 4

    def test_multi_hop_routing(self):
        """A → B → C → done."""
        a = _mock_member("a", content="pass to b")
        b = _mock_member("b", content="pass to c")
        c = _mock_member("c", content="fin", declared_done=True)

        orch, mcs = _make_orch({"a": a, "b": b, "c": c})
        mcs["a"].routes = [_make_route("b", if_contains="pass to b")]
        mcs["b"].routes = [_make_route("c", if_contains="pass to c")]

        conditional_routing(orch)

        assert a.take_turn.call_count == 1
        assert b.take_turn.call_count == 1
        assert c.take_turn.call_count == 1
