from team.config import MemberConfig, TeamConfig, Defaults, WorkflowConfig
from team.personas import render_system_prompt, PROTOCOL


def _team() -> TeamConfig:
    members = [
        MemberConfig(name="pi", role="PI", model="m", persona="lead the lab"),
        MemberConfig(name="postdoc", role="PostDoc", model="m", persona="write"),
    ]
    return TeamConfig(
        name="t",
        goal="discover stuff",
        workspace=__import__("pathlib").Path("/tmp/t"),
        workflow=WorkflowConfig(),
        defaults=Defaults(),
        members=members,
    )


def test_persona_includes_role_and_teammates() -> None:
    team = _team()
    out = render_system_prompt(team, team.members[0])
    assert "@pi" in out
    assert "PI" in out
    assert "@postdoc" in out
    assert "discover stuff" in out
    # protocol is included
    assert "[[TEAM_DONE]]" in out
    assert "file:" in out


def test_extra_system_appended() -> None:
    team = _team()
    team.members[0].extra_system = "ALWAYS REPLY IN HAIKU"
    out = render_system_prompt(team, team.members[0])
    assert "ALWAYS REPLY IN HAIKU" in out


def test_protocol_mentions_private_workspace() -> None:
    assert "/private" in PROTOCOL
    assert "private workspace" in PROTOCOL.lower()
