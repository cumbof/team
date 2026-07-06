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


def _ti(server, name, desc="A tool.", schema=None):
    from team.mcp.bus import ToolInfo

    return ToolInfo(
        server=server,
        name=name,
        wire_name=f"{server}__{name}",
        description=desc,
        input_schema=schema or {"type": "object", "properties": {}, "required": []},
    )


def test_tool_section_included_when_tools_provided() -> None:
    team = _team()
    tools = [_ti("web", "web_search"), _ti("code", "run_python")]
    out = render_system_prompt(team, team.members[0], tools=tools)
    assert "Tool use" in out
    assert "web__web_search" in out
    assert "code__run_python" in out


def test_tool_section_absent_in_native_mode() -> None:
    # Native mode gets schemas via the function-calling API; the text-mode
    # fenced-block protocol must NOT be injected (it would mislead the model).
    team = _team()
    tools = [_ti("web", "web_search"), _ti("code", "run_python")]
    out = render_system_prompt(team, team.members[0], tools=tools, tool_mode="native")
    assert "Tool use" not in out
    assert "```tool:" not in out


def test_tool_section_absent_when_no_tools() -> None:
    team = _team()
    out = render_system_prompt(team, team.members[0], tools=[])
    assert "Tool use" not in out


def test_tool_section_absent_when_tools_none() -> None:
    team = _team()
    out = render_system_prompt(team, team.members[0], tools=None)
    assert "Tool use" not in out
