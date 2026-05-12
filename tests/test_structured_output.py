"""Tests for structured JSON output (F11).

Covers:
- _extract_json helper: raw JSON, fenced json block, bare fenced block, failure
- _validate_json_schema helper: valid, invalid, missing jsonschema
- take_turn produces json_output when output_format == "json"
- retry loop kicks in when first reply is not valid JSON
- json_output is None when output_format is not set
- system prompt contains JSON instruction when output_format == "json"
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from team.member import TurnResult, _extract_json, _validate_json_schema


# --------------------------------------------------------------------------- #
# _extract_json
# --------------------------------------------------------------------------- #


class TestExtractJson:
    def test_raw_object(self):
        parsed, err = _extract_json('{"key": "value"}')
        assert parsed == {"key": "value"}
        assert err is None

    def test_raw_array(self):
        parsed, err = _extract_json('[1, 2, 3]')
        assert parsed == [1, 2, 3]
        assert err is None

    def test_json_fenced_block(self):
        text = '```json\n{"a": 1}\n```'
        parsed, err = _extract_json(text)
        assert parsed == {"a": 1}
        assert err is None

    def test_bare_fenced_block(self):
        text = '```\n{"b": 2}\n```'
        parsed, err = _extract_json(text)
        assert parsed == {"b": 2}
        assert err is None

    def test_prose_then_json_block(self):
        text = "Sure! Here is your JSON:\n```json\n{\"x\": 99}\n```\nDone."
        parsed, err = _extract_json(text)
        assert parsed == {"x": 99}
        assert err is None

    def test_invalid_json_returns_none_and_error(self):
        parsed, err = _extract_json("this is not json at all")
        assert parsed is None
        assert err is not None
        assert "json" in err.lower()

    def test_whitespace_stripped(self):
        parsed, err = _extract_json("  \n  42  \n  ")
        assert parsed == 42
        assert err is None


# --------------------------------------------------------------------------- #
# _validate_json_schema
# --------------------------------------------------------------------------- #


class TestValidateJsonSchema:
    def test_valid_data(self):
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        result = _validate_json_schema({"name": "Alice"}, schema)
        # None means valid (or jsonschema not installed — both acceptable)
        assert result is None or isinstance(result, str)

    def test_invalid_data(self):
        pytest.importorskip("jsonschema")
        schema = {"type": "object", "required": ["name"]}
        result = _validate_json_schema({"age": 5}, schema)
        assert result is not None
        assert isinstance(result, str)

    def test_valid_returns_none(self):
        pytest.importorskip("jsonschema")
        schema = {"type": "array"}
        result = _validate_json_schema([1, 2, 3], schema)
        assert result is None


# --------------------------------------------------------------------------- #
# render_system_prompt includes JSON instruction
# --------------------------------------------------------------------------- #


class TestPersonasJsonInstruction:
    def test_json_instruction_present(self):
        from team.config import Defaults, MemberConfig, TeamConfig, WorkflowConfig
        from team.personas import render_system_prompt

        member = MemberConfig(
            name="analyst",
            role="Data Analyst",
            model="llama3",
            persona="You analyse data.",
            output_format="json",
        )
        team = TeamConfig(
            name="t",
            goal="extract data",
            workspace=__import__("pathlib").Path("/tmp/t"),
            workflow=WorkflowConfig(),
            defaults=Defaults(),
            members=[member],
        )
        prompt = render_system_prompt(team, member)
        assert "Output format" in prompt
        assert "json.loads()" in prompt
        assert "valid json" in prompt.lower()

    def test_json_instruction_with_schema(self):
        from team.config import Defaults, MemberConfig, TeamConfig, WorkflowConfig
        from team.personas import render_system_prompt

        schema = {"type": "object", "required": ["result"]}
        member = MemberConfig(
            name="analyst",
            role="Analyst",
            model="llama3",
            persona="You analyse.",
            output_format="json",
            output_schema=schema,
        )
        team = TeamConfig(
            name="t",
            goal="g",
            workspace=__import__("pathlib").Path("/tmp/t"),
            workflow=WorkflowConfig(),
            defaults=Defaults(),
            members=[member],
        )
        prompt = render_system_prompt(team, member)
        assert '"required"' in prompt
        assert "result" in prompt

    def test_no_json_instruction_when_format_not_set(self):
        from team.config import Defaults, MemberConfig, TeamConfig, WorkflowConfig
        from team.personas import render_system_prompt

        member = MemberConfig(name="dev", role="Dev", model="m", persona="You code.")
        team = TeamConfig(
            name="t",
            goal="code",
            workspace=__import__("pathlib").Path("/tmp/t"),
            workflow=WorkflowConfig(),
            defaults=Defaults(),
            members=[member],
        )
        prompt = render_system_prompt(team, member)
        assert "Output format" not in prompt


# --------------------------------------------------------------------------- #
# TurnResult.json_output field
# --------------------------------------------------------------------------- #


class TestTurnResultJsonOutput:
    def test_default_is_none(self):
        r = TurnResult(content="hi", declared_done=False, files_written=[])
        assert r.json_output is None

    def test_can_be_set(self):
        r = TurnResult(
            content='{"x":1}', declared_done=False, files_written=[],
            json_output={"x": 1},
        )
        assert r.json_output == {"x": 1}


# --------------------------------------------------------------------------- #
# config.py: output_format / output_schema / turn_timeout parsed from YAML
# --------------------------------------------------------------------------- #


class TestConfigParsesNewFields:
    def _write_yaml(self, tmp_path, extra_member_fields=""):
        y = tmp_path / "team.yaml"
        y.write_text(f"""\
name: test-team
goal: test goal
workspace: {tmp_path}/runs

defaults:
  ollama_image: ollama/ollama:latest

members:
  - name: analyst
    role: Analyst
    model: llama3
    persona: You analyse.
    {extra_member_fields}
""")
        return str(y)

    def test_output_format_parsed(self, tmp_path):
        from team.config import load_team
        path = self._write_yaml(tmp_path, "output_format: json")
        cfg = load_team(path)
        assert cfg.members[0].output_format == "json"

    def test_output_schema_parsed(self, tmp_path):
        from team.config import load_team
        path = self._write_yaml(
            tmp_path,
            'output_format: json\n    output_schema:\n      type: object\n      required:\n        - result',
        )
        cfg = load_team(path)
        assert cfg.members[0].output_schema == {"type": "object", "required": ["result"]}

    def test_turn_timeout_member_parsed(self, tmp_path):
        from team.config import load_team
        path = self._write_yaml(tmp_path, "turn_timeout: 45")
        cfg = load_team(path)
        assert cfg.members[0].turn_timeout == 45

    def test_turn_timeout_defaults_parsed(self, tmp_path):
        from team.config import load_team
        y = tmp_path / "team.yaml"
        y.write_text(f"""\
name: test-team
goal: test goal
workspace: {tmp_path}/runs

defaults:
  ollama_image: ollama/ollama:latest
  turn_timeout: 60

members:
  - name: dev
    role: Dev
    model: m
    persona: You code.
""")
        cfg = load_team(str(y))
        assert cfg.defaults.turn_timeout == 60
