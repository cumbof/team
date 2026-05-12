"""Tests for the LLM retry / backoff feature.

Coverage:
* LLMRetryExhaustedError is a subclass of OllamaError
* Per-member config parsing (max_retries, retry_backoff)
* resolve_member_setting wires per-member values to the client
* CLI 'run' command catches LLMRetryExhaustedError and shows a red panel
* CLI 'test' command catches LLMRetryExhaustedError
* OpenAICompatClient also raises LLMRetryExhaustedError (not OllamaError)
"""

from __future__ import annotations

import textwrap
from unittest.mock import MagicMock, patch

import pytest
import requests

from team.ollama_client import (
    ChatMessage,
    LLMRetryExhaustedError,
    OllamaClient,
    OllamaError,
    OpenAICompatClient,
)


# --------------------------------------------------------------------------- #
# LLMRetryExhaustedError hierarchy
# --------------------------------------------------------------------------- #


def test_retry_exhausted_is_subclass_of_ollama_error():
    assert issubclass(LLMRetryExhaustedError, OllamaError)


def test_retry_exhausted_is_subclass_of_runtime_error():
    assert issubclass(LLMRetryExhaustedError, RuntimeError)


def test_retry_exhausted_caught_as_ollama_error():
    """Existing except OllamaError blocks still catch it."""
    with pytest.raises(OllamaError):
        raise LLMRetryExhaustedError("boom")


def test_retry_exhausted_caught_specifically():
    with pytest.raises(LLMRetryExhaustedError):
        raise LLMRetryExhaustedError("boom")


# --------------------------------------------------------------------------- #
# OllamaClient raises LLMRetryExhaustedError (not plain OllamaError)
# --------------------------------------------------------------------------- #


def _chat_response(content: str, *, status: int = 200) -> MagicMock:
    m = MagicMock()
    m.status_code = status
    m.text = "err" if status >= 400 else ""
    import json
    m.json.return_value = {
        "message": {"content": content},
        "prompt_eval_count": 5,
        "eval_count": 3,
    }
    return m


def test_ollama_client_chat_exhausted_raises_llm_retry_error():
    client = OllamaClient("http://localhost:11434", max_retries=1, retry_backoff=0.001)
    fail = requests.ConnectionError("refused")
    with patch.object(client._session, "post", side_effect=[fail, fail]):
        with patch("team.ollama_client.time.sleep"):
            with pytest.raises(LLMRetryExhaustedError):
                client.chat("m", [ChatMessage("user", "hi")])


def test_ollama_client_stream_chat_exhausted_raises_llm_retry_error():
    client = OllamaClient("http://localhost:11434", max_retries=1, retry_backoff=0.001)
    fail = requests.ConnectionError("refused")
    with patch.object(client._session, "post", side_effect=[fail, fail]):
        with patch("team.ollama_client.time.sleep"):
            with pytest.raises(LLMRetryExhaustedError):
                list(client.stream_chat("m", [ChatMessage("user", "hi")]))


def test_openai_compat_chat_exhausted_raises_llm_retry_error():
    client = OpenAICompatClient("http://localhost:8000", max_retries=1, retry_backoff=0.001)
    fail = requests.ConnectionError("refused")
    with patch.object(client._session, "post", side_effect=[fail, fail]):
        with patch("team.ollama_client.time.sleep"):
            with pytest.raises(LLMRetryExhaustedError):
                client.chat("m", [ChatMessage("user", "hi")])


def test_openai_compat_stream_chat_exhausted_raises_llm_retry_error():
    client = OpenAICompatClient("http://localhost:8000", max_retries=1, retry_backoff=0.001)
    fail = requests.ConnectionError("refused")
    with patch.object(client._session, "post", side_effect=[fail, fail]):
        with patch("team.ollama_client.time.sleep"):
            with pytest.raises(LLMRetryExhaustedError):
                list(client.stream_chat("m", [ChatMessage("user", "hi")]))


# --------------------------------------------------------------------------- #
# Config parsing
# --------------------------------------------------------------------------- #


def _write_yaml(tmp_path, content: str):
    p = tmp_path / "team.yaml"
    p.write_text(textwrap.dedent(content))
    return p


class TestConfigParsing:
    def test_defaults_max_retries_loaded(self, tmp_path):
        from team.config import load_team
        p = _write_yaml(tmp_path, """
            name: t
            goal: g
            defaults:
              max_retries: 7
            members:
              - name: alice
                model: m
                persona: hi
                role: R
        """)
        tc = load_team(p)
        assert tc.defaults.max_retries == 7

    def test_defaults_retry_backoff_loaded(self, tmp_path):
        from team.config import load_team
        p = _write_yaml(tmp_path, """
            name: t
            goal: g
            defaults:
              retry_backoff: 5.0
            members:
              - name: alice
                model: m
                persona: hi
                role: R
        """)
        tc = load_team(p)
        assert tc.defaults.retry_backoff == 5.0

    def test_per_member_max_retries(self, tmp_path):
        from team.config import load_team
        p = _write_yaml(tmp_path, """
            name: t
            goal: g
            members:
              - name: alice
                model: m
                persona: hi
                role: R
                max_retries: 1
        """)
        tc = load_team(p)
        assert tc.member("alice").max_retries == 1

    def test_per_member_retry_backoff(self, tmp_path):
        from team.config import load_team
        p = _write_yaml(tmp_path, """
            name: t
            goal: g
            members:
              - name: alice
                model: m
                persona: hi
                role: R
                retry_backoff: 0.5
        """)
        tc = load_team(p)
        assert tc.member("alice").retry_backoff == 0.5

    def test_member_none_defaults(self, tmp_path):
        """max_retries/retry_backoff are None in MemberConfig when not set."""
        from team.config import load_team
        p = _write_yaml(tmp_path, """
            name: t
            goal: g
            members:
              - name: alice
                model: m
                persona: hi
                role: R
        """)
        tc = load_team(p)
        assert tc.member("alice").max_retries is None
        assert tc.member("alice").retry_backoff is None


# --------------------------------------------------------------------------- #
# resolve_member_setting wires per-member values
# --------------------------------------------------------------------------- #


class TestResolveRetrySettings:
    def _make_defaults(self, max_retries=3, retry_backoff=2.0):
        from team.config import Defaults
        d = Defaults()
        d.max_retries = max_retries
        d.retry_backoff = retry_backoff
        return d

    def test_member_override_wins_over_default(self):
        from team.config import resolve_member_setting, MemberConfig
        defaults = self._make_defaults(max_retries=3)
        mc = MemberConfig(name="a", role="R", model="m", persona="p", max_retries=10)
        result = resolve_member_setting(mc, defaults, "max_retries")
        assert result == 10

    def test_falls_back_to_default(self):
        from team.config import resolve_member_setting, MemberConfig
        defaults = self._make_defaults(max_retries=5)
        mc = MemberConfig(name="a", role="R", model="m", persona="p")
        # member has no override (None) → resolve_member_setting returns the default value
        result = resolve_member_setting(mc, defaults, "max_retries")
        assert result == 5

    def test_backoff_member_override(self):
        from team.config import resolve_member_setting, MemberConfig
        defaults = self._make_defaults(retry_backoff=2.0)
        mc = MemberConfig(name="a", role="R", model="m", persona="p", retry_backoff=0.1)
        result = resolve_member_setting(mc, defaults, "retry_backoff")
        assert result == 0.1


# --------------------------------------------------------------------------- #
# CLI: 'run' command catches LLMRetryExhaustedError
# --------------------------------------------------------------------------- #


def test_run_cli_catches_llm_retry_exhausted_error(tmp_path):
    """CLI 'run' should print a red panel and exit 0 on LLMRetryExhaustedError."""
    from click.testing import CliRunner
    from team.cli import cli

    team_yaml = tmp_path / "team.yaml"
    team_yaml.write_text(textwrap.dedent("""
        name: t
        goal: g
        members:
          - name: alice
            model: llama3
            persona: hi
            role: R
    """))

    runner = CliRunner()

    with patch("team.orchestrator.ContainerManager") as mock_cm, \
         patch("team.orchestrator.Orchestrator.up"), \
         patch("team.orchestrator.Orchestrator.down"), \
         patch("team.orchestrator.Orchestrator.run",
               side_effect=LLMRetryExhaustedError("all retries failed")):
        result = runner.invoke(cli, ["run", str(team_yaml)])

    assert result.exit_code == 0
    assert "LLM connection failed" in result.output or "retries" in result.output.lower()


def test_test_cmd_cli_catches_llm_retry_exhausted_error(tmp_path):
    """CLI 'test' command exits 1 (not crash) on LLMRetryExhaustedError."""
    from click.testing import CliRunner
    from team.cli import cli

    team_yaml = tmp_path / "team.yaml"
    team_yaml.write_text(textwrap.dedent("""
        name: t
        goal: g
        tests:
          - type: contains
            member: alice
            text: hello
        members:
          - name: alice
            model: llama3
            persona: hi
            role: R
    """))

    runner = CliRunner()

    with patch("team.orchestrator.ContainerManager"), \
         patch("team.orchestrator.Orchestrator.up"), \
         patch("team.orchestrator.Orchestrator.down"), \
         patch("team.orchestrator.Orchestrator.run",
               side_effect=LLMRetryExhaustedError("all retries failed")):
        result = runner.invoke(cli, ["test", str(team_yaml)])

    # Should exit with code 1 because the run failed completely
    assert result.exit_code == 1
    assert "retries" in result.output.lower() or "connection" in result.output.lower() or "failed" in result.output.lower()
