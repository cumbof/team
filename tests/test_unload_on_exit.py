"""Tests for model unload-on-exit (unload_on_exit config flag).

Covers:
- OllamaClient.unload_model() sends the correct POST request.
- Orchestrator.down() calls unload_model when unload_on_exit=True for
  local-Ollama members (container=None).
- Orchestrator.down() does NOT call unload_model when the flag is False
  or the member runs in a Docker container.
- A failed unload is logged as a warning and does not raise.
- Config: unload_on_exit parsed from defaults and per-member YAML.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from team.config import Defaults, MemberConfig, TeamConfig, WorkflowConfig
from team.ollama_client import OllamaClient, OllamaError


# --------------------------------------------------------------------------- #
# OllamaClient.unload_model
# --------------------------------------------------------------------------- #


class TestUnloadModel:
    def _client(self) -> OllamaClient:
        return OllamaClient(base_url="http://localhost:11434")

    def test_sends_generate_request_with_keep_alive_zero(self):
        client = self._client()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        with patch.object(client._session, "post", return_value=mock_resp) as mock_post:
            client.unload_model("llama3")
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        assert "/api/generate" in call_kwargs[0][0]
        payload = call_kwargs[1]["json"]
        assert payload["model"] == "llama3"
        assert payload["keep_alive"] == 0

    def test_raises_ollama_error_on_request_failure(self):
        client = self._client()
        with patch.object(
            client._session, "post", side_effect=requests.ConnectionError("refused")
        ):
            with pytest.raises(OllamaError, match="failed to unload"):
                client.unload_model("llama3")

    def test_raises_ollama_error_on_http_error(self):
        client = self._client()
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.HTTPError("500")
        with patch.object(client._session, "post", return_value=mock_resp):
            with pytest.raises(OllamaError, match="failed to unload"):
                client.unload_model("llama3")


# --------------------------------------------------------------------------- #
# Config parsing
# --------------------------------------------------------------------------- #


class TestConfigParsing:
    def test_defaults_unload_on_exit_is_false(self):
        assert Defaults().unload_on_exit is False

    def test_member_unload_on_exit_is_none_by_default(self):
        m = MemberConfig(name="a", role="R", model="m", persona="p")
        assert m.unload_on_exit is None

    def test_load_team_defaults_unload_on_exit(self, tmp_path):
        from team.config import load_team
        yaml_text = (
            f"name: t\ngoal: g\nworkspace: {tmp_path/'ws'}\n"
            "defaults:\n  unload_on_exit: true\n"
            "members:\n  - name: a\n    role: R\n    model: m\n    persona: p\n"
            "workflow:\n  type: round_robin\n  max_rounds: 1\n"
        )
        p = tmp_path / "t.yaml"
        p.write_text(yaml_text)
        cfg = load_team(p)
        assert cfg.defaults.unload_on_exit is True

    def test_load_team_member_unload_on_exit(self, tmp_path):
        from team.config import load_team
        yaml_text = (
            f"name: t\ngoal: g\nworkspace: {tmp_path/'ws'}\n"
            "members:\n  - name: a\n    role: R\n    model: m\n    persona: p\n"
            "    unload_on_exit: true\n"
            "workflow:\n  type: round_robin\n  max_rounds: 1\n"
        )
        p = tmp_path / "t.yaml"
        p.write_text(yaml_text)
        cfg = load_team(p)
        assert cfg.members[0].unload_on_exit is True


# --------------------------------------------------------------------------- #
# Orchestrator.down() integration
# --------------------------------------------------------------------------- #


def _make_team(tmp_path, *, unload_default=False, member_unload=None):
    defaults = Defaults()
    defaults.unload_on_exit = unload_default
    member = MemberConfig(
        name="worker",
        role="Worker",
        model="smollm2:135m",
        persona="You work.",
        unload_on_exit=member_unload,
    )
    return TeamConfig(
        name="t",
        goal="g",
        workspace=tmp_path / "ws",
        workflow=WorkflowConfig(type="round_robin", max_rounds=1),
        defaults=defaults,
        members=[member],
    )


def _make_orch_with_member(team, *, container=None):
    """Build an Orchestrator with a mocked Member attached."""
    from team.orchestrator import Orchestrator
    from team.container import MemberRuntime

    mock_cm = MagicMock()
    orch = Orchestrator(team, container_manager=mock_cm)
    orch.checkpoints = MagicMock()

    # Build a mock Member with a controllable runtime and OllamaClient.
    mock_client = MagicMock(spec=OllamaClient)
    runtime = MemberRuntime(
        member=team.members[0],
        container=container,
        host_port=None,
        base_url="http://localhost:11434",
    )
    mock_member = MagicMock()
    mock_member.name = "worker"
    mock_member.config = team.members[0]
    mock_member.client = mock_client
    mock_member.runtime = runtime
    orch.members["worker"] = mock_member
    return orch, mock_client


class TestOrchestratorDown:
    def test_unload_called_for_local_ollama_when_flag_true(self, tmp_path):
        team = _make_team(tmp_path, unload_default=True)
        orch, mock_client = _make_orch_with_member(team, container=None)
        orch.down()
        mock_client.unload_model.assert_called_once_with("smollm2:135m")

    def test_unload_not_called_when_flag_false(self, tmp_path):
        team = _make_team(tmp_path, unload_default=False)
        orch, mock_client = _make_orch_with_member(team, container=None)
        orch.down()
        mock_client.unload_model.assert_not_called()

    def test_unload_not_called_for_docker_container_member(self, tmp_path):
        team = _make_team(tmp_path, unload_default=True)
        fake_container = MagicMock()  # non-None container = Docker member
        orch, mock_client = _make_orch_with_member(team, container=fake_container)
        orch.down()
        mock_client.unload_model.assert_not_called()

    def test_member_level_override_true(self, tmp_path):
        """Per-member unload_on_exit=True overrides defaults=False."""
        team = _make_team(tmp_path, unload_default=False, member_unload=True)
        orch, mock_client = _make_orch_with_member(team, container=None)
        orch.down()
        mock_client.unload_model.assert_called_once_with("smollm2:135m")

    def test_failed_unload_logs_warning_not_raises(self, tmp_path, caplog):
        import logging
        team = _make_team(tmp_path, unload_default=True)
        orch, mock_client = _make_orch_with_member(team, container=None)
        mock_client.unload_model.side_effect = OllamaError("connection refused")
        with caplog.at_level(logging.WARNING, logger="team.orchestrator"):
            orch.down()  # must not raise
        assert "could not unload" in caplog.text
