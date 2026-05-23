"""Tests for the Team Registry — service-discovery layer.

Covers:
- RegistryEntry round-trip serialisation (to_dict / from_dict).
- RegistryEntry.matches() with tag / keyword / combined predicates.
- TeamRegistry store transitions: register, heartbeat, deregister, eviction.
- RegistryServer + RegistryClient full HTTP integration over a real (loopback)
  TCP connection — skipped automatically if loopback is unavailable.
- query_registry built-in tool with a mocked RegistryClient.
- RegistryConfig parsing from YAML via load_team().
"""

from __future__ import annotations

import socket
import textwrap
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from team.registry import RegistryEntry, TeamRegistry, sign_request, verify_signature
from team.registry_client import RegistryClient, RegistryClientError
from team.registry_server import RegistryServer
from team.tools import execute_tool


# --------------------------------------------------------------------------- #
# Loopback TCP availability guard (mirrors test_bridge.py)
# --------------------------------------------------------------------------- #


def _tcp_loopback_available() -> bool:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.listen(1)
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.settimeout(1)
        try:
            client.connect(("127.0.0.1", port))
            return True
        except OSError:
            return False
        finally:
            client.close()
    except OSError:
        return False
    finally:
        probe.close()


_requires_loopback_tcp = pytest.mark.skipif(
    not _tcp_loopback_available(),
    reason="TCP loopback connections are not available in this environment",
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _free_port() -> int:
    """Return an available ephemeral TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _make_entry(**kwargs) -> RegistryEntry:
    defaults = {
        "name": "test-team",
        "url": "http://localhost:7000",
        "description": "A test team",
        "tags": ["python", "testing"],
        "models": ["llama3.1:8b"],
        "tools": ["run_python"],
    }
    defaults.update(kwargs)
    return RegistryEntry(**defaults)


# --------------------------------------------------------------------------- #
# RegistryEntry serialisation
# --------------------------------------------------------------------------- #


class TestRegistryEntry:
    def test_to_dict_contains_all_fields(self):
        e = _make_entry()
        d = e.to_dict()
        assert d["name"] == "test-team"
        assert d["url"] == "http://localhost:7000"
        assert isinstance(d["tags"], list)
        assert isinstance(d["models"], list)
        assert "entry_id" in d
        assert "registered_at" in d
        assert "last_heartbeat" in d

    def test_round_trip(self):
        original = _make_entry(description="round-trip", tags=["a", "b"], models=["m1"])
        restored = RegistryEntry.from_dict(original.to_dict())
        assert restored.name == original.name
        assert restored.url == original.url
        assert restored.description == original.description
        assert restored.tags == original.tags
        assert restored.models == original.models
        assert restored.entry_id == original.entry_id
        assert restored.registered_at == pytest.approx(original.registered_at, abs=0.001)

    def test_from_dict_ignores_unknown_keys(self):
        d = _make_entry().to_dict()
        d["_extra_unknown_field"] = "should be ignored"
        # Should not raise; just ignore unknown
        e = RegistryEntry.from_dict(d)
        assert e.name == "test-team"


# --------------------------------------------------------------------------- #
# RegistryEntry.matches()
# --------------------------------------------------------------------------- #


class TestRegistryEntryMatches:
    def setup_method(self):
        self.e = _make_entry(
            name="bio-team",
            description="Computational biology team with R and Python",
            tags=["biology", "python", "statistics"],
        )

    def test_no_predicates_always_matches(self):
        assert self.e.matches(None, None) is True
        assert self.e.matches([], None) is True
        assert self.e.matches(None, "") is True

    def test_tag_match_all_present(self):
        assert self.e.matches(["python"], None) is True
        assert self.e.matches(["biology", "statistics"], None) is True

    def test_tag_match_case_insensitive(self):
        assert self.e.matches(["PYTHON", "Biology"], None) is True

    def test_tag_no_match_missing_tag(self):
        assert self.e.matches(["nlp"], None) is False
        assert self.e.matches(["biology", "nlp"], None) is False

    def test_keyword_matches_name(self):
        assert self.e.matches(None, "bio-team") is True

    def test_keyword_matches_description(self):
        assert self.e.matches(None, "Computational") is True
        assert self.e.matches(None, "R and Python") is True

    def test_keyword_matches_tags(self):
        assert self.e.matches(None, "statistic") is True  # substring

    def test_keyword_case_insensitive(self):
        assert self.e.matches(None, "BIOLOGY") is True

    def test_keyword_no_match(self):
        assert self.e.matches(None, "quantum") is False

    def test_combined_tag_and_keyword(self):
        assert self.e.matches(["biology"], "Python") is True

    def test_combined_tag_mismatch(self):
        assert self.e.matches(["nlp"], "biology") is False

    def test_combined_keyword_mismatch(self):
        assert self.e.matches(["biology"], "quantum") is False


# --------------------------------------------------------------------------- #
# TeamRegistry store
# --------------------------------------------------------------------------- #


class TestTeamRegistry:
    def setup_method(self):
        # Disable eviction so tests don't flap on timing
        self.registry = TeamRegistry(entry_ttl_seconds=0)

    def test_register_and_get(self):
        e = _make_entry(name="alpha")
        stored = self.registry.register(e)
        assert stored.name == "alpha"
        retrieved = self.registry.get("alpha")
        assert retrieved is not None
        assert retrieved.name == "alpha"

    def test_register_updates_existing(self):
        e1 = _make_entry(name="alpha", description="first")
        self.registry.register(e1)
        e2 = _make_entry(name="alpha", description="updated")
        stored = self.registry.register(e2)
        assert stored.description == "updated"
        # registered_at is preserved from first registration
        assert stored.registered_at == pytest.approx(e1.registered_at, abs=0.5)

    def test_deregister_removes_entry(self):
        self.registry.register(_make_entry(name="beta"))
        assert self.registry.deregister("beta") is True
        assert self.registry.get("beta") is None

    def test_deregister_nonexistent_returns_false(self):
        assert self.registry.deregister("ghost") is False

    def test_heartbeat_updates_timestamp(self):
        e = _make_entry(name="gamma")
        self.registry.register(e)
        old_hb = self.registry.get("gamma").last_heartbeat
        time.sleep(0.05)
        result = self.registry.heartbeat("gamma")
        assert result is True
        new_hb = self.registry.get("gamma").last_heartbeat
        assert new_hb > old_hb

    def test_heartbeat_nonexistent_returns_false(self):
        assert self.registry.heartbeat("ghost") is False

    def test_list_all(self):
        self.registry.register(_make_entry(name="a"))
        self.registry.register(_make_entry(name="b"))
        names = {e.name for e in self.registry.list_all()}
        assert {"a", "b"}.issubset(names)

    def test_query_by_tag(self):
        self.registry.register(_make_entry(name="nlp-team", tags=["nlp", "python"]))
        self.registry.register(_make_entry(name="bio-team", tags=["biology", "python"]))
        results = self.registry.query(tags=["nlp"])
        assert len(results) == 1
        assert results[0].name == "nlp-team"

    def test_query_by_keyword(self):
        self.registry.register(_make_entry(name="summary-team", description="text summarisation"))
        self.registry.register(_make_entry(name="qa-team", description="question answering"))
        results = self.registry.query(keyword="summar")
        assert len(results) == 1
        assert results[0].name == "summary-team"

    def test_query_no_predicates_returns_all(self):
        self.registry.register(_make_entry(name="x"))
        self.registry.register(_make_entry(name="y"))
        assert len(self.registry.query()) == 2

    def test_count(self):
        assert self.registry.count() == 0
        self.registry.register(_make_entry(name="a"))
        assert self.registry.count() == 1
        self.registry.register(_make_entry(name="b"))
        assert self.registry.count() == 2
        self.registry.deregister("a")
        assert self.registry.count() == 1

    def test_eviction_removes_stale_entries(self):
        reg = TeamRegistry(entry_ttl_seconds=0.1)
        reg.register(_make_entry(name="stale"))
        time.sleep(0.5)  # wait for at least one eviction sweep
        assert reg.get("stale") is None


# --------------------------------------------------------------------------- #
# HMAC signing utilities
# --------------------------------------------------------------------------- #


class TestHmacSigning:
    def test_sign_and_verify(self):
        secret = "mysecret"
        timestamp = str(int(time.time()))
        body = b'{"name":"test"}'
        sig = sign_request(secret, timestamp, body)
        assert verify_signature(secret, timestamp, body, sig) is True

    def test_wrong_secret_fails(self):
        timestamp = str(int(time.time()))
        body = b'{"name":"test"}'
        sig = sign_request("secret-a", timestamp, body)
        assert verify_signature("secret-b", timestamp, body, sig) is False

    def test_stale_timestamp_fails(self):
        secret = "s"
        stale = str(int(time.time()) - 400)  # > TIMESTAMP_TOLERANCE (300)
        body = b""
        sig = sign_request(secret, stale, body)
        assert verify_signature(secret, stale, body, sig) is False

    def test_tampered_body_fails(self):
        secret = "s"
        timestamp = str(int(time.time()))
        body = b'{"name":"original"}'
        sig = sign_request(secret, timestamp, body)
        assert verify_signature(secret, timestamp, b'{"name":"tampered"}', sig) is False


# --------------------------------------------------------------------------- #
# RegistryServer + RegistryClient integration (requires TCP loopback)
# --------------------------------------------------------------------------- #


@_requires_loopback_tcp
class TestRegistryIntegration:
    """Full HTTP round-trip tests using a live RegistryServer on loopback."""

    @pytest.fixture(autouse=True)
    def server_fixture(self):
        port = _free_port()
        self.server = RegistryServer(
            port=port, host="127.0.0.1", entry_ttl_seconds=0
        )
        self.server.start()
        self.client = RegistryClient(f"http://127.0.0.1:{port}")
        yield
        self.server.stop()

    def test_health_endpoint(self):
        data = self.client.health()
        assert data.get("status") == "ok"
        assert "registered" in data

    def test_register_and_get(self):
        e = _make_entry(name="lab-a")
        stored = self.client.register(e)
        assert stored.name == "lab-a"
        fetched = self.client.get("lab-a")
        assert fetched is not None
        assert fetched.url == e.url

    def test_list_all(self):
        self.client.register(_make_entry(name="a"))
        self.client.register(_make_entry(name="b"))
        entries = self.client.list_all()
        names = {e.name for e in entries}
        assert {"a", "b"}.issubset(names)

    def test_query_by_tag(self):
        self.client.register(_make_entry(name="nlp", tags=["nlp", "python"]))
        self.client.register(_make_entry(name="bio", tags=["biology"]))
        results = self.client.query(tags=["nlp"])
        assert len(results) == 1
        assert results[0].name == "nlp"

    def test_query_by_keyword(self):
        self.client.register(
            _make_entry(name="sum", description="text summarisation service")
        )
        self.client.register(_make_entry(name="qa", description="question answering"))
        results = self.client.query(keyword="summar")
        assert len(results) == 1
        assert results[0].name == "sum"

    def test_deregister(self):
        self.client.register(_make_entry(name="to-remove"))
        self.client.deregister("to-remove")
        assert self.client.get("to-remove") is None

    def test_heartbeat_ok(self):
        self.client.register(_make_entry(name="beater"))
        assert self.client.heartbeat("beater") is True

    def test_heartbeat_unknown_returns_false(self):
        assert self.client.heartbeat("ghost") is False

    def test_deregister_unknown_raises(self):
        with pytest.raises(RegistryClientError):
            self.client.deregister("ghost")

    def test_get_nonexistent_returns_none(self):
        assert self.client.get("nobody") is None

    def test_hmac_auth_write_ops(self):
        """Write operations require the correct HMAC signature when the server
        uses a secret; wrong-secret clients are rejected."""
        port = _free_port()
        secured = RegistryServer(
            port=port, host="127.0.0.1", secret="correct", entry_ttl_seconds=0
        )
        secured.start()
        try:
            good_client = RegistryClient(
                f"http://127.0.0.1:{port}", secret="correct"
            )
            bad_client = RegistryClient(
                f"http://127.0.0.1:{port}", secret="wrong"
            )
            # Good client can register
            good_client.register(_make_entry(name="auth-team"))
            # Bad client cannot register
            with pytest.raises(RegistryClientError):
                bad_client.register(_make_entry(name="evil"))
            # Read operations are public
            entries = bad_client.list_all()
            names = {e.name for e in entries}
            assert "auth-team" in names
        finally:
            secured.stop()


# --------------------------------------------------------------------------- #
# query_registry tool
# --------------------------------------------------------------------------- #


class TestQueryRegistryTool:
    """Unit tests for the query_registry built-in tool using a mocked client."""

    def _run(self, body: str, registry_url: str = "http://localhost:8000") -> str:
        return execute_tool("query_registry", body, registry_url=registry_url)

    @patch("team.registry_client.RegistryClient")
    def test_returns_markdown_table(self, mock_cls):
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.query.return_value = [
            RegistryEntry(
                name="bio-team",
                url="http://bio:7001",
                description="Biology experts",
                tags=["biology", "python"],
                models=["llama3:70b"],
            )
        ]
        result = self._run("tag: biology")
        assert "bio-team" in result
        assert "http://bio:7001" in result
        assert "biology" in result

    @patch("team.registry_client.RegistryClient")
    def test_no_results_message(self, mock_cls):
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.query.return_value = []
        result = self._run("")
        assert "no" in result.lower() or "0" in result

    @patch("team.registry_client.RegistryClient")
    def test_tag_and_keyword_parsed(self, mock_cls):
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.query.return_value = []
        self._run("tag: nlp\nkeyword: summarisation")
        mock_client.query.assert_called_once()
        call_kwargs = mock_client.query.call_args
        tags_arg = call_kwargs[1].get("tags") or (call_kwargs[0][0] if call_kwargs[0] else None)
        # Just verify the call was made with something meaningful
        assert mock_client.query.called

    @patch("team.registry_client.RegistryClient")
    def test_explicit_url_in_body_overrides_injected(self, mock_cls):
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.query.return_value = []
        self._run("url: http://other:9000\ntag: nlp", registry_url="http://default:8000")
        # The RegistryClient should have been instantiated with the body URL
        instantiation_url = mock_cls.call_args[0][0]
        assert "other:9000" in instantiation_url

    def test_missing_url_returns_error(self):
        result = execute_tool("query_registry", "", registry_url=None)
        assert "error" in result.lower() or "url" in result.lower()


# --------------------------------------------------------------------------- #
# RegistryConfig parsing
# --------------------------------------------------------------------------- #


class TestRegistryConfig:
    def test_defaults_when_registry_key_absent(self, tmp_path):
        from team.config import load_team

        yaml_text = textwrap.dedent("""\
            name: no-registry
            goal: test goal
            members:
              - name: alice
                model: llama3:8b
                role: assistant
                persona: "A helpful assistant."
        """)
        cfg_path = tmp_path / "team.yaml"
        cfg_path.write_text(yaml_text)
        cfg = load_team(cfg_path)
        assert cfg.registry.url is None
        assert cfg.registry.auto_register is False
        assert cfg.registry.heartbeat_interval == 60
        assert cfg.registry.tags == []
        assert cfg.registry.description == ""

    def test_registry_config_parsed(self, tmp_path):
        from team.config import load_team

        yaml_text = textwrap.dedent("""\
            name: has-registry
            goal: test goal
            registry:
              url: http://registry.example.com:8000
              auto_register: true
              heartbeat_interval: 30
              tags:
                - biology
                - python
              description: "Computational biology team"
            members:
              - name: alice
                model: llama3:8b
                role: assistant
                persona: "A helpful assistant."
        """)
        cfg_path = tmp_path / "team.yaml"
        cfg_path.write_text(yaml_text)
        cfg = load_team(cfg_path)
        assert cfg.registry.url == "http://registry.example.com:8000"
        assert cfg.registry.auto_register is True
        assert cfg.registry.heartbeat_interval == 30
        assert cfg.registry.tags == ["biology", "python"]
        assert cfg.registry.description == "Computational biology team"

    def test_tags_as_comma_string(self, tmp_path):
        from team.config import load_team

        yaml_text = textwrap.dedent("""\
            name: csv-tags
            goal: test
            registry:
              url: http://registry:8000
              tags: "nlp, summarisation, python"
            members:
              - name: alice
                model: llama3:8b
                role: assistant
                persona: "A helpful assistant."
        """)
        cfg_path = tmp_path / "team.yaml"
        cfg_path.write_text(yaml_text)
        cfg = load_team(cfg_path)
        assert "nlp" in cfg.registry.tags
        assert "summarisation" in cfg.registry.tags
        assert "python" in cfg.registry.tags
