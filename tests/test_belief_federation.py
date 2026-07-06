"""Tests for the Federated Belief Board — cross-team belief synchronization.

Covers:
- claim_hash / normalize_claim deduplication utilities.
- BeliefBundle round-trip serialisation (to_dict / from_dict).
- export_beliefs — status filter, limit, bundle metadata.
- import_beliefs — dedup (same claim not re-imported), new beliefs added as
  pending with correct author attribution.
- BeliefBoard.receive_remote_belief — provenance annotation, no votes, status.
- BridgeServer GET /beliefs and POST /beliefs/import endpoints over loopback.
- BridgeClient.pull_beliefs / push_beliefs methods.
- sync_beliefs built-in tool with mocked BridgeClient.
"""

from __future__ import annotations

import socket
import textwrap
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from team.beliefs import BeliefBoard
from team.belief_federation import (
    BeliefBundle,
    claim_hash,
    export_beliefs,
    import_beliefs,
)


# --------------------------------------------------------------------------- #
# Loopback guard (mirrors test_bridge.py)
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


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _board(tmp_path: Path, members: list[str] | None = None) -> BeliefBoard:
    return BeliefBoard(
        path=tmp_path / "beliefs.json",
        member_names=members or ["alice", "bob"],
    )


# --------------------------------------------------------------------------- #
# claim_hash / normalization
# --------------------------------------------------------------------------- #


class TestClaimHash:
    def test_same_text_gives_same_hash(self):
        assert claim_hash("The sky is blue.") == claim_hash("The sky is blue.")

    def test_case_insensitive(self):
        assert claim_hash("The Sky Is Blue") == claim_hash("the sky is blue")

    def test_punctuation_stripped(self):
        assert claim_hash("Python is great!") == claim_hash("Python is great")

    def test_whitespace_collapsed(self):
        assert claim_hash("hello  world") == claim_hash("hello world")

    def test_different_claims_differ(self):
        assert claim_hash("A is true") != claim_hash("B is true")

    def test_returns_16_hex_chars(self):
        h = claim_hash("some claim")
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)


# --------------------------------------------------------------------------- #
# BeliefBundle serialisation
# --------------------------------------------------------------------------- #


class TestBeliefBundle:
    def test_to_dict_contains_all_fields(self):
        bundle = BeliefBundle(
            source_team="lab-b",
            source_url="http://lab-b:7001",
            beliefs=[{"id": "abc", "claim": "X is true"}],
            filter_status="accepted",
        )
        d = bundle.to_dict()
        assert d["source_team"] == "lab-b"
        assert d["source_url"] == "http://lab-b:7001"
        assert len(d["beliefs"]) == 1
        assert d["filter_status"] == "accepted"
        assert "exported_at" in d

    def test_round_trip(self):
        original = BeliefBundle(
            source_team="lab-a",
            source_url="http://lab-a:7000",
            beliefs=[{"claim": "AI is useful"}],
            filter_status=None,
        )
        restored = BeliefBundle.from_dict(original.to_dict())
        assert restored.source_team == original.source_team
        assert restored.source_url == original.source_url
        assert restored.beliefs == original.beliefs
        assert restored.filter_status is None
        assert restored.exported_at == pytest.approx(original.exported_at, abs=0.001)

    def test_from_dict_ignores_unknown_keys(self):
        d = BeliefBundle("t", "http://t").to_dict()
        d["_unknown"] = "ignored"
        bundle = BeliefBundle.from_dict(d)
        assert bundle.source_team == "t"


# --------------------------------------------------------------------------- #
# export_beliefs
# --------------------------------------------------------------------------- #


class TestExportBeliefs:
    def test_exports_all_beliefs(self, tmp_path):
        board = _board(tmp_path)
        board.assert_belief("Claim A", "alice", confidence=0.8)
        board.assert_belief("Claim B", "bob", confidence=0.6)
        bundle = export_beliefs(board, "my-team", "http://me:7000")
        assert bundle.source_team == "my-team"
        assert bundle.source_url == "http://me:7000"
        assert len(bundle.beliefs) == 2
        assert bundle.filter_status is None

    def test_status_filter(self, tmp_path):
        board = _board(tmp_path, members=["alice"])
        # With 1 member, assert_belief auto-accepts (ratio_for = 1.0 >= 0.5)
        board.assert_belief("Accepted claim", "alice", confidence=0.9)
        board.assert_belief("Another claim", "alice", confidence=0.5)
        accepted_bundle = export_beliefs(board, "t", "http://t", status="accepted")
        assert len(accepted_bundle.beliefs) >= 1
        assert accepted_bundle.filter_status == "accepted"
        for b in accepted_bundle.beliefs:
            assert b["status"] == "accepted"

    def test_limit(self, tmp_path):
        board = _board(tmp_path)
        for i in range(10):
            board.assert_belief(f"Claim {i}", "alice")
        bundle = export_beliefs(board, "t", "http://t", limit=3)
        assert len(bundle.beliefs) == 3


# --------------------------------------------------------------------------- #
# import_beliefs
# --------------------------------------------------------------------------- #


class TestImportBeliefs:
    def test_imports_new_belief(self, tmp_path):
        board = _board(tmp_path)
        bundle = BeliefBundle(
            source_team="remote-team",
            source_url="http://remote:7001",
            beliefs=[{
                "id": "abc",
                "claim": "Remote finding is significant",
                "author": "carol",
                "confidence": 0.85,
                "evidence": "See paper XYZ",
                "status": "accepted",
                "votes_for": ["carol"],
                "votes_against": [],
                "reasons": {},
                "created_at": time.time(),
                "updated_at": time.time(),
            }],
        )
        imported, skipped = import_beliefs(board, bundle)
        assert imported == 1
        assert skipped == 0
        beliefs = board.list_beliefs()
        assert len(beliefs) == 1
        b = beliefs[0]
        assert b.status == "pending"  # imported as pending
        assert "remote-team" in b.author
        assert "carol" in b.author
        assert b.votes_for == []  # no votes on import

    def test_dedup_skips_same_claim(self, tmp_path):
        board = _board(tmp_path)
        board.assert_belief("The earth is round", "alice")
        bundle = BeliefBundle(
            source_team="remote",
            source_url="http://remote",
            beliefs=[{
                "id": "xyz",
                "claim": "The earth is round!",  # same claim, punctuation differs
                "author": "bob",
                "confidence": 0.9,
                "evidence": "",
                "status": "accepted",
                "votes_for": ["bob"],
                "votes_against": [],
                "reasons": {},
                "created_at": time.time(),
                "updated_at": time.time(),
            }],
        )
        imported, skipped = import_beliefs(board, bundle)
        assert imported == 0
        assert skipped == 1
        # Board still has only 1 belief
        assert board.count() == 1

    def test_dedup_case_insensitive(self, tmp_path):
        board = _board(tmp_path)
        board.assert_belief("Python is great", "alice")
        bundle = BeliefBundle(
            source_team="remote",
            source_url="http://remote",
            beliefs=[{
                "id": "y",
                "claim": "PYTHON IS GREAT",
                "author": "bob",
                "confidence": 0.7,
                "evidence": "",
                "status": "accepted",
                "votes_for": ["bob"],
                "votes_against": [],
                "reasons": {},
                "created_at": time.time(),
                "updated_at": time.time(),
            }],
        )
        imported, skipped = import_beliefs(board, bundle)
        assert imported == 0
        assert skipped == 1

    def test_intra_bundle_dedup(self, tmp_path):
        """Two beliefs with the same claim in one bundle — only one imported."""
        board = _board(tmp_path)
        dup_belief = {
            "id": "dup",
            "claim": "Same claim",
            "author": "bob",
            "confidence": 0.8,
            "evidence": "",
            "status": "accepted",
            "votes_for": ["bob"],
            "votes_against": [],
            "reasons": {},
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        bundle = BeliefBundle(
            source_team="remote",
            source_url="http://remote",
            beliefs=[dup_belief, {**dup_belief, "id": "dup2"}],
        )
        imported, skipped = import_beliefs(board, bundle)
        assert imported == 1
        assert skipped == 1

    def test_provenance_in_evidence(self, tmp_path):
        board = _board(tmp_path)
        bundle = BeliefBundle(
            source_team="lab-z",
            source_url="http://lab-z",
            beliefs=[{
                "id": "prov",
                "claim": "This was discovered by lab-z",
                "author": "zoe",
                "confidence": 0.7,
                "evidence": "Original evidence text",
                "status": "accepted",
                "votes_for": ["zoe"],
                "votes_against": [],
                "reasons": {},
                "created_at": time.time(),
                "updated_at": time.time(),
            }],
        )
        import_beliefs(board, bundle)
        b = board.list_beliefs()[0]
        assert "lab-z" in b.evidence
        assert "prov" in b.evidence  # original_id


# --------------------------------------------------------------------------- #
# BeliefBoard.receive_remote_belief
# --------------------------------------------------------------------------- #


class TestReceiveRemoteBelief:
    def test_creates_pending_belief_with_no_votes(self, tmp_path):
        board = _board(tmp_path)
        b = board.receive_remote_belief(
            claim="X leads to Y",
            source_team="team-b",
            original_author="alice",
            confidence=0.75,
            evidence="Study results",
            original_id="abc123",
        )
        assert b.status == "pending"
        assert b.votes_for == []
        assert b.votes_against == []
        assert b.author == "team-b/alice"
        assert "team-b" in b.evidence
        assert "abc123" in b.evidence

    def test_saved_to_disk(self, tmp_path):
        board = _board(tmp_path)
        board.receive_remote_belief(
            claim="Persistence test",
            source_team="remote",
            original_author="bob",
        )
        # Reload from disk
        board2 = _board(tmp_path)
        beliefs = board2.list_beliefs()
        assert len(beliefs) == 1
        assert beliefs[0].claim == "Persistence test"


# --------------------------------------------------------------------------- #
# Bridge server/client belief federation integration (requires loopback TCP)
# --------------------------------------------------------------------------- #


@_requires_loopback_tcp
class TestBridgeFederationIntegration:
    """Full HTTP round-trip for belief sync endpoints."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        from team.bridge_server import BridgeServer

        def _stub_runner(ws, goal, ctx, files):
            return "stub", {}

        port = _free_port()
        self.beliefs_path = tmp_path / "beliefs.json"
        self.tmp_path = tmp_path

        # Populate the server's belief board with one belief.
        server_board = BeliefBoard(self.beliefs_path, ["alice"])
        server_board.assert_belief("Shared finding", "alice", confidence=0.9)

        self.server = BridgeServer(
            runner=_stub_runner,
            port=port,
            workspace_root=tmp_path / "ws",
            beliefs_path=self.beliefs_path,
            capabilities={"name": "remote-team"},
        )
        self.server.start()
        from team.bridge_client import BridgeClient
        self.client = BridgeClient(f"http://127.0.0.1:{port}")
        yield
        self.server.stop()

    def test_get_beliefs_returns_bundle(self):
        bundle = self.client.pull_beliefs()
        assert bundle is not None
        assert bundle.source_team == "remote-team"
        assert len(bundle.beliefs) == 1
        assert bundle.beliefs[0]["claim"] == "Shared finding"

    def test_get_beliefs_status_filter(self):
        bundle = self.client.pull_beliefs(status="accepted")
        assert bundle is not None
        for b in bundle.beliefs:
            assert b["status"] == "accepted"

    def test_get_beliefs_empty_when_no_file(self, tmp_path):
        """Server with beliefs_path pointing to non-existent file returns 404."""
        from team.bridge_server import BridgeServer

        def _stub_runner(ws, goal, ctx, files):
            return "stub", {}

        port = _free_port()
        server = BridgeServer(
            runner=_stub_runner,
            port=port,
            workspace_root=tmp_path / "ws2",
            beliefs_path=tmp_path / "nonexistent_beliefs.json",
        )
        server.start()
        try:
            from team.bridge_client import BridgeClient
            client = BridgeClient(f"http://127.0.0.1:{port}")
            result = client.pull_beliefs()
            assert result is None  # 404 → None
        finally:
            server.stop()

    def test_pull_beliefs_returns_none_when_no_path(self, tmp_path):
        """Server without beliefs_path returns 404 → client returns None."""
        from team.bridge_server import BridgeServer

        def _stub_runner(ws, goal, ctx, files):
            return "stub", {}

        port = _free_port()
        server = BridgeServer(
            runner=_stub_runner,
            port=port,
            workspace_root=tmp_path / "ws3",
            beliefs_path=None,  # no beliefs configured
        )
        server.start()
        try:
            from team.bridge_client import BridgeClient
            client = BridgeClient(f"http://127.0.0.1:{port}")
            result = client.pull_beliefs()
            assert result is None
        finally:
            server.stop()

    def test_push_beliefs_and_import(self, tmp_path):
        """Push a bundle to the server and verify it was imported."""
        # Use a hand-crafted bundle with one new belief
        bundle_to_push = BeliefBundle(
            source_team="lab-a",
            source_url="http://lab-a",
            beliefs=[{
                "id": "new1",
                "claim": "Cross-team discovery about X",
                "author": "carol",
                "confidence": 0.8,
                "evidence": "Computed from dataset",
                "status": "accepted",
                "votes_for": ["carol"],
                "votes_against": [],
                "reasons": {},
                "created_at": time.time(),
                "updated_at": time.time(),
            }],
        )
        result = self.client.push_beliefs(bundle_to_push)
        assert result is not None
        imported, skipped = result
        assert imported == 1
        assert skipped == 0
        # Verify the server's board was updated
        updated_board = BeliefBoard(self.beliefs_path, ["alice"])
        claims = {b.claim for b in updated_board.list_beliefs()}
        assert "Cross-team discovery about X" in claims

    def test_push_dedup_skips_existing(self):
        """Pushing a belief that already exists should be skipped."""
        bundle = BeliefBundle(
            source_team="dup-team",
            source_url="http://dup",
            beliefs=[{
                "id": "dup",
                "claim": "Shared finding",  # same as server's existing belief
                "author": "eve",
                "confidence": 0.7,
                "evidence": "",
                "status": "accepted",
                "votes_for": ["eve"],
                "votes_against": [],
                "reasons": {},
                "created_at": time.time(),
                "updated_at": time.time(),
            }],
        )
        result = self.client.push_beliefs(bundle)
        assert result is not None
        imported, skipped = result
        assert imported == 0
        assert skipped == 1


# --------------------------------------------------------------------------- #
# sync_beliefs tool
# --------------------------------------------------------------------------- #


class TestSyncBeliefsTool:
    """Unit tests for the sync_beliefs tool using a mocked BridgeClient."""

    def _run(self, body: str, beliefs=None, workspace_path=None) -> str:
        from tests._tool_compat import execute_tool
        return execute_tool(
            "sync_beliefs",
            body,
            beliefs=beliefs,
            workspace_path=workspace_path,
        )

    def test_missing_url_returns_error(self):
        result = self._run("direction: pull")
        assert "error" in result.lower() or "url" in result.lower()

    @patch("team.bridge_client.BridgeClient")
    def test_pull_direction(self, mock_cls, tmp_path):
        board = _board(tmp_path)
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.pull_beliefs.return_value = BeliefBundle(
            source_team="lab-b",
            source_url="http://lab-b",
            beliefs=[{
                "id": "x",
                "claim": "Remote fact is true",
                "author": "bob",
                "confidence": 0.8,
                "evidence": "",
                "status": "accepted",
                "votes_for": ["bob"],
                "votes_against": [],
                "reasons": {},
                "created_at": time.time(),
                "updated_at": time.time(),
            }],
        )
        result = self._run("url: http://lab-b:7001\ndirection: pull", beliefs=board)
        assert "lab-b" in result
        assert "1" in result  # 1 imported
        assert board.count() == 1

    @patch("team.bridge_client.BridgeClient")
    def test_push_direction(self, mock_cls, tmp_path):
        board = _board(tmp_path, members=["alice"])
        board.assert_belief("Local knowledge", "alice")
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.push_beliefs.return_value = (1, 0)
        result = self._run(
            "url: http://remote:7001\ndirection: push\nlocal_team: myteam",
            beliefs=board,
        )
        assert "pushed" in result.lower() or "1" in result

    @patch("team.bridge_client.BridgeClient")
    def test_none_pull_response_reported(self, mock_cls, tmp_path):
        board = _board(tmp_path)
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.pull_beliefs.return_value = None
        result = self._run("url: http://remote:7001\ndirection: pull", beliefs=board)
        assert "not expose" in result.lower() or "not available" in result.lower()

    def test_no_board_returns_error(self, tmp_path):
        result = self._run("url: http://remote:7001", beliefs=None, workspace_path=None)
        assert "error" in result.lower()
