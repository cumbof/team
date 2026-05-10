"""Tests for OllamaClient — specifically the new stream_chat() method.

No real HTTP server is needed: we patch requests.Session.post with a mock
that returns NDJSON lines in the same format as the Ollama streaming API.
"""

from __future__ import annotations

import json
from typing import Iterator
from unittest.mock import MagicMock, patch

import pytest

from team.ollama_client import ChatMessage, OllamaClient, OllamaError


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _stream_response(tokens: list[str], *, error_status: int = 200) -> MagicMock:
    """Return a context-manager mock that streams NDJSON chunks."""
    if error_status >= 400:
        mock = MagicMock()
        mock.status_code = error_status
        mock.text = "server error"
        mock.__enter__ = lambda s: s
        mock.__exit__ = MagicMock(return_value=False)
        return mock

    lines: list[bytes] = []
    for i, token in enumerate(tokens):
        done = i == len(tokens) - 1
        chunk = {"message": {"role": "assistant", "content": token}, "done": done}
        lines.append(json.dumps(chunk).encode())

    mock = MagicMock()
    mock.status_code = 200
    mock.iter_lines.return_value = iter(lines)
    mock.__enter__ = lambda s: s
    mock.__exit__ = MagicMock(return_value=False)
    return mock


def _client() -> OllamaClient:
    return OllamaClient(base_url="http://localhost:11434", timeout=10)


# --------------------------------------------------------------------------- #
# stream_chat tests
# --------------------------------------------------------------------------- #


def test_stream_chat_yields_all_tokens() -> None:
    client = _client()
    tokens = ["Hello", ", ", "world", "!"]
    with patch.object(client._session, "post", return_value=_stream_response(tokens)):
        result = list(client.stream_chat("m", [ChatMessage("user", "hi")]))
    assert result == tokens


def test_stream_chat_assembles_full_content() -> None:
    client = _client()
    tokens = ["The", " answer", " is", " 42", "."]
    with patch.object(client._session, "post", return_value=_stream_response(tokens)):
        content = "".join(client.stream_chat("m", [ChatMessage("user", "?")]))
    assert content == "The answer is 42."


def test_stream_chat_raises_on_http_error() -> None:
    client = _client()
    with patch.object(
        client._session, "post", return_value=_stream_response([], error_status=500)
    ):
        with pytest.raises(OllamaError, match="chat failed"):
            list(client.stream_chat("m", [ChatMessage("user", "hi")]))


def test_stream_chat_skips_empty_token_chunks() -> None:
    """Chunks with empty content (e.g. the final done=True chunk) are skipped."""
    client = _client()
    lines = [
        json.dumps({"message": {"content": "word"}, "done": False}).encode(),
        json.dumps({"message": {"content": ""}, "done": True}).encode(),
    ]
    mock = MagicMock()
    mock.status_code = 200
    mock.iter_lines.return_value = iter(lines)
    mock.__enter__ = lambda s: s
    mock.__exit__ = MagicMock(return_value=False)

    with patch.object(client._session, "post", return_value=mock):
        result = list(client.stream_chat("m", [ChatMessage("user", "hi")]))
    assert result == ["word"]


def test_stream_chat_stops_at_done_flag() -> None:
    """No chunks are consumed after a done=True line."""
    client = _client()
    # The done=True chunk may still carry a content token; we yield it
    # and then stop — the third line must never be reached.
    lines = [
        json.dumps({"message": {"content": "a"}, "done": False}).encode(),
        json.dumps({"message": {"content": "b"}, "done": True}).encode(),
        # Would only be consumed if we didn't stop at done=True:
        json.dumps({"message": {"content": "c"}, "done": False}).encode(),
    ]
    mock = MagicMock()
    mock.status_code = 200
    # Wrap in a list that raises if exhausted past index 2.
    consumed: list[int] = []

    def _iter():
        for i, line in enumerate(lines):
            consumed.append(i)
            yield line

    mock.iter_lines.return_value = _iter()
    mock.__enter__ = lambda s: s
    mock.__exit__ = MagicMock(return_value=False)

    with patch.object(client._session, "post", return_value=mock):
        result = list(client.stream_chat("m", [ChatMessage("user", "hi")]))

    assert result == ["a", "b"]  # "b" is the last token (in the done chunk)
    assert 2 not in consumed      # the third line was never read
