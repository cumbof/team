"""Tests for OllamaClient — specifically the new stream_chat() method.

No real HTTP server is needed: we patch requests.Session.post with a mock
that returns NDJSON lines in the same format as the Ollama streaming API.
"""

from __future__ import annotations

import json
from typing import Iterator
from unittest.mock import MagicMock, call, patch

import pytest
import requests

from team.ollama_client import ChatMessage, OllamaClient, OllamaError


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _stream_response(tokens: list[str], *, status: int = 200) -> MagicMock:
    """Return a context-manager mock that streams NDJSON chunks."""
    mock = MagicMock()
    mock.status_code = status
    mock.text = "error body" if status >= 400 else ""
    mock.__enter__ = lambda s: s
    mock.__exit__ = MagicMock(return_value=False)
    if status < 400:
        lines: list[bytes] = []
        for i, token in enumerate(tokens):
            done = i == len(tokens) - 1
            chunk = {"message": {"role": "assistant", "content": token}, "done": done}
            lines.append(json.dumps(chunk).encode())
        mock.iter_lines.return_value = iter(lines)
    return mock


def _chat_response(content: str, *, status: int = 200) -> MagicMock:
    """Return a plain (non-streaming) mock response for chat()."""
    mock = MagicMock()
    mock.status_code = status
    mock.text = "error body" if status >= 400 else ""
    if status < 400:
        mock.json.return_value = {"message": {"role": "assistant", "content": content}}
    return mock


def _client(max_retries: int = 0) -> OllamaClient:
    """Build a client with no sleep overhead by default (max_retries=0)."""
    return OllamaClient(
        base_url="http://localhost:11434",
        timeout=10,
        max_retries=max_retries,
        retry_backoff=0.001,  # near-zero sleep for tests that do retry
    )


# --------------------------------------------------------------------------- #
# stream_chat tests (basic)
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


def test_stream_chat_raises_on_4xx() -> None:
    client = _client()
    with patch.object(
        client._session, "post", return_value=_stream_response([], status=400)
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
    lines = [
        json.dumps({"message": {"content": "a"}, "done": False}).encode(),
        json.dumps({"message": {"content": "b"}, "done": True}).encode(),
        # Would only be consumed if we didn't stop at done=True:
        json.dumps({"message": {"content": "c"}, "done": False}).encode(),
    ]
    mock = MagicMock()
    mock.status_code = 200
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


# --------------------------------------------------------------------------- #
# chat() tests (basic)
# --------------------------------------------------------------------------- #


def test_chat_returns_content() -> None:
    client = _client()
    with patch.object(client._session, "post", return_value=_chat_response("hello")):
        result = client.chat("m", [ChatMessage("user", "hi")])
    assert result == "hello"


def test_chat_raises_on_4xx_no_retry() -> None:
    """4xx errors are not retried — raised immediately."""
    client = _client(max_retries=3)
    with patch.object(
        client._session, "post", return_value=_chat_response("", status=400)
    ) as mock_post:
        with pytest.raises(OllamaError, match="chat failed"):
            client.chat("m", [ChatMessage("user", "hi")])
    assert mock_post.call_count == 1  # no retries


# --------------------------------------------------------------------------- #
# Retry tests — chat()
# --------------------------------------------------------------------------- #


def test_chat_retries_on_connection_error_then_succeeds() -> None:
    client = _client(max_retries=2)
    fail = requests.ConnectionError("refused")
    with patch.object(
        client._session,
        "post",
        side_effect=[fail, fail, _chat_response("ok")],
    ):
        with patch("team.ollama_client.time.sleep"):
            result = client.chat("m", [ChatMessage("user", "hi")])
    assert result == "ok"


def test_chat_raises_after_exhausting_retries() -> None:
    client = _client(max_retries=2)
    fail = requests.ConnectionError("down")
    with patch.object(client._session, "post", side_effect=[fail, fail, fail]):
        with patch("team.ollama_client.time.sleep"):
            with pytest.raises(OllamaError, match="3 attempt"):
                client.chat("m", [ChatMessage("user", "hi")])


def test_chat_retries_on_5xx() -> None:
    """HTTP 5xx from the server is treated as a transient error and retried."""
    client = _client(max_retries=1)
    with patch.object(
        client._session,
        "post",
        side_effect=[_chat_response("", status=503), _chat_response("recovered")],
    ):
        with patch("team.ollama_client.time.sleep"):
            result = client.chat("m", [ChatMessage("user", "hi")])
    assert result == "recovered"


def test_chat_sleep_uses_backoff() -> None:
    """Retry wait times follow retry_backoff ** attempt."""
    client = OllamaClient(
        base_url="http://localhost:11434",
        timeout=10,
        max_retries=2,
        retry_backoff=3.0,
    )
    fail = requests.ConnectionError("x")
    with patch.object(
        client._session,
        "post",
        side_effect=[fail, fail, _chat_response("done")],
    ) as _mock:
        with patch("team.ollama_client.time.sleep") as mock_sleep:
            client.chat("m", [ChatMessage("user", "hi")])
    # attempt 0 → sleep(3.0**0 = 1.0), attempt 1 → sleep(3.0**1 = 3.0)
    assert mock_sleep.call_args_list == [call(1.0), call(3.0)]


# --------------------------------------------------------------------------- #
# Retry tests — stream_chat()
# --------------------------------------------------------------------------- #


def test_stream_chat_retries_on_connection_error_before_tokens() -> None:
    client = _client(max_retries=1)
    fail = requests.ConnectionError("refused")
    with patch.object(
        client._session,
        "post",
        side_effect=[fail, _stream_response(["hi"])],
    ):
        with patch("team.ollama_client.time.sleep"):
            result = list(client.stream_chat("m", [ChatMessage("user", "hello")]))
    assert result == ["hi"]


def test_stream_chat_no_retry_after_partial_tokens() -> None:
    """If tokens have already been yielded, a mid-stream error is terminal."""
    client = _client(max_retries=2)
    tokens = ["a", "b"]
    # First response yields one token then raises ConnectionError mid-stream.
    mock = MagicMock()
    mock.status_code = 200
    mock.__enter__ = lambda s: s
    mock.__exit__ = MagicMock(return_value=False)

    def _broken_iter():
        yield json.dumps({"message": {"content": "a"}, "done": False}).encode()
        raise requests.ConnectionError("dropped mid-stream")

    mock.iter_lines.return_value = _broken_iter()
    post_calls = [0]

    def _side_effect(*args, **kwargs):
        post_calls[0] += 1
        return mock

    with patch.object(client._session, "post", side_effect=_side_effect):
        gen = client.stream_chat("m", [ChatMessage("user", "hi")])
        collected = []
        with pytest.raises(OllamaError, match="interrupted"):
            for token in gen:
                collected.append(token)

    # "a" was yielded before the error; no retry was attempted
    assert collected == ["a"]
    assert post_calls[0] == 1  # only one HTTP call was made


def test_stream_chat_raises_after_exhausting_retries() -> None:
    client = _client(max_retries=2)
    fail = requests.ConnectionError("down")
    with patch.object(client._session, "post", side_effect=[fail, fail, fail]):
        with patch("team.ollama_client.time.sleep"):
            with pytest.raises(OllamaError, match="3 attempt"):
                list(client.stream_chat("m", [ChatMessage("user", "hi")]))
