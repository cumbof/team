"""Thin Ollama HTTP client.

The orchestrator only needs three operations:

* ``ping`` — verify the daemon is reachable.
* ``pull`` — make sure a model is present (streamed for progress).
* ``chat`` — send a chat-completion request.

We deliberately avoid taking a hard dependency on the official ``ollama``
Python SDK so that the server can be any Ollama-compatible HTTP endpoint
(useful for running tests against a fake).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Iterable, Iterator

import requests


class OllamaError(RuntimeError):
    pass


@dataclass
class ChatMessage:
    role: str  # "system" | "user" | "assistant"
    content: str

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


class OllamaClient:
    def __init__(self, base_url: str, timeout: int = 600):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()

    # ----- low level ---------------------------------------------------- #

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    # ----- health / lifecycle ------------------------------------------- #

    def ping(self) -> bool:
        try:
            r = self._session.get(self._url("/api/tags"), timeout=5)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def wait_ready(self, deadline_seconds: int = 120, interval: float = 1.0) -> None:
        start = time.monotonic()
        while time.monotonic() - start < deadline_seconds:
            if self.ping():
                return
            time.sleep(interval)
        raise OllamaError(
            f"Ollama at {self.base_url} not ready after {deadline_seconds}s"
        )

    def list_models(self) -> list[str]:
        r = self._session.get(self._url("/api/tags"), timeout=self.timeout)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]

    # ----- model management --------------------------------------------- #

    def pull(self, model: str, timeout: int | None = None) -> Iterator[dict]:
        """Stream pull progress events. Yields dicts with at least ``status``."""
        timeout = timeout or self.timeout
        with self._session.post(
            self._url("/api/pull"),
            json={"name": model, "stream": True},
            stream=True,
            timeout=timeout,
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    def ensure_model(self, model: str, timeout: int | None = None) -> None:
        if model in self.list_models():
            return
        for _ in self.pull(model, timeout=timeout):
            pass
        if model not in self.list_models():
            raise OllamaError(f"failed to pull model {model!r}")

    # ----- chat --------------------------------------------------------- #

    def chat(
        self,
        model: str,
        messages: Iterable[ChatMessage],
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        num_ctx: int | None = None,
        stop: list[str] | None = None,
    ) -> str:
        options: dict = {}
        if temperature is not None:
            options["temperature"] = temperature
        if top_p is not None:
            options["top_p"] = top_p
        if num_ctx is not None:
            options["num_ctx"] = num_ctx
        if stop:
            options["stop"] = stop

        payload = {
            "model": model,
            "messages": [m.to_dict() for m in messages],
            "stream": False,
            "options": options,
        }
        r = self._session.post(
            self._url("/api/chat"), json=payload, timeout=self.timeout
        )
        if r.status_code >= 400:
            raise OllamaError(f"chat failed ({r.status_code}): {r.text}")
        data = r.json()
        msg = data.get("message", {})
        content = msg.get("content", "")
        if not content:
            raise OllamaError(f"chat returned no content: {data}")
        return content

    def stream_chat(
        self,
        model: str,
        messages: Iterable[ChatMessage],
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        num_ctx: int | None = None,
        stop: list[str] | None = None,
    ) -> Iterator[str]:
        """Yield content tokens as they stream from the Ollama API.

        Each yielded string is one raw token chunk from the model.  Callers
        that need the full response should join the chunks themselves.
        """
        options: dict = {}
        if temperature is not None:
            options["temperature"] = temperature
        if top_p is not None:
            options["top_p"] = top_p
        if num_ctx is not None:
            options["num_ctx"] = num_ctx
        if stop:
            options["stop"] = stop

        payload = {
            "model": model,
            "messages": [m.to_dict() for m in messages],
            "stream": True,
            "options": options,
        }
        with self._session.post(
            self._url("/api/chat"), json=payload, stream=True, timeout=self.timeout
        ) as r:
            if r.status_code >= 400:
                raise OllamaError(f"chat failed ({r.status_code}): {r.text}")
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                token = data.get("message", {}).get("content", "")
                if token:
                    yield token
                if data.get("done"):
                    break
