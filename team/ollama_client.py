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
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Iterable, Iterator

import requests

log = logging.getLogger(__name__)


class OllamaError(RuntimeError):
    pass


@dataclass
class ChatMessage:
    role: str  # "system" | "user" | "assistant"
    content: str

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


@dataclass
class TokenUsage:
    """Token counts returned by the LLM backend after a chat completion."""
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        timeout: int = 600,
        max_retries: int = 3,
        retry_backoff: float = 2.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self._session = requests.Session()
        # Populated after every chat() or stream_chat() call.
        self.last_usage: TokenUsage | None = None

    # ----- low level ---------------------------------------------------- #

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    @staticmethod
    def _build_options(
        temperature: float | None,
        top_p: float | None,
        num_ctx: int | None,
        stop: list[str] | None,
    ) -> dict:
        options: dict = {}
        if temperature is not None:
            options["temperature"] = temperature
        if top_p is not None:
            options["top_p"] = top_p
        if num_ctx is not None:
            options["num_ctx"] = num_ctx
        if stop:
            options["stop"] = stop
        return options

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
        options = self._build_options(temperature, top_p, num_ctx, stop)
        payload = {
            "model": model,
            "messages": [m.to_dict() for m in messages],
            "stream": False,
            "options": options,
        }
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                r = self._session.post(
                    self._url("/api/chat"), json=payload, timeout=self.timeout
                )
                if r.status_code >= 500:
                    # 5xx = server-side error — safe to retry because the request
                    # never completed from the server's perspective.
                    raise requests.HTTPError(
                        f"server error {r.status_code}", response=r
                    )
                if r.status_code >= 400:
                    # 4xx = client error (wrong model name, malformed request, …).
                    # These won't self-heal on retry, so raise immediately.
                    raise OllamaError(f"chat failed ({r.status_code}): {r.text}")
                data = r.json()
                content = data.get("message", {}).get("content", "")
                if not content:
                    raise OllamaError(f"chat returned no content: {data}")
                self.last_usage = TokenUsage(
                    prompt_tokens=data.get("prompt_eval_count", 0),
                    completion_tokens=data.get("eval_count", 0),
                )
                return content
            except OllamaError:
                raise  # 4xx and empty-content errors are not retried
            except (
                requests.ConnectionError,
                requests.Timeout,
                requests.HTTPError,
            ) as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    wait = self.retry_backoff**attempt
                    log.warning(
                        "chat: transient error (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1,
                        self.max_retries + 1,
                        wait,
                        exc,
                    )
                    time.sleep(wait)
        raise OllamaError(
            f"chat failed after {self.max_retries + 1} attempt(s): {last_exc}"
        ) from last_exc

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

        Retries the entire request on transient network/server errors, but
        only if no tokens have been yielded yet (a partial stream cannot be
        safely replayed).

        Each yielded string is one raw token chunk from the model.  Callers
        that need the full response should join the chunks themselves.
        """
        options = self._build_options(temperature, top_p, num_ctx, stop)
        payload = {
            "model": model,
            "messages": [m.to_dict() for m in messages],
            "stream": True,
            "options": options,
        }
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            # Track whether we've yielded any tokens during this attempt.
            # Once tokens have been yielded to the caller we cannot retry:
            # the caller already received partial output and restarting the
            # stream would produce duplicate tokens.
            tokens_yielded = False
            try:
                with self._session.post(
                    self._url("/api/chat"),
                    json=payload,
                    stream=True,
                    timeout=self.timeout,
                ) as r:
                    if r.status_code >= 500:
                        raise requests.HTTPError(
                            f"server error {r.status_code}", response=r
                        )
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
                            tokens_yielded = True
                            yield token
                        if data.get("done"):
                            self.last_usage = TokenUsage(
                                prompt_tokens=data.get("prompt_eval_count", 0),
                                completion_tokens=data.get("eval_count", 0),
                            )
                            break
                return  # generator exhausted normally
            except OllamaError:
                raise
            except (
                requests.ConnectionError,
                requests.Timeout,
                requests.HTTPError,
            ) as exc:
                if tokens_yielded:
                    # Cannot retry a partially consumed stream — raise immediately.
                    raise OllamaError(
                        f"stream interrupted after partial output: {exc}"
                    ) from exc
                last_exc = exc
                if attempt < self.max_retries:
                    wait = self.retry_backoff**attempt
                    log.warning(
                        "stream_chat: transient error (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1,
                        self.max_retries + 1,
                        wait,
                        exc,
                    )
                    time.sleep(wait)
        raise OllamaError(
            f"stream_chat failed after {self.max_retries + 1} attempt(s): {last_exc}"
        ) from last_exc


# --------------------------------------------------------------------------- #
# OpenAI-compatible client (works with OpenAI, LM Studio, vLLM, llama.cpp,
# Anthropic via litellm proxy, and any /v1/chat/completions endpoint).
# --------------------------------------------------------------------------- #


def _resolve_api_key(api_key: str | None) -> str | None:
    """Resolve an API key value, expanding ``env:<VAR>`` references."""
    if api_key is None:
        return None
    if api_key.startswith("env:"):
        return os.environ.get(api_key[4:])
    return api_key


class OpenAICompatClient:
    """Thin client for OpenAI-compatible ``/v1/chat/completions`` endpoints.

    Mirrors the ``OllamaClient`` interface so ``Member`` can swap backends
    transparently.  Set ``api_key`` to a literal value or ``env:<VAR>`` to
    read from an environment variable at runtime.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout: int = 600,
        max_retries: int = 3,
        retry_backoff: float = 2.0,
    ):
        self.base_url = base_url.rstrip("/")
        self._api_key = _resolve_api_key(api_key)
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self._session = requests.Session()
        if self._api_key:
            self._session.headers["Authorization"] = f"Bearer {self._api_key}"
        self.last_usage: TokenUsage | None = None

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    @staticmethod
    def _build_options(
        temperature: float | None,
        top_p: float | None,
        num_ctx: int | None,
        stop: list[str] | None,
    ) -> dict:
        opts: dict = {}
        if temperature is not None:
            opts["temperature"] = temperature
        if top_p is not None:
            opts["top_p"] = top_p
        if num_ctx is not None:
            opts["max_tokens"] = num_ctx
        if stop:
            opts["stop"] = stop
        return opts

    def ping(self) -> bool:
        try:
            r = self._session.get(self._url("/v1/models"), timeout=5)
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
            f"OpenAI-compat endpoint at {self.base_url} not ready after {deadline_seconds}s"
        )

    def list_models(self) -> list[str]:
        r = self._session.get(self._url("/v1/models"), timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        return [m.get("id", "") for m in data.get("data", [])]

    def ensure_model(self, model: str, timeout: int | None = None) -> None:
        """Verify the model is available; cannot pull via OpenAI-compat endpoints."""
        try:
            available = self.list_models()
        except Exception:
            # If listing fails (e.g. endpoint doesn't expose /v1/models), skip.
            log.warning("openai_compat: could not list models at %s, skipping check", self.base_url)
            return
        if available and model not in available:
            log.warning(
                "openai_compat: model %r not found in %s; proceeding anyway",
                model,
                available[:5],
            )

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
        opts = self._build_options(temperature, top_p, num_ctx, stop)
        payload = {
            "model": model,
            "messages": [m.to_dict() for m in messages],
            "stream": False,
            **opts,
        }
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                r = self._session.post(
                    self._url("/v1/chat/completions"), json=payload, timeout=self.timeout
                )
                if r.status_code >= 500:
                    raise requests.HTTPError(f"server error {r.status_code}", response=r)
                if r.status_code >= 400:
                    raise OllamaError(f"chat failed ({r.status_code}): {r.text}")
                data = r.json()
                content = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                if not content:
                    raise OllamaError(f"chat returned no content: {data}")
                usage = data.get("usage") or {}
                self.last_usage = TokenUsage(
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                )
                return content
            except OllamaError:
                raise
            except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    wait = self.retry_backoff**attempt
                    log.warning(
                        "openai_compat chat: transient error (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1, self.max_retries + 1, wait, exc,
                    )
                    time.sleep(wait)
        raise OllamaError(
            f"chat failed after {self.max_retries + 1} attempt(s): {last_exc}"
        ) from last_exc

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
        """Yield content tokens as they arrive via OpenAI SSE streaming."""
        opts = self._build_options(temperature, top_p, num_ctx, stop)
        payload = {
            "model": model,
            "messages": [m.to_dict() for m in messages],
            "stream": True,
            "stream_options": {"include_usage": True},
            **opts,
        }
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            tokens_yielded = False
            try:
                with self._session.post(
                    self._url("/v1/chat/completions"),
                    json=payload,
                    stream=True,
                    timeout=self.timeout,
                ) as r:
                    if r.status_code >= 500:
                        raise requests.HTTPError(f"server error {r.status_code}", response=r)
                    if r.status_code >= 400:
                        raise OllamaError(f"chat failed ({r.status_code}): {r.text}")
                    for line in r.iter_lines():
                        if not line:
                            continue
                        # SSE lines are prefixed with "data: "
                        text = line.decode("utf-8") if isinstance(line, bytes) else line
                        if not text.startswith("data:"):
                            continue
                        payload_str = text[len("data:"):].strip()
                        if payload_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(payload_str)
                        except json.JSONDecodeError:
                            continue
                        delta = (
                            chunk.get("choices", [{}])[0]
                            .get("delta", {})
                            .get("content", "")
                        )
                        if delta:
                            tokens_yielded = True
                            yield delta
                        usage = chunk.get("usage")
                        if usage:
                            self.last_usage = TokenUsage(
                                prompt_tokens=usage.get("prompt_tokens", 0),
                                completion_tokens=usage.get("completion_tokens", 0),
                            )
                return
            except OllamaError:
                raise
            except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
                if tokens_yielded:
                    raise OllamaError(
                        f"stream interrupted after partial output: {exc}"
                    ) from exc
                last_exc = exc
                if attempt < self.max_retries:
                    wait = self.retry_backoff**attempt
                    log.warning(
                        "openai_compat stream_chat: transient error (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1, self.max_retries + 1, wait, exc,
                    )
                    time.sleep(wait)
        raise OllamaError(
            f"stream_chat failed after {self.max_retries + 1} attempt(s): {last_exc}"
        ) from last_exc

