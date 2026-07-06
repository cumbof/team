"""``expert`` server — delegate_to_expert (external cloud LLM)."""

from __future__ import annotations

import logging
import os
from typing import Annotated

import requests
from pydantic import Field

from team.mcp.builtin import BuiltinContext
from team.mcp.builtin._util import offload, truncate

log = logging.getLogger(__name__)

#: Supported external providers and their defaults.
_EXPERT_PROVIDERS: dict[str, dict] = {
    "openai": {
        "env_key": "OPENAI_API_KEY",
        "default_model": "gpt-4o",
        "api_url": "https://api.openai.com/v1/chat/completions",
    },
    "anthropic": {
        "env_key": "ANTHROPIC_API_KEY",
        "default_model": "claude-opus-4-5",
        "api_url": "https://api.anthropic.com/v1/messages",
    },
    "google": {
        "env_key": "GOOGLE_API_KEY",
        "default_model": "gemini-1.5-pro",
        "api_url": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
    },
}


def _call_openai(api_url, api_key, model, prompt, max_tokens, temperature, timeout) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    r = requests.post(api_url, json=payload, headers=headers, timeout=timeout)
    r.raise_for_status()
    choices = r.json().get("choices") or []
    if not choices:
        return "ERROR: OpenAI returned no choices in response"
    return choices[0].get("message", {}).get("content", "").strip()


def _call_anthropic(api_url, api_key, model, prompt, max_tokens, temperature, timeout) -> str:
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    r = requests.post(api_url, json=payload, headers=headers, timeout=timeout)
    r.raise_for_status()
    content = r.json().get("content") or []
    texts = [block.get("text", "") for block in content if block.get("type") == "text"]
    return "\n".join(texts).strip() or "ERROR: Anthropic returned no text content"


def _call_google(api_url_template, api_key, model, prompt, max_tokens, temperature, timeout) -> str:
    url = api_url_template.format(model=model)
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": max_tokens, "temperature": temperature},
    }
    r = requests.post(url, json=payload, params={"key": api_key}, timeout=timeout)
    r.raise_for_status()
    candidates = r.json().get("candidates") or []
    if not candidates:
        return "ERROR: Google returned no candidates in response"
    parts = candidates[0].get("content", {}).get("parts") or []
    texts = [p.get("text", "") for p in parts]
    return "\n".join(texts).strip() or "ERROR: Google returned no text content"


def build(ctx: BuiltinContext) -> "object":
    from mcp.server.fastmcp import FastMCP

    srv = FastMCP("expert")
    timeout = ctx.tool_timeout

    @srv.tool()
    @offload
    def delegate_to_expert(
        provider: Annotated[
            str, Field(description="Cloud provider: openai | anthropic | google.")
        ],
        prompt: Annotated[str, Field(description="The prompt or question to send to the expert model.")],
        model: Annotated[
            str, Field(description="Model identifier (optional; provider default used if omitted).")
        ] = "",
        max_tokens: Annotated[
            int, Field(description="Maximum tokens in the response (default 2048).")
        ] = 2048,
        temperature: Annotated[
            float, Field(description="Sampling temperature 0–2 (default 0.2).")
        ] = 0.2,
    ) -> str:
        """Send a prompt to an external cloud LLM (OpenAI, Anthropic, Google) for expert assistance.

        WARNING: the prompt text is sent to an external API. Requires the
        provider's API key in the environment (OPENAI_API_KEY, ANTHROPIC_API_KEY,
        or GOOGLE_API_KEY).
        """
        name = (provider or "").strip().lower()
        if not name:
            return "ERROR: provide provider: <name> — supported: " + ", ".join(_EXPERT_PROVIDERS)
        if name not in _EXPERT_PROVIDERS:
            return f"ERROR: unknown provider {name!r}. Supported: {', '.join(_EXPERT_PROVIDERS)}"
        info = _EXPERT_PROVIDERS[name]
        api_key = os.environ.get(info["env_key"], "").strip()
        if not api_key:
            return (
                f"ERROR: {info['env_key']} environment variable is not set. "
                f"Export it on the host before running team to use provider {name!r}."
            )
        mdl = (model or "").strip() or info["default_model"]
        text = (prompt or "").strip()
        if not text:
            return "ERROR: provide a prompt"
        log.info("delegate_to_expert: calling %s/%s (max_tokens=%d)", name, mdl, max_tokens)
        try:
            if name == "openai":
                reply = _call_openai(info["api_url"], api_key, mdl, text, max_tokens, temperature, timeout)
            elif name == "anthropic":
                reply = _call_anthropic(info["api_url"], api_key, mdl, text, max_tokens, temperature, timeout)
            else:
                reply = _call_google(info["api_url"], api_key, mdl, text, max_tokens, temperature, timeout)
        except requests.exceptions.Timeout:
            return f"ERROR: request to {name} timed out after {timeout}s"
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "?"
            return f"ERROR: {name} API returned HTTP {status}: {exc}"
        except Exception as exc:  # noqa: BLE001
            return f"ERROR: delegate_to_expert failed ({name}): {exc}"
        if reply.startswith("ERROR"):
            return reply
        return truncate(f"[Expert response from {name}/{mdl}]\n\n{reply}")

    return srv
