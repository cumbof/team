"""``web`` server — web_search, read_url."""

from __future__ import annotations

import logging
import re
from typing import Annotated

import requests
from pydantic import Field

from team.mcp.builtin import BuiltinContext
from team.mcp.builtin._util import _MAX_SEARCH_OUTPUT, truncate

log = logging.getLogger(__name__)


def build(ctx: BuiltinContext) -> "object":
    from mcp.server.fastmcp import FastMCP

    srv = FastMCP("web")
    timeout = ctx.tool_timeout

    @srv.tool()
    def web_search(
        query: Annotated[str, Field(description="Search terms.")],
    ) -> str:
        """Search the web via the DuckDuckGo instant-answer API and return a summary."""
        q = query.strip()
        if not q:
            return "ERROR: provide query: <search terms>"
        try:
            r = requests.get(
                "https://api.duckduckgo.com/",
                params={"q": q, "format": "json", "no_html": "1", "skip_disambig": "1"},
                timeout=timeout,
                headers={"User-Agent": "team-agent/0.1"},
            )
            r.raise_for_status()
            data = r.json()
        except Exception as exc:  # noqa: BLE001
            return f"ERROR: web_search failed: {exc}"

        parts: list[str] = []
        if data.get("Heading"):
            parts.append(f"**{data['Heading']}**")
        if data.get("AbstractText"):
            parts.append(data["AbstractText"])
        if data.get("Answer"):
            parts.append(f"Answer: {data['Answer']}")
        for topic in data.get("RelatedTopics", [])[:8]:
            if isinstance(topic, dict) and topic.get("Text"):
                parts.append(f"- {topic['Text']}")
        if not parts:
            return (
                f"No instant answer found for: {q!r}\n"
                "Tip: DuckDuckGo instant answers work best for factual queries. "
                "For broader results, try read_url with a specific page URL."
            )
        return truncate("\n".join(parts), _MAX_SEARCH_OUTPUT)

    @srv.tool()
    def read_url(
        url: Annotated[str, Field(description="URL to fetch.")],
    ) -> str:
        """Fetch a URL and return its text content (HTML tags stripped)."""
        target = url.strip()
        if not target:
            return "ERROR: provide url: <url>"
        try:
            r = requests.get(
                target, timeout=timeout, headers={"User-Agent": "team-agent/0.1"}
            )
            r.raise_for_status()
            content_type = r.headers.get("Content-Type", "")
            if "text/html" in content_type:
                text = re.sub(r"<style[^>]*>.*?</style>", " ", r.text, flags=re.DOTALL)
                text = re.sub(r"<script[^>]*>.*?</script>", " ", text, flags=re.DOTALL)
                text = re.sub(r"<[^>]+>", " ", text)
                text = re.sub(r"&[a-z]+;", " ", text)
                text = re.sub(r"\s{2,}", " ", text).strip()
            else:
                text = r.text
            return truncate(text)
        except Exception as exc:  # noqa: BLE001
            return f"ERROR: read_url failed: {exc}"

    return srv
