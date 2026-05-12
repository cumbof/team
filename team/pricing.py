"""LLM pricing table and cost estimation helpers.

Prices are expressed as **USD per 1 million tokens** and were accurate at the
time of this release.  Model providers update their prices regularly — check
the provider's pricing page for the latest figures and update this table as
needed.

Usage::

    from team.pricing import estimate_cost, format_cost

    cost = estimate_cost("gpt-4o", prompt_tokens=1500, completion_tokens=300)
    if cost is not None:
        print(format_cost(cost))   # "$0.0158"
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Pricing table
# ---------------------------------------------------------------------------
# Keys are normalised model prefixes (lowercase, no version dates).
# Values are dicts with:
#   prompt     — USD per 1 000 000 prompt tokens
#   completion — USD per 1 000 000 completion tokens
#
# Matching is done by: exact key → longest prefix contained in the model name.
# Model names coming from YAML may carry version suffixes (e.g.
# "gpt-4o-2024-08-06") so prefix/substring matching is essential.
# ---------------------------------------------------------------------------

PRICES: dict[str, dict[str, float]] = {
    # OpenAI -------------------------------------------------------------------
    "gpt-4.1":             {"prompt":  2.00, "completion":  8.00},
    "gpt-4.1-mini":        {"prompt":  0.40, "completion":  1.60},
    "gpt-4.1-nano":        {"prompt":  0.10, "completion":  0.40},
    "gpt-4o":              {"prompt":  2.50, "completion": 10.00},
    "gpt-4o-mini":         {"prompt":  0.15, "completion":  0.60},
    "gpt-4-turbo":         {"prompt": 10.00, "completion": 30.00},
    "gpt-4":               {"prompt": 30.00, "completion": 60.00},
    "gpt-3.5-turbo":       {"prompt":  0.50, "completion":  1.50},
    "o3":                  {"prompt": 10.00, "completion": 40.00},
    "o3-mini":             {"prompt":  1.10, "completion":  4.40},
    "o1":                  {"prompt": 15.00, "completion": 60.00},
    "o1-mini":             {"prompt":  3.00, "completion": 12.00},

    # Anthropic ----------------------------------------------------------------
    "claude-opus-4":       {"prompt": 15.00, "completion": 75.00},
    "claude-sonnet-4":     {"prompt":  3.00, "completion": 15.00},
    "claude-3-5-sonnet":   {"prompt":  3.00, "completion": 15.00},
    "claude-3-5-haiku":    {"prompt":  0.80, "completion":  4.00},
    "claude-3-opus":       {"prompt": 15.00, "completion": 75.00},
    "claude-3-sonnet":     {"prompt":  3.00, "completion": 15.00},
    "claude-3-haiku":      {"prompt":  0.25, "completion":  1.25},

    # Google -------------------------------------------------------------------
    "gemini-2.0-flash":    {"prompt":  0.10, "completion":  0.40},
    "gemini-1.5-pro":      {"prompt":  1.25, "completion":  5.00},
    "gemini-1.5-flash":    {"prompt":  0.075,"completion":  0.30},

    # Mistral (cloud) ----------------------------------------------------------
    "mistral-large":       {"prompt":  2.00, "completion":  6.00},
    "mistral-medium":      {"prompt":  2.70, "completion":  8.10},
    "mistral-small":       {"prompt":  0.20, "completion":  0.60},
    "codestral":           {"prompt":  0.30, "completion":  0.90},

    # Meta / Llama (hosted via cloud providers) --------------------------------
    "llama-3.1-405b":      {"prompt":  3.00, "completion":  3.00},
    "llama-3.1-70b":       {"prompt":  0.90, "completion":  0.90},
    "llama-3.1-8b":        {"prompt":  0.18, "completion":  0.18},
    "llama-3-70b":         {"prompt":  0.59, "completion":  0.79},
    "llama-3-8b":          {"prompt":  0.20, "completion":  0.20},
}

# Normalise a model string: lowercase + strip common Ollama tag suffixes like
# ":latest", ":8b-instruct-q4_K_M", etc.
_TAG_RE = re.compile(r":.*$")


def _normalise(model: str) -> str:
    return _TAG_RE.sub("", model).lower().strip()


def lookup_price(model: str) -> dict[str, float] | None:
    """Return ``{prompt: $/1M, completion: $/1M}`` for *model*, or ``None``.

    Matching strategy (first match wins):
    1. Exact match after normalisation.
    2. Longest key that is a prefix of the normalised model name.
    3. Longest key that appears as a substring of the normalised model name.
    """
    key = _normalise(model)

    if key in PRICES:
        return PRICES[key]

    # Prefix match — "gpt-4o-2024-08-06" → "gpt-4o"
    prefix_match = max(
        (k for k in PRICES if key.startswith(k)),
        key=len,
        default=None,
    )
    if prefix_match:
        return PRICES[prefix_match]

    # Substring match — "accounts/fireworks/models/llama-3.1-70b" → "llama-3.1-70b"
    substr_match = max(
        (k for k in PRICES if k in key),
        key=len,
        default=None,
    )
    if substr_match:
        return PRICES[substr_match]

    return None


def estimate_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    *,
    is_local: bool = False,
) -> float | None:
    """Return estimated USD cost for *model* given token counts.

    Parameters
    ----------
    model:
        The model name exactly as it appears in the YAML (e.g. ``"gpt-4o"``).
    prompt_tokens:
        Number of prompt (input) tokens consumed.
    completion_tokens:
        Number of completion (output) tokens produced.
    is_local:
        If ``True``, the model runs locally (Ollama default backend) and the
        cost is always **$0.00**.

    Returns
    -------
    float | None
        Estimated cost in USD, or ``None`` if *model* is not found in the
        pricing table and *is_local* is ``False``.
    """
    if is_local:
        return 0.0

    prices = lookup_price(model)
    if prices is None:
        return None

    return (
        prompt_tokens * prices["prompt"] / 1_000_000
        + completion_tokens * prices["completion"] / 1_000_000
    )


def format_cost(cost: float | None) -> str:
    """Format a USD cost value for display.

    * ``None``   → ``"?"``  (unknown pricing)
    * ``0.0``    → ``"$0.00 (local)"``
    * ``< 0.01`` → ``"$0.000123"`` (6 significant decimal places)
    * ``≥ 0.01`` → ``"$1.23"`` (2 decimal places)
    """
    if cost is None:
        return "?"
    if cost == 0.0:
        return "$0.00 (local)"
    if cost < 0.01:
        return f"${cost:.6f}"
    return f"${cost:.2f}"
