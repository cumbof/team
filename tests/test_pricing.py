"""Tests for team/pricing.py — cost estimation helpers."""

from __future__ import annotations

import pytest

from team.pricing import (
    PRICES,
    estimate_cost,
    format_cost,
    lookup_price,
)


# --------------------------------------------------------------------------- #
# lookup_price
# --------------------------------------------------------------------------- #


class TestLookupPrice:
    def test_exact_match(self):
        result = lookup_price("gpt-4o")
        assert result is not None
        assert result["prompt"] == 2.50
        assert result["completion"] == 10.00

    def test_exact_match_case_insensitive(self):
        assert lookup_price("GPT-4O") is not None

    def test_prefix_match_versioned(self):
        # "gpt-4o-2024-08-06" should match "gpt-4o"
        result = lookup_price("gpt-4o-2024-08-06")
        assert result is not None
        assert result["prompt"] == 2.50

    def test_prefix_match_mini_is_precise(self):
        # "gpt-4o-mini" must NOT fall through to "gpt-4o"
        result = lookup_price("gpt-4o-mini")
        assert result is not None
        assert result["prompt"] == 0.15

    def test_prefix_match_mini_versioned(self):
        result = lookup_price("gpt-4o-mini-2024-07-18")
        assert result is not None
        assert result["prompt"] == 0.15

    def test_ollama_tag_stripped(self):
        # Local Ollama tags like ":latest" or ":8b-instruct" are stripped before lookup
        result = lookup_price("gpt-4o:latest")
        assert result is not None

    def test_substring_match(self):
        # A path-style model name as used by some providers
        result = lookup_price("accounts/fireworks/models/llama-3.1-70b-instruct")
        assert result is not None
        assert "llama-3.1-70b" in str(result) or result["prompt"] == pytest.approx(0.90)

    def test_unknown_model_returns_none(self):
        assert lookup_price("totally-unknown-model-xyz") is None

    def test_claude_versioned(self):
        result = lookup_price("claude-3-5-sonnet-20241022")
        assert result is not None
        assert result["prompt"] == 3.00

    def test_all_price_keys_are_in_prices(self):
        # Sanity check: every entry in PRICES has both required keys
        for model_key, prices in PRICES.items():
            assert "prompt" in prices, f"{model_key} missing 'prompt'"
            assert "completion" in prices, f"{model_key} missing 'completion'"
            assert prices["prompt"] >= 0
            assert prices["completion"] >= 0


# --------------------------------------------------------------------------- #
# estimate_cost
# --------------------------------------------------------------------------- #


class TestEstimateCost:
    def test_known_model_computes_correctly(self):
        # gpt-4o: $2.50/1M prompt + $10.00/1M completion
        # 1000 prompt + 500 completion = 0.0025 + 0.005 = $0.0075
        cost = estimate_cost("gpt-4o", 1000, 500)
        assert cost == pytest.approx(0.0075)

    def test_zero_tokens_returns_zero(self):
        cost = estimate_cost("gpt-4o", 0, 0)
        assert cost == pytest.approx(0.0)

    def test_only_prompt_tokens(self):
        # 1_000_000 prompt tokens = $2.50 for gpt-4o
        cost = estimate_cost("gpt-4o", 1_000_000, 0)
        assert cost == pytest.approx(2.50)

    def test_only_completion_tokens(self):
        # 1_000_000 completion tokens = $10.00 for gpt-4o
        cost = estimate_cost("gpt-4o", 0, 1_000_000)
        assert cost == pytest.approx(10.00)

    def test_unknown_model_returns_none(self):
        cost = estimate_cost("model-does-not-exist", 1000, 500)
        assert cost is None

    def test_is_local_returns_zero(self):
        # Local backend → always $0 regardless of model name
        cost = estimate_cost("llama3", 99999, 99999, is_local=True)
        assert cost == 0.0

    def test_is_local_unknown_model_still_returns_zero(self):
        cost = estimate_cost("some-custom-local-model", 1000, 500, is_local=True)
        assert cost == 0.0

    def test_is_local_default_is_false(self):
        # Without is_local=True, unknown model → None
        cost = estimate_cost("llama3", 1000, 500)
        assert cost is None

    def test_gpt_4o_mini_different_from_gpt_4o(self):
        # Verifies prefix matching picks the more specific key
        cost_mini = estimate_cost("gpt-4o-mini", 1_000_000, 1_000_000)
        cost_full = estimate_cost("gpt-4o", 1_000_000, 1_000_000)
        assert cost_mini is not None
        assert cost_full is not None
        assert cost_mini < cost_full

    def test_claude_3_5_sonnet(self):
        # $3.00/1M prompt + $15.00/1M completion
        cost = estimate_cost("claude-3-5-sonnet-20241022", 500_000, 100_000)
        expected = 500_000 * 3.00 / 1_000_000 + 100_000 * 15.00 / 1_000_000
        assert cost == pytest.approx(expected)

    def test_gemini_flash(self):
        cost = estimate_cost("gemini-1.5-flash", 1_000_000, 1_000_000)
        expected = 0.075 + 0.30
        assert cost == pytest.approx(expected)

    def test_versioned_openai_model(self):
        cost = estimate_cost("gpt-4o-2024-08-06", 1000, 0)
        expected = 1000 * 2.50 / 1_000_000
        assert cost == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# format_cost
# --------------------------------------------------------------------------- #


class TestFormatCost:
    def test_none_shows_question_mark(self):
        assert format_cost(None) == "?"

    def test_zero_shows_local(self):
        assert format_cost(0.0) == "$0.00 (local)"

    def test_tiny_amount_six_decimals(self):
        result = format_cost(0.000123)
        assert result == "$0.000123"

    def test_small_amount_six_decimals(self):
        result = format_cost(0.005)
        assert result.startswith("$0.005")

    def test_normal_amount_two_decimals(self):
        assert format_cost(1.5) == "$1.50"

    def test_large_amount_two_decimals(self):
        assert format_cost(25.0) == "$25.00"

    def test_exactly_one_cent(self):
        # $0.01 is ≥ 0.01 → two decimal format
        assert format_cost(0.01) == "$0.01"

    def test_just_under_one_cent(self):
        # $0.009 < 0.01 → six decimal format
        result = format_cost(0.009)
        assert result.startswith("$0.009")

    def test_format_real_run_cost(self):
        # Simulate a real small run: 2000 prompt + 500 completion on gpt-4o-mini
        # = 2000*0.15/1M + 500*0.60/1M = 0.0003 + 0.0003 = $0.0006
        cost = estimate_cost("gpt-4o-mini", 2000, 500)
        assert cost is not None
        formatted = format_cost(cost)
        assert formatted.startswith("$")
        assert "?" not in formatted
