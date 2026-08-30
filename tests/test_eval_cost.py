"""Tests for evaluate.py's #130 cost accounting -- pricing, token folding, and the
pipeline + RAGAS-judge cost/request assembly written into results.json.

No network: the judge-usage path is fed a stub RAGAS result.
"""

from __future__ import annotations

from types import SimpleNamespace

import evaluation.evaluate as ev

# ---------------------------------------------------------------------------
# _cost_usd
# ---------------------------------------------------------------------------


def test_cost_usd_prices_a_known_model() -> None:
    # gpt-4o-mini: $0.15 / 1M input, $0.60 / 1M output
    assert ev._cost_usd("gpt-4o-mini", 1_000_000, 0) == 0.15
    assert ev._cost_usd("gpt-4o-mini", 0, 1_000_000) == 0.60
    assert ev._cost_usd("gpt-4o-mini", 1_000_000, 1_000_000) == 0.75


def test_cost_usd_returns_none_for_unpriced_model() -> None:
    assert ev._cost_usd("some-future-model", 1000, 1000) is None


def test_judge_model_and_embedding_model_are_in_the_rate_table() -> None:
    assert ev._RAGAS_JUDGE_MODEL in ev._PRICE_PER_1M_TOKENS
    assert ev._RAGAS_JUDGE_EMBEDDING_MODEL in ev._PRICE_PER_1M_TOKENS


# ---------------------------------------------------------------------------
# _add_tokens
# ---------------------------------------------------------------------------


def test_add_tokens_folds_per_model() -> None:
    acc: dict[str, dict[str, int]] = {}
    ev._add_tokens(acc, "gpt-4o", {"input": 100, "output": 20, "total": 120})
    ev._add_tokens(acc, "gpt-4o", {"input": 50, "output": 10, "total": 60})
    ev._add_tokens(acc, "gpt-4o-mini", {"input": 5, "output": 1, "total": 6})
    assert acc == {
        "gpt-4o": {"input": 150, "output": 30},
        "gpt-4o-mini": {"input": 5, "output": 1},
    }


# ---------------------------------------------------------------------------
# _judge_usage
# ---------------------------------------------------------------------------


def _stub_ragas_result(*usages: tuple[int, int]) -> SimpleNamespace:
    usage_data = [SimpleNamespace(input_tokens=i, output_tokens=o) for i, o in usages]
    return SimpleNamespace(cost_cb=SimpleNamespace(usage_data=usage_data))


def test_judge_usage_counts_requests_and_tokens() -> None:
    result = _stub_ragas_result((100, 20), (200, 40), (300, 60))
    usage = ev._judge_usage(result)
    assert usage["captured"] is True
    assert usage["requests"] == 3
    assert usage["input_tokens"] == 600
    assert usage["output_tokens"] == 120
    assert usage["model"] == ev._RAGAS_JUDGE_MODEL
    # 600 * 0.15/1M + 120 * 0.60/1M = 0.00009 + 0.000072 = 0.000162 -> 0.0002
    assert usage["usd"] == round((600 * 0.15 + 120 * 0.60) / 1_000_000, 4)
    assert usage["embedding_cost_excluded"] is True


def test_judge_usage_uncaptured_when_no_cost_callback() -> None:
    assert ev._judge_usage(SimpleNamespace(cost_cb=None)) == {"captured": False}
    assert ev._judge_usage(SimpleNamespace()) == {"captured": False}
    assert ev._judge_usage(_stub_ragas_result()) == {"captured": False}


# ---------------------------------------------------------------------------
# _assemble_costs
# ---------------------------------------------------------------------------


def test_assemble_costs_sums_pipeline_and_judge() -> None:
    pipeline_tokens = {
        "gpt-4o": {"input": 1_000_000, "output": 100_000},
        "gpt-4o-mini": {"input": 200_000, "output": 20_000},
    }
    judge_usage = {"captured": True, "usd": 0.05, "requests": 400}
    costs = ev._assemble_costs(pipeline_tokens, pipeline_requests=94, judge_usage=judge_usage)

    gpt4o = (1_000_000 * 2.50 + 100_000 * 10.00) / 1_000_000
    mini = (200_000 * 0.15 + 20_000 * 0.60) / 1_000_000
    assert costs["pipeline"]["usd"] == round(gpt4o + mini, 4)
    assert costs["pipeline"]["requests"] == 94
    assert costs["total_usd"] == round(round(gpt4o + mini, 4) + 0.05, 4)
    assert costs["request_count"] == 94 + 400


def test_assemble_costs_pipeline_usd_none_when_a_model_is_unpriced() -> None:
    costs = ev._assemble_costs(
        {"mystery-model": {"input": 10, "output": 10}},
        pipeline_requests=2,
        judge_usage={"captured": True, "usd": 0.01, "requests": 8},
    )
    assert costs["pipeline"]["usd"] is None
    assert costs["total_usd"] is None  # can't total a partial figure
    assert costs["request_count"] == 10


def test_assemble_costs_handles_uncaptured_judge() -> None:
    costs = ev._assemble_costs(
        {"gpt-4o": {"input": 1000, "output": 100}},
        pipeline_requests=2,
        judge_usage={"captured": False},
    )
    assert costs["pipeline"]["usd"] is not None
    assert costs["total_usd"] is None
    assert costs["request_count"] == 2
