"""Tests for scripts/benchmark_from_langfuse.py pure helpers (#115).

Only the percentile / cost math is covered — the Langfuse API calls are a live
diagnostic, not unit-tested (same convention as the other scripts/).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from benchmark_from_langfuse import _cost_at_list, _pctile  # noqa: E402


def test_pctile_interpolates() -> None:
    assert _pctile([1, 2, 3, 4, 5], 0.5) == 3.0
    assert _pctile([1, 2, 3, 4, 5], 0.0) == 1.0
    assert _pctile([1, 2, 3, 4, 5], 1.0) == 5.0
    assert _pctile([1, 2, 3, 4, 5], 0.95) == 4.8


def test_pctile_single_value() -> None:
    assert _pctile([42.0], 0.5) == 42.0
    assert _pctile([42.0], 0.95) == 42.0


def test_cost_at_list_uses_current_rates() -> None:
    # gpt-4o current list: $2.50 / 1M input
    assert _cost_at_list("gpt-4o", 1_000_000, 0) == 2.5
    assert _cost_at_list("gpt-4o", 0, 1_000_000) == 10.0


def test_cost_at_list_unknown_model_is_none() -> None:
    assert _cost_at_list("some-model-not-in-the-table", 100, 100) is None
