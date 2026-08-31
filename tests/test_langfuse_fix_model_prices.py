"""Tests for scripts/langfuse_fix_model_prices.py pure helpers (#137).

Only the price math and the decision step (`plan_changes`) are covered — the
Langfuse API calls are a live ops diagnostic, not unit-tested (same convention as
the other scripts/, see tests/test_benchmark_script.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from langfuse_fix_model_prices import _per_1m, _row_current, plan_changes  # noqa: E402

_SPEC = {
    "modelName": "gpt-4o",
    "pattern": r"(?i)^(gpt-4o)$",
    "input_per_1m": 2.50,
    "output_per_1m": 10.00,
    "tokenizer": "gpt-4o",
}


def _row(
    *, in_1m: float, out_1m: float, managed: bool, pattern: str = r"(?i)^(gpt-4o)$", mid: str = "x"
) -> dict:
    return {
        "id": mid,
        "matchPattern": pattern,
        "isLangfuseManaged": managed,
        "inputPrice": in_1m / 1_000_000,
        "outputPrice": out_1m / 1_000_000,
    }


# --- _per_1m / _row_current -------------------------------------------------


def test_per_1m_converts_per_token_to_per_million() -> None:
    assert _per_1m(2.5e-06) == 2.5
    assert _per_1m(1e-05) == 10.0
    assert _per_1m(None) == 0.0


def test_row_current_true_only_when_both_prices_match() -> None:
    assert _row_current(_row(in_1m=2.50, out_1m=10.00, managed=False), _SPEC)
    assert not _row_current(_row(in_1m=5.00, out_1m=15.00, managed=False), _SPEC)
    assert not _row_current(_row(in_1m=2.50, out_1m=15.00, managed=False), _SPEC)


# --- plan_changes ---------------------------------------------------------


def test_plan_fresh_project_stale_managed_price_creates_override() -> None:
    existing = [_row(in_1m=5.00, out_1m=15.00, managed=True)]
    plan = plan_changes(_SPEC, existing)
    assert plan == {"status": "fixed", "delete": [], "create": True}


def test_plan_managed_price_already_current_skips() -> None:
    existing = [_row(in_1m=2.50, out_1m=10.00, managed=True)]
    plan = plan_changes(_SPEC, existing)
    assert plan["status"] == "skipped"
    assert plan["create"] is False


def test_plan_force_creates_override_even_when_managed_is_current() -> None:
    existing = [_row(in_1m=2.50, out_1m=10.00, managed=True)]
    plan = plan_changes(_SPEC, existing, force=True)
    assert plan["status"] == "fixed"
    assert plan["create"] is True


def test_plan_steady_state_one_correct_override_is_noop() -> None:
    existing = [
        _row(in_1m=5.00, out_1m=15.00, managed=True),
        _row(in_1m=2.50, out_1m=10.00, managed=False, mid="ovr"),
    ]
    assert plan_changes(_SPEC, existing) == {"status": "ok", "delete": [], "create": False}


def test_plan_stale_override_is_deleted_and_recreated() -> None:
    existing = [_row(in_1m=5.00, out_1m=15.00, managed=False, mid="stale-ovr")]
    plan = plan_changes(_SPEC, existing)
    assert plan["status"] == "fixed"
    assert plan["delete"] == ["stale-ovr"]
    assert plan["create"] is True


def test_plan_dedupes_extra_correct_overrides_keeping_one() -> None:
    existing = [
        _row(in_1m=2.50, out_1m=10.00, managed=False, mid="keep"),
        _row(in_1m=2.50, out_1m=10.00, managed=False, mid="dupe"),
    ]
    plan = plan_changes(_SPEC, existing)
    assert plan["status"] == "fixed"
    assert plan["delete"] == ["dupe"]
    assert plan["create"] is False


def test_plan_ignores_rows_for_other_patterns() -> None:
    existing = [
        _row(in_1m=1.00, out_1m=1.00, managed=False, pattern=r"(?i)^(gpt-4o-mini)$"),
        _row(in_1m=5.00, out_1m=15.00, managed=True),
    ]
    plan = plan_changes(_SPEC, existing)
    assert plan == {"status": "fixed", "delete": [], "create": True}
