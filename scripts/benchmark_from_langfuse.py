#!/usr/bin/env python3
"""Aggregate query-path and ingest-path latency + cost from the Langfuse trace DB.

Read-only against the Langfuse public API — makes no OpenAI calls and writes
nothing. It re-derives the numbers in ``docs/BENCHMARKS.md`` from whatever traces
are currently in Langfuse, so the doc stays reproducible: run an eval pass with
``LANGFUSE_*`` set (``python evaluation/evaluate.py --split test``), then run this.

Cost is reported twice: as Langfuse computes it (its model-price table), and
recomputed from token counts at the rates in ``evaluation/evaluate.py``'s
``_PRICE_PER_1M_TOKENS``. These agree once ``scripts/langfuse_fix_model_prices.py``
has run against the project; without it, self-hosted Langfuse 2.x prices gpt-4o at
its stale mid-2024 launch rate, ~1.9x the current list price (#137, BENCHMARKS.md).

Usage:
    python scripts/benchmark_from_langfuse.py [--query-run-size 53]
"""

from __future__ import annotations

import argparse
import base64
import json
import statistics as st
import sys
import urllib.request
from pathlib import Path

from dotenv import dotenv_values

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.evaluate import _PRICE_PER_1M_TOKENS  # noqa: E402

_ENV = {**dotenv_values(Path(__file__).resolve().parent.parent / ".env")}
_HOST = _ENV.get("LANGFUSE_HOST", "http://localhost:3000").rstrip("/")
_QUERY_SPANS = ("rewrite", "filter_extraction", "hybrid_search", "rerank", "generate")


def _api(path: str) -> dict:
    token = base64.b64encode(
        f"{_ENV['LANGFUSE_PUBLIC_KEY']}:{_ENV['LANGFUSE_SECRET_KEY']}".encode()
    ).decode()
    req = urllib.request.Request(_HOST + path, headers={"Authorization": f"Basic {token}"})
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 (localhost)
        return json.load(resp)


def _pctile(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    lo = int(k)
    if lo + 1 >= len(xs):
        return xs[lo]
    return xs[lo] + (xs[lo + 1] - xs[lo]) * (k - lo)


def _paged(path: str, want: int) -> list[dict]:
    out: list[dict] = []
    page = 1
    while len(out) < want and page <= 40:
        data = _api(f"{path}&limit=100&page={page}").get("data", [])
        if not data:
            break
        out += data
        page += 1
    return out


def _cost_at_list(model: str, prompt_tok: float, completion_tok: float) -> float | None:
    rate = _PRICE_PER_1M_TOKENS.get(model)
    if rate is None:
        return None
    return (prompt_tok * rate[0] + completion_tok * rate[1]) / 1_000_000


def query_path(run_size: int) -> None:
    traces = _paged("/api/public/traces?name=eval_question", want=run_size * 2)
    traces.sort(key=lambda t: t["timestamp"], reverse=True)
    run = traces[:run_size]
    if not run:
        print("no eval_question traces found — run evaluate.py with LANGFUSE_* set first")
        return
    run_ids = {t["id"] for t in run}
    e2e = [t["latency"] for t in run if t.get("latency")]
    lf_cost = [t["totalCost"] for t in run if t.get("totalCost")]

    print(f"\n== QUERY PATH ==  n={len(run)}  "
          f"{run[-1]['timestamp'][:19]} .. {run[0]['timestamp'][:19]}")
    print(f"  {'stage':16} {'n':>4} {'p50':>8} {'p95':>8} {'mean':>8}  tokens (prompt/compl)")
    for span in _QUERY_SPANS:
        obs = [
            o for o in _paged(f"/api/public/observations?name={span}", want=run_size * 3)
            if o.get("traceId") in run_ids and o.get("latency") is not None
        ]
        if not obs:
            print(f"  {span:16} {'—':>4}  (not in this run)")
            continue
        lat = [o["latency"] for o in obs]
        pin = st.mean([o.get("promptTokens") or 0 for o in obs])
        pout = st.mean([o.get("completionTokens") or 0 for o in obs])
        tok = f"{pin:.0f}/{pout:.0f}" if (pin or pout) else "—"
        print(f"  {span:16} {len(lat):>4} {_pctile(lat, .5):>7.3f}s {_pctile(lat, .95):>7.3f}s "
              f"{st.mean(lat):>7.3f}s  {tok}")

    print(f"  {'end-to-end':16} {len(e2e):>4} {_pctile(e2e, .5):>7.3f}s "
          f"{_pctile(e2e, .95):>7.3f}s {st.mean(e2e):>7.3f}s")

    gen = [
        o for o in _paged("/api/public/observations?name=generate", want=run_size * 3)
        if o.get("traceId") in run_ids
    ]
    pin = st.mean([o.get("promptTokens") or 0 for o in gen])
    pout = st.mean([o.get("completionTokens") or 0 for o in gen])
    model = gen[0].get("model") if gen else "?"
    listed = _cost_at_list(model, pin, pout)
    print(f"\n  cost/query — Langfuse:   ${st.mean(lf_cost):.5f} mean  "
          f"(p95 ${_pctile(lf_cost, .95):.5f})")
    if listed is not None:
        print(f"  cost/query — {model} @ current list ({_PRICE_PER_1M_TOKENS[model][0]}/"
              f"{_PRICE_PER_1M_TOKENS[model][1]} per 1M): ${listed:.5f} generate only")


def ingest_path() -> None:
    traces = [
        t for t in _paged("/api/public/traces?name=full_ingest", want=20)
        if (t.get("output") or {}).get("chunks_indexed")
    ]
    traces.sort(key=lambda t: t["timestamp"], reverse=True)
    print("\n== INGEST PATH ==")
    for t in traces[:5]:
        full = _api(f"/api/public/traces/{t['id']}")
        embeds = [o for o in full.get("observations", []) if o.get("name") == "embed"]
        etok = sum(o.get("promptTokens") or o.get("totalTokens") or 0 for o in embeds)
        ecost = sum(o.get("calculatedTotalCost") or 0 for o in embeds)
        out = t.get("output") or {}
        wall = t.get("latency") or 0
        docs = out.get("documents_indexed", 0)
        chunks = out.get("chunks_indexed", 0)
        rate = f"{docs / wall:.1f} docs/s, {chunks / wall:.1f} chunks/s" if wall else "—"
        print(f"  {t['timestamp'][:19]}  wall={wall:6.1f}s  {docs} docs / {chunks} chunks  "
              f"{rate}  embed_tok={etok}  embed_$={ecost:.6f}")

    webhooks = _paged("/api/public/traces?name=webhook_reindex", want=10)
    ok = [t for t in webhooks if (t.get("output") or {}).get("status") == "indexed"]
    if ok:
        lat = [t["latency"] for t in ok if t.get("latency")]
        print(f"  webhook_reindex (1 doc): n={len(lat)}  "
              f"p50={_pctile(lat, .5):.2f}s  p95={_pctile(lat, .95):.2f}s")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--query-run-size",
        type=int,
        default=53,
        help="how many recent eval_question traces make one run (default 53 = --split test)",
    )
    args = ap.parse_args()
    query_path(args.query_run_size)
    ingest_path()


if __name__ == "__main__":
    main()
