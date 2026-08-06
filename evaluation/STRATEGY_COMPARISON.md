# Query rewrite strategy comparison: hyde vs step_back vs none

RAGAS evaluation of `evaluation/test_dataset.json` (6 questions) run three times against the live
pipeline, once per `QUERY_REWRITE_STRATEGY` value. `none` is a new baseline (no LLM rewrite call —
the raw question is embedded directly) added in `pipeline/query_rewriter.py` specifically for this
comparison.

## How to reproduce

```bash
docker compose up qdrant langfuse postgres -d   # Qdrant must be reachable
source .venv/bin/activate
QUERY_REWRITE_STRATEGY=hyde      python evaluation/evaluate.py --output evaluation/results_hyde.json
QUERY_REWRITE_STRATEGY=step_back python evaluation/evaluate.py --output evaluation/results_step_back.json
QUERY_REWRITE_STRATEGY=none      python evaluation/evaluate.py --output evaluation/results_none.json
```

Each run makes real OpenAI calls (rewrite + generation + RAGAS's LLM-judge metrics), so this isn't free
or instant — budget for ~20-30 API calls per run on this 6-question dataset. Output files are gitignored
(`evaluation/results_*.json`), same as `evaluation/results.json` — this doc is the durable record.

## Results (run against a live Qdrant with 61 real ingested points from the demo data)

| Strategy    | faithfulness | answer_relevancy | context_recall | context_precision |
|-------------|-------------:|------------------:|----------------:|--------------------:|
| `hyde`      | 0.4594       | 0.3242             | 0.0833          | 0.1667               |
| `step_back` | 0.4466       | 0.4883             | 0.0833          | 0.1667               |
| `none`      | 0.3333       | 0.4807             | 0.0833          | 0.1667               |

## Caveats that matter more than the numbers above

**These numbers aren't a clean signal for picking a strategy.** Two things confound the comparison:

1. **Dataset/corpus mismatch.** `test_dataset.json`'s ground-truth contexts were authored against the
   original Step 14 synthetic fixtures ("Contract CON-00001 with Alpha Supplies Ltd: Payment Terms: Net
   30 days...", etc.). The live Qdrant collection now holds the real demo data seeded later (31
   Contracts, `Zuckerman Security Ltd.`/`Summit Traders Ltd.`/`Alpha Supplies Ltd.` + 19 more — see
   `CLAUDE.local.md`), whose actual chunk text doesn't verbatim-match those ground-truth strings even
   when the *topic* overlaps. That's the likely reason `context_recall` (0.0833) and `context_precision`
   (0.1667) came out **identical across all three strategies** — retrieval is bottlenecked on
   corpus/dataset mismatch, not on rewrite strategy, so there's a low ceiling all three strategies hit
   regardless of query phrasing. The previous single historical run in `evaluation/results.json`
   (faithfulness 0.81, context_recall 0.82) looks much better only because it ran against an *empty*
   Qdrant, which `evaluate.py` self-seeds with the exact ground-truth text — i.e. it was measuring
   round-trip retrieval of text it had just inserted verbatim, not real corpus retrieval.
2. **Reranking always scores against the original question, not the rewritten one.** Both
   `pipeline/query_pipeline.py:121` (production) and `evaluation/evaluate.py:198` (this eval harness)
   call `reranker.rerank(question, candidates, top_n=5)` with the *original* user question, never the
   HyDE/step-back rewrite. The rewrite strategy only influences which ~20 candidates
   `HybridSearch.search()` surfaces (via the rewritten embedding) — the final top-5 that actually reach
   the LLM are re-scored against the unmodified question. This is consistent with the near-identical
   context metrics: whichever strategy is active, the reranker pulls the final set back toward what the
   raw question would have retrieved anyway, damping how much the rewrite step can move the needle on
   what the LLM sees.

`faithfulness`/`answer_relevancy` did vary per strategy (hyde highest faithfulness, step_back highest
answer_relevancy, none lowest on both) but on n=6 questions with a stochastic LLM judge, those gaps are
within noise — not enough to declare a winner.

## Conclusion

No strategy demonstrated a robust win here; the experiment surfaced two real bugs/gaps in the eval setup
(dataset staleness vs. current corpus, and rerank-on-original-question capping rewrite impact) rather than
a clean HyDE-vs-step_back-vs-none verdict. Keeping `hyde` as the default is reasonable pending either:

- refreshing `test_dataset.json`'s ground truth to match the current live demo corpus, and/or
- reconsidering whether `reranker.rerank()` should score against the rewritten query instead of (or in
  addition to) the original question, if the intent is for rewriting to meaningfully affect final context
  selection and not just the initial top-20 candidate pool.

Both are follow-up items, not implemented here — this run's purpose was purely to answer "have we ever
compared these strategies" and produce a documented, reproducible baseline.
