# Retrieval Tuning — `top_k`, `top_n`, and friends

How the retrieval knobs would be tuned properly. Right now they hold conventional
"retrieve wide, rerank narrow" defaults — **not values tuned on this corpus.** The
eval harness that could tune them (`evaluation/evaluate.py`, per-`case_class`
slicing) exists as of #102, but the dataset is still 2–4 questions per slice —
too small to move a knob without fitting to noise.

## The knobs

| Knob | Value | Where | What it controls |
|---|---|---|---|
| `chunk_size` | 512 | `ingestion/chunker.py:23` | characters per chunk |
| `chunk_overlap` | 64 | `ingestion/chunker.py:24` | overlap between adjacent chunks |
| RRF `k` | 60 | `retrieval/hybrid_search.py:25` | rank-fusion smoothing constant (not a "top-k") |
| `top_k` | 20 | `retrieval/hybrid_search.py:53`, `pipeline/query_pipeline.py` | candidate pool the cross-encoder rescores |
| `top_n` | 5 | `retrieval/reranker.py:28`, `pipeline/query_pipeline.py` | chunks the LLM actually sees |

`top_k` and `top_n` answer different questions and are tuned separately, in order.

## Why the current defaults are defensible

**`top_k = 20`** — the reranker can only reorder what it is handed; it cannot
recover a chunk that RRF ranked 25th. So the pool must be wide enough to *contain*
the gold chunk, but every extra candidate is another cross-encoder forward pass
(cost is linear in pool size). On a ~60-chunk corpus, 20 is a third of everything
and recall into the pool is effectively 1.0.

**`top_n = 5`** — LLMs degrade with irrelevant context ("lost in the middle",
distraction), and each chunk is ~512 tokens, so 5 ≈ 2.5k context tokens/query vs
~10k for 20 — 4× the generation cost for little gain on point-lookup questions.
"3–5 passages" is the standard for passage-grounded QA.

The one place `top_n = 5` clearly hurts is enumeration ("list all cancelled
contracts" drops the 3rd match — see `docs/ARCHITECTURE.md` § Known Limitations).
Raising `top_n` is the *wrong* fix there; that needs a separate metadata-filtered
retrieval path (#45). Enumeration is scoped out rather than distorting `top_n` for
the common case.

## Prerequisites for a real tuning pass

1. **qrels** — for each eval question, a label of which chunks are actually
   relevant (the gold set). `ground_truth_contexts` in `test_dataset.json` is
   close but is hand-authored prose, not chunk IDs; a tuning pass needs the
   labels tied to real `docname:chunk_index` keys.
2. **A bigger dataset** — ~100–200 questions, 15–30 per `case_class` (see the
   power analysis in Step 3). Tracked as the follow-up to #102.
3. **A dev/test split** — tune `k` on dev, report final numbers once on held-out
   test. Tuning and reporting on the same set overfits `k` to the eval.

## Step 1 — `top_k`: a retrieval-recall question, no LLM needed

1. For each dev question, run `HybridSearch.search()` and record the fused rank
   of every gold chunk.
2. Compute **`recall@k`** for k ∈ {10, 20, 30, 50, 100} — the fraction of gold
   chunks that land in the top-k of the fused list. Deterministic, no API calls.
3. Plot `recall@k` and find the **knee** — the smallest k where recall plateaus
   (e.g. ≥ 0.95). Past the knee you pay reranker latency for nothing.
4. Break `recall@k` out **by `case_class`**. Recall degrades on the hard slices
   (`semantic-no-anchor`, `disambiguation`) — tune to those, not the mean.
5. Cross-check against **reranker latency vs pool size** (linear in `top_k`). If
   recall is still climbing at the latency budget, take the largest k you can
   afford; otherwise stop at the knee.

Output: one `top_k` value.

## Step 2 — `top_n`: needs the full pipeline + judge

Hold `top_k` at the Step 1 value. Sweep `top_n ∈ {3, 5, 8, 10, 15}`. For each,
run `evaluate.py` and record **per `case_class`**:

| Signal | What it tells you |
|---|---|
| `faithfulness`, `answer_relevancy` on `showcase` | does more context *dilute* the easy cases? |
| `context_recall` on `disambiguation` / `precision-multi` | does more context *recover* multi-answer material? |
| `context_precision` | drops mechanically as `top_n` rises — watch, don't over-index |
| answer correctness (exact / human, not just RAGAS) | the ground truth |
| tokens/query, $/query, p95 latency | the cost axis, on the same plot |

The value is the inflection point: where `context_recall` on the hard slices
**stops improving** but `faithfulness` and cost **keep getting worse**.

## Step 3 — statistical rigor (the part usually skipped)

RAGAS scores are LLM-judged and wobble ±0.03–0.05 run-to-run on identical inputs.
To claim `top_n = 5` ≠ `top_n = 8` the confidence intervals must separate.

- **Paired comparison** — same questions across every config → use **Wilcoxon
  signed-rank**, not an unpaired test. Much more sensitive.
- **Bootstrap** the per-question scores for CIs, or run each config 3× and
  average out judge noise.
- **Power** — with per-question SD ~0.15 and wanting to detect a 0.05 gap at 80%
  power, you need ~70 questions *per slice*. Hence ~100–200 total.
- **Decision rule** — the cheapest, fastest config whose quality is
  statistically within noise of the best.

## Step 4 — expect the answer to not be a single number

Likely conclusions:

- `top_k` = one global value (recall plateaus).
- `top_n` = **intent-dependent** — 5 for point lookups, a wider value (or the
  metadata-filter path) for enumeration / comparison intents. This would mean
  classifying intent before retrieval, or always retrieving wide and letting the
  generator's own filtering handle the common case.
- `chunk_size` **interacts** with `top_n` — bigger chunks carry more per chunk,
  so fewer are needed. A thorough pass sweeps `chunk_size × top_n` as a small
  grid rather than fixing one.

## Related

- `docs/ARCHITECTURE.md` § Evaluation — the harness and `case_class` slices
- `docs/ARCHITECTURE.md` § Known Limitations — why enumeration/temporal are out of scope
- #102 — the harder eval dataset (prerequisite); its follow-up is the ~150-question set
- #49 — CI threshold gating, which also wants per-`case_class` numbers
- #45 — the enumeration investigation, closed as out of scope
