# Benchmarks — latency & cost

Reproducible performance and cost baseline for the query and ingest paths. **The
method matters more than the absolute numbers here:** this is a single-user demo
over a ~60-chunk corpus with no concurrency — too small for retrieval latency to
be interesting. What transfers is *how* each number was measured and *what would
change under real load* (§ [At scale](#what-changes-at-scale)).

The figures below were taken on the actual production box (AWS EC2 `t3.large`)
during the #139 EC2 migration rehearsal, not on a laptop — so the network path to
OpenAI and the 2-vCPU CPU inference are the real deployment's, not an M1's. An
earlier revision of this doc measured the same pipeline on an Apple M1; where the
gap is instructive it is called out inline.

Regenerate every number below with:

```bash
# 1. produce a fresh run of query traces (no RAGAS judge -- latency/cost only, ~5x faster, ~40% cheaper)
python evaluation/evaluate.py --split test --no-judge
# 2. aggregate them
python scripts/benchmark_from_langfuse.py        # query + ingest, read-only from the Langfuse trace DB
```

Step 1 is also the right smoke test after moving the stack to new infrastructure —
it exercises every question end-to-end and records real per-stage latency on the
new host without paying for the judge (whose scores don't depend on the host).
On a *fresh* deployment step 2 needs `.env` reachable inside the `app` container
and the golden-set dataset is not required — see **#146**.

---

## Methodology

| | |
|---|---|
| **Source** | Langfuse trace DB (self-hosted, `docker compose`). Every pipeline step is a span/generation with real wall-clock latency; `rewrite`, `embed_query` and `generate` are `generation`s with token usage. No numbers are hand-timed. |
| **Query-path sample** | 53 `eval_question` traces — one `python evaluation/evaluate.py --split test --no-judge` pass. Same span shape as the production `query` pipeline **minus `filter_extraction`** (the eval harness passes filters explicitly; see the note in the query table). |
| **Ingest-path sample** | `full_ingest` traces from `POST /ingest/full` over the demo corpus, plus `webhook_reindex` traces from two single-document webhook edits. |
| **Build under test** | `5cda5d5` (2026-08-31). Includes `REWRITE_MODEL=gpt-4o-mini` (#128), `rewrite`-as-`generation` (#134 — the sampled traces carry real rewrite token cost) and the #138 query-embed split (the dense retrieval leg no longer re-embeds; `embed_query` is its own span). |
| **Hardware** | AWS EC2 `t3.large` — 2 vCPU (burstable), 8 GB RAM, Ubuntu 22.04. All services (Qdrant, Langfuse + Postgres, reranker, nginx) co-resident on the one box (Option B topology, `docs/DEPLOYMENT.md`). Cross-encoder inference is CPU-only — no MPS/CUDA. |
| **Corpus** | 59 chunks — 40 documents (31 Contracts + 9 Terms and Conditions; Contract PDF attachments index under the parent). `text-embedding-3-small`, 1536-dim, cosine. |
| **Warm vs cold** | Query numbers are **warm** — reranker loaded, BM25 index built. Cold start is measured separately in § [Cold start](#cold-start). |
| **Concurrency** | None. One sequential caller. Every p95 here is single-request variance, not contention. |
| **Cost rates** | Recomputed from token counts at current OpenAI list prices (`evaluation/evaluate.py`, `_PRICE_PER_1M_TOKENS`). `scripts/langfuse_fix_model_prices.py` was run on this box (Phase 3 of `docs/CLIENT_DEPLOYMENT_RUNBOOK.md`), so Langfuse's own `totalCost` now matches the recompute — see § [Cost](#cost). |

---

## Query path

Per-stage and end-to-end latency, 53-question `--split test --no-judge` run:

| Stage | Type | n | p50 | p95 | mean | Notes |
|---|---|---|---|---|---|---|
| `rewrite` | generation | 53 | 1.461 s | 2.130 s | 1.571 s | one `gpt-4o-mini` chat call (HyDE — writes a hypothetical answer paragraph); 71 prompt / 140 completion tokens mean. Latency is OpenAI round-trip from `eu-central-1`, not local compute |
| `filter_extraction` | span | — | — | — | — | **not measured** — the eval harness skips it. In the `query` pipeline it is a synchronous keyword scan of the question string (`_extract_filters`), no I/O, sub-millisecond |
| `embed_query` | generation | 53 | ~0.3 s | — | — | **not broken out** by `benchmark_from_langfuse.py`; the ~0.32 s residual between the summed stages (3.57 s) and end-to-end p50 (3.88 s) is `embed_query` (one `text-embedding-3-small` call, ~75 tokens) plus orchestration overhead |
| `hybrid_search` | span | 53 | 0.011 s | 0.027 s | 0.013 s | BM25 (in-memory) ∥ Qdrant vector search, RRF fused, top-20. **~15× faster than the M1 revision (0.166 s)** — #138 removed the duplicate dense-leg embed that used to sit inside this span |
| `rerank` | span | 53 | 1.150 s | 1.617 s | 1.216 s | cross-encoder over 20 `(query, chunk)` pairs on 2 vCPU, top-5. ~7–8× the M1 (0.15 s) — no fast Apple PyTorch path, burstable CPU |
| `generate` | generation | 53 | 0.943 s | 2.722 s | 1.110 s | `gpt-4o`, ~745 prompt / ~61 completion tokens mean (top-5 context + system prompt) |
| **end-to-end** `/query` | trace | 53 | **3.883 s** | **5.482 s** | 4.087 s | |

**`rewrite` + `rerank` are ~67 % of median latency now** (1.46 s + 1.15 s of a
3.88 s median). On the M1 revision `generate` dominated at ~68 %; here it is only
~24 %. The shift is entirely the host: `rewrite` pays OpenAI + trans-Atlantic
round-trip, `rerank` runs the cross-encoder on 2 burstable vCPUs. The Qdrant/BM25
retrieval stage is now negligible (`0.011 s`).

p95 spread is `generate` (OpenAI-side variance, 0.94 → 2.72 s) plus `rewrite`
(2.13 s).

---

## Cold start

Paid once, at API startup (`api/main.py` checks the collection, rebuilds the BM25
index, and warms the reranker before serving). Measured on the `t3.large` from a
`docker compose restart app`, via Docker's log timestamps:

| Phase | Time | Notes |
|---|---|---|
| Container restart + Python import | **~14 s** | SIGTERM to uvicorn's "Started server process". Dominated by importing `torch` + `sentence-transformers` on 2 burstable vCPUs (the M1 revision isolated this slice at ~6.6 s). |
| Lifespan startup hook | **~1.4 s** | `GET /collections` + BM25 rebuild over 59 chunks + reranker `warm_up()` + Langfuse init |
| **`restart` → serving** | **~15 s** | to "Application startup complete" / `/health` |
| First `rerank` after boot | ~1–1.5 s extra | lazy first-forward init on the first real query (M1 revision); not isolated on this run |
| Reranker model download | ~90 MB, first-ever run only | `cross-encoder/ms-marco-MiniLM-L-6-v2`, then cached in the image layer |
| BM25 index build at startup | part of the ~1.4 s above | `O(n)` in corpus size — instant at 59 chunks, see § [At scale](#what-changes-at-scale) |

So `/health` is meaningful **~15 s** after `docker compose restart app`, and the
first query after a restart can see ~1 s extra in `rerank`. The `torch` import,
not the pipeline warm-up, is the bulk of it.

---

## Ingest path

`POST /ingest/full` over the 59-chunk corpus:

| Metric | Value |
|---|---|
| Wall time | **17.7 s** for 40 documents / 59 chunks |
| Throughput | ~2.3 docs/s, ~3.3 chunks/s |
| Dominant cost | `embed` — 40 sequential `text-embedding-3-small` calls, **~0.44 s each** (network round-trip `eu-central-1` → OpenAI; ~2× the M1's ~0.19 s). Parsing, chunking and Qdrant upsert are near-zero next to the embedding round-trips. |
| Embed tokens | 3,032 total for the whole corpus |

Embedding is **one batched call per document, documents processed serially** — the
wall time is 40 network round-trips in sequence, not compute. On a box further
from the OpenAI endpoint this term grows; batching across documents (not currently
done) or a local embedding model (#101) would cut it.

**Incremental (webhook) re-index:** p50 **1.66 s**, p95 1.70 s (n=2) for a single
1-chunk document — fetch + parse + chunk + 1 embed + upsert + full BM25 rebuild.
~2.5× the M1's ~0.6 s (network again). The first re-index after a restart also
pays the cold-reranker cost on top.

---

## Cost

All figures recomputed from **measured token counts** at **current OpenAI list
prices**. `scripts/langfuse_fix_model_prices.py` has run on this project, so
Langfuse's own `totalCost` agrees.

### Per query

| Component | Model | Tokens (mean) | Cost |
|---|---|---|---|
| `generate` | `gpt-4o` | 745 in / 61 out | **$0.00247** |
| `rewrite` (HyDE) | `gpt-4o-mini` | 71 in / 140 out | **$0.00009** |
| `embed_query` | `text-embedding-3-small` | ~75 | <$0.000002 (negligible) |
| **Total** | | | **≈ $0.0026 / query** |

Cross-check — Langfuse's own cost on these 53 traces: **$0.00257 mean, $0.00365
p95**. The recompute and Langfuse now agree because the project's `gpt-4o` price
override is in place (**#137**; without it self-hosted Langfuse 2.x prices
`gpt-4o` at its stale mid-2024 launch rate, ~1.9× high — the failure mode the
earlier revision of this doc was measured under). `results.json`'s `costs` block
(#130) prices independently at current rates and was never affected.

### Per full ingest

**$0.000061** — the entire 59-chunk corpus embeds for ~3,032
`text-embedding-3-small` tokens. Embedding prices are unchanged, so Langfuse
agrees here.

### Projected monthly

Generation dominates the marginal cost; ingest is a rounding error. At
**$0.0026/query**:

| Query volume | OpenAI / month | + VM (`t3.large` on-demand, `eu-central-1`, + 30 GB gp3) | Total |
|---|---|---|---|
| 200 queries/day | ~$16 | ~$60 | **~$75/mo** |
| 1,000 queries/day | ~$78 | ~$60 | **~$140/mo** |

Assumption: query mix and token counts match the `--split test` set (contract
Q&A, ~745-token prompts). A verbose-answer or larger-context workload scales the
`generate` term linearly. A reserved instance or Savings Plan roughly halves the
VM term.

---

## What changes at scale

These numbers are from 59 chunks and one caller. What moves first as either grows:

- **BM25 index rebuild** (`retrieval/hybrid_search.py`) — the in-memory index is
  rebuilt *in full* at startup **and on every webhook**. `O(n)` in corpus size;
  part of the ~1.4 s warm-up at 59 chunks, seconds at 100k, and it blocks the
  event loop during a webhook re-index. This is the first thing to bite — tracked
  in #97 (migrate the lexical leg to Qdrant native sparse vectors).
- **Reranker under concurrency** — one lazy singleton, CPU inference on 2
  burstable vCPUs, no batching across requests. Single-caller `rerank` is already
  1.15 s here; concurrent queries serialize on it and the p95 would climb sharply
  above one concurrent caller. This is the tightest constraint on this instance
  size — see #140.
- **Qdrant vector search** — HNSW latency is roughly flat into the millions of
  vectors, so this stays sub-second. Memory is the constraint: ~6 KB per 1536-dim
  point plus payload.
- **`generate`** — unchanged by corpus size (fixed top-5 context). Scales with
  OpenAI-side latency and rate limits, not anything local.
- **Ingest** — serial per-document embedding means full-ingest wall time is linear
  in document count (~0.44 s each from this box). Batching across documents would
  cut it; not currently done.

---

## Related

- `docs/ARCHITECTURE.md` § Observability — the span/generation shapes these numbers come from
- #139 — the EC2 migration rehearsal these `t3.large` figures were taken during
- #146 — `benchmark_from_langfuse.py` / `evaluate.py` friction on a fresh deploy
- #140 — concurrency benchmark + concurrent-user capacity envelope
- #130 — per-run cost capture in `results.json` (`costs` block, `request_count`)
- #137 — Langfuse self-hosted `gpt-4o` stale price, fixed by `scripts/langfuse_fix_model_prices.py`
- #138 — query-embed split (why `hybrid_search` dropped to ~0.01 s)
- #97 — BM25 → Qdrant sparse vectors (the scale bottleneck above)
- #101 — local embedding model (would remove the embed round-trip from ingest and query)
- #50 — quality monitoring (the other axis: is the answer good, not how fast/cheap)
