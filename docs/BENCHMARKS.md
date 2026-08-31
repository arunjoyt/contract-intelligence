# Benchmarks — latency & cost

Reproducible performance and cost baseline for the query and ingest paths. **The
method matters more than the absolute numbers here:** this is a single-user demo
over a ~60-chunk corpus on a laptop — too small for retrieval latency to be
interesting and with no concurrency. What transfers is *how* each number was
measured and *what would change under real load* (§ [At scale](#what-changes-at-scale)).

Regenerate every number below with:

```bash
python scripts/benchmark_from_langfuse.py       # query + ingest, read-only from the Langfuse trace DB
```

---

## Methodology

| | |
|---|---|
| **Source** | Langfuse trace DB (self-hosted, `docker compose`). Every pipeline step is a span/generation with real wall-clock latency; `generate` and (since #134) `rewrite` are `generation`s with token usage. No numbers are hand-timed. |
| **Query-path sample** | 53 `eval_question` traces — one `python evaluation/evaluate.py --split test` pass. Same span shape as the production `query` pipeline **minus `filter_extraction`** (the eval harness passes filters explicitly; see the note in the query table). |
| **Ingest-path sample** | `full_ingest` traces from `POST /ingest/full` over the demo corpus. |
| **Build under test** | ~`284b891` (2026-08-29). `REWRITE_MODEL=gpt-4o-mini` (#128) in effect; `rewrite`-as-`generation` (#134) **not** yet — so the sampled traces carry no rewrite token cost, added analytically in § [Cost](#cost). |
| **Hardware** | Apple M1, 8 GB RAM, macOS 26.5. All services (Qdrant, Langfuse + Postgres, reranker) on the one machine. |
| **Corpus** | 61 chunks — 33 Contracts (52 chunks) + 9 Terms and Conditions (9 chunks). `text-embedding-3-small`, 1536-dim, cosine. |
| **Warm vs cold** | Query numbers are **warm** — reranker loaded, BM25 index built. Cold start is measured separately in § [Cold start](#cold-start). |
| **Concurrency** | None. One sequential caller. Every p95 here is single-request variance, not contention. |
| **Cost rates** | Recomputed from token counts at current OpenAI list prices (the table in `evaluation/evaluate.py`, `_PRICE_PER_1M_TOKENS`). Langfuse's own `totalCost` on these traces is ~1.9× higher — the project's `gpt-4o` price was still the stale launch rate when this ran (#137, since fixed by `scripts/langfuse_fix_model_prices.py`); see § [Cost](#cost). |

---

## Query path

Per-stage and end-to-end latency, 53-question `--split test` run:

| Stage | Type | n | p50 | p95 | mean | Notes |
|---|---|---|---|---|---|---|
| `rewrite` | generation | 53 | 0.156 s | 0.176 s | 0.160 s | one `gpt-4o-mini` chat call (HyDE) **+ one `text-embedding-3-small` call** on this build — #138 later split that embed into its own `embed_query` generation (~0.15–0.2 s), so on current builds `rewrite` is chat-only and `embed_query` is a sibling span; end-to-end is unchanged minus one now-removed duplicate embed |
| `filter_extraction` | span | — | — | — | — | **not measured** — the eval harness skips it. In the `query` pipeline it is a synchronous keyword scan of the question string (`_extract_filters`), no I/O, sub-millisecond |
| `hybrid_search` | span | 53 | 0.166 s | 0.193 s | 0.168 s | BM25 (in-memory) ∥ Qdrant vector search, RRF fused, top-20 |
| `rerank` | span | 53 | 0.150 s | 0.197 s | 0.166 s | cross-encoder over 20 `(query, chunk)` pairs on CPU, top-5 |
| `generate` | generation | 53 | 1.007 s | 2.018 s | 1.097 s | `gpt-4o`, ~1340 prompt / ~73 completion tokens |
| **end-to-end** `/query` | trace | 53 | **1.487 s** | **2.629 s** | 1.592 s | |

**`generate` is ~68 % of median latency** and essentially all of the p95 spread
(OpenAI-side variance). The three local retrieval stages are ~0.15–0.17 s each and
flat — at this corpus size they are not doing meaningful work.

---

## Cold start

Paid once, at API startup (`api/main.py` warms the reranker and rebuilds the BM25
index before serving). Measured on the same M1:

| Step | Time | Notes |
|---|---|---|
| `import retrieval.reranker` | ~6.6 s | pulls in `torch` + `sentence-transformers` — one-time process cost |
| `Reranker().warm_up()` | ~1.4 s | model load from the local HF cache + init |
| First `rerank` after boot | ~1.5 s | lazy CUDA/MPS-absent init on the first real forward pass; subsequent calls are ~0.15 s (see the query table) |
| Reranker model download | ~90 MB | first ever run only — `cross-encoder/ms-marco-MiniLM-L-6-v2` from HuggingFace, then cached |
| BM25 index build at startup | instant @ 61 chunks | `O(n)` in corpus size — see § [At scale](#what-changes-at-scale) |

So `/health` is meaningful ~8 s after `docker compose up`, and the **first query
after a restart** can see an extra ~1.5 s in the `rerank` stage.

---

## Ingest path

`POST /ingest/full` over the 61-chunk corpus, representative isolated runs:

| Metric | Value |
|---|---|
| Wall time | **~10 s** for 42 documents / 61 chunks (isolated runs 9.4–10.0 s; three *overlapping* runs measured 23–25 s — contention, not the steady-state number) |
| Throughput | ~4.3 docs/s, ~6.1 chunks/s |
| Dominant cost | `embed` — 42 sequential `text-embedding-3-small` calls, ~0.19 s each, **~82 % of wall time**. Parsing, chunking and Qdrant upsert are near-zero next to the embedding round-trips. |
| ERPNext listing | 2 `list:<doctype>` calls, ~0.06 s each |

Embedding is **one batched call per document, documents processed serially** — the
wall time is 42 network round-trips in sequence, not compute.

**Incremental (webhook) re-index:** ~0.6 s for a single 1-chunk document (fetch +
parse + chunk + 1 embed + upsert + full BM25 rebuild); the first re-index after a
restart pays the cold-reranker cost on top (~2.5 s observed).

---

## Cost

All figures recomputed from **measured token counts** at **current OpenAI list
prices**. Where Langfuse's own `totalCost` differs, both are shown.

### Per query

| Component | Model | Tokens (mean) | Cost |
|---|---|---|---|
| `generate` | `gpt-4o` | 1340 in / 73 out | **$0.0041** |
| `rewrite` (HyDE) | `gpt-4o-mini` | ~80 in / ~150 out (est.) | **~$0.0001** |
| query embedding | `text-embedding-3-small` | ~150 | <$0.00001 (negligible) |
| **Total** | | | **≈ $0.0042 / query** |

> **These numbers were measured against a Langfuse project whose `gpt-4o` price
> was the stale mid-2024 launch rate** ($5 / $15 per 1M in/out vs. the current
> $2.50 / $10), so the raw traces read ~$0.0078/query — nearly 2×. Token counts
> were always correct; only Langfuse's bundled price table was stale (**#137**).
> Fixed by `scripts/langfuse_fix_model_prices.py`, which adds a project-level
> price override at the current rate — run per deployment (Phase 3 of
> `docs/CLIENT_DEPLOYMENT_RUNBOOK.md`). `results.json`'s `costs` block (#130)
> prices independently at current rates and was never affected.

### Per full ingest

**$0.00006** — the entire 61-chunk corpus embeds for ~3,000 `text-embedding-3-small`
tokens. Embedding prices have not changed, so Langfuse agrees here.

### Projected monthly

Generation dominates; ingest is a rounding error. At **$0.0042/query**:

| Query volume | OpenAI / month | + VM (`t3.large`, on-demand) | Total |
|---|---|---|---|
| 200 queries/day | ~$25 | ~$60 | **~$85/mo** |
| 1,000 queries/day | ~$126 | ~$60 | **~$186/mo** |

Assumption: query mix and token counts match the `--split test` set (contract Q&A,
~1340-token prompts). A verbose-answer or larger-context workload scales the
`generate` term linearly.

---

## What changes at scale

These numbers are from 61 chunks and one caller. What moves first as either grows:

- **BM25 index rebuild** (`retrieval/hybrid_search.py`) — the in-memory index is
  rebuilt *in full* at startup **and on every webhook**. `O(n)` in corpus size;
  instant at 61 chunks, seconds at 100k, and it blocks the event loop during a
  webhook re-index. This is the first thing to bite — tracked in #97 (migrate the
  lexical leg to Qdrant native sparse vectors).
- **Reranker under concurrency** — one lazy singleton, CPU inference, no batching
  across requests. Concurrent queries serialize on it; the `rerank` p95 would
  climb sharply above one concurrent caller.
- **Qdrant vector search** — HNSW latency is roughly flat into the millions of
  vectors, so this stays ~0.15 s. Memory is the constraint: ~6 KB per 1536-dim
  point plus payload.
- **`generate`** — unchanged by corpus size (fixed top-5 context). Scales with
  OpenAI-side latency and rate limits, not anything local.
- **Ingest** — serial per-document embedding means full-ingest wall time is linear
  in document count (~0.19 s each). Batching across documents would cut it; not
  currently done.

---

## Related

- `docs/ARCHITECTURE.md` § Observability — the span/generation shapes these numbers come from
- #130 — per-run cost capture in `results.json` (`costs` block, `request_count`)
- #137 — Langfuse self-hosted `gpt-4o` stale price, fixed by `scripts/langfuse_fix_model_prices.py`
- #97 — BM25 → Qdrant sparse vectors (the scale bottleneck above)
- #101 — local embedding model (would remove the embed round-trip from ingest and query)
- #50 — quality monitoring (the other axis: is the answer good, not how fast/cheap)
