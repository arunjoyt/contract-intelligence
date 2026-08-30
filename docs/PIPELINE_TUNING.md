# Pipeline Tuning — every knob from chunk to answer

How the retrieval and generation knobs would be tuned properly. Right now they all
hold conventional "retrieve wide, rerank narrow" defaults — **not values tuned on
this corpus.** The eval harness that could tune them (`evaluation/evaluate.py`,
per-`case_class` slicing) exists as of #102; #112 grew the dataset to 92 entries
with a dev/test split. That is enough to tune the generation-side knobs on the
judge headline. The ~38-clause demo corpus is too small to *separate* the
retrieval-only knobs, and expanding it was considered and rejected — those knobs
stay at conventional defaults and are owned by per-client validation instead. See
[the corpus-size ceiling](#the-corpus-size-ceiling--what-113-can-and-cannot-tune-here).

This doc walks the whole pipeline order: ingest → embed → rewrite → retrieve →
rerank → generate. It describes the **one-time** tuning pass on the reference
corpus (the demo data) — issue #113. Per-client work is a lighter validation
pass, not a re-run; see [Per-client tuning](#per-client-tuning--tune-once-validate-per-client)
below and #127.

## The knobs

| Knob | Value | Where | Stage | Controls | Tuning method |
|---|---|---|---|---|---|
| `chunk_size` | 512 | `ingestion/chunker.py` | ingest | characters per chunk | judge (interacts with `top_n`) |
| `chunk_overlap` | 64 | `ingestion/chunker.py` | ingest | overlap between adjacent chunks | judge |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | `config.py` | ingest + query | the dense vector space | deterministic (`recall@k`) |
| `QUERY_REWRITE_STRATEGY` | `hyde` | env / `pipeline/query_rewriter.py` | pre-retrieval | what text is embedded as the query vector | deterministic (`recall@k`) |
| `REWRITE_MODEL` | `gpt-4o-mini` | `config.py` | pre-retrieval | model for the HyDE / step-back chat call — quality of the scaffold vs. its latency/cost | deterministic (`recall@k`) + latency |
| HyDE gen params | `temperature=0.7`, `max_tokens=256` | `pipeline/query_rewriter.py` | pre-retrieval | shape of the hypothetical doc | deterministic |
| RRF `k` | 60 | `retrieval/hybrid_search.py:30` | fusion | rank-fusion smoothing constant (not a "top-k") | deterministic |
| `RETRIEVAL_TOP_K` | 20 | `pipeline/constants.py:17` | retrieve → rerank | candidate pool the cross-encoder rescores | deterministic (`recall@k`) |
| `RERANKER_MODEL` | `ms-marco-MiniLM-L-6-v2` | `retrieval/reranker.py` | rerank | cross-encoder that reorders the pool | deterministic (`nDCG@n`) |
| `RERANK_TOP_N` | 5 | `pipeline/constants.py:18` | rerank → generate | chunks the LLM actually sees | judge |
| `OPENAI_MODEL` | `gpt-4o` | `config.py` | generate | answer synthesis + citation quality | judge |
| answer system prompt | — | `pipeline/query_pipeline.py`, `evaluation/evaluate.py` | generate | grounding, citation format, refusal behaviour | judge |

## Two kinds of knob

**Retrieval-only** — the knob changes only *which chunks come back*, not what the
LLM does with them. Tune with deterministic IR metrics (`recall@k`, `MRR`,
`nDCG@n`) computed straight from the fused/ranked list against qrels: no LLM judge,
no OpenAI quota, no run-to-run noise. Covers `EMBEDDING_MODEL`,
`QUERY_REWRITE_STRATEGY`, `REWRITE_MODEL`, the HyDE params, RRF `k`,
`RETRIEVAL_TOP_K`, `RERANKER_MODEL`.

**Generator-dependent** — the knob changes what the LLM sees or is. Needs the full
pipeline + RAGAS judge + the statistical treatment below. Covers `RERANK_TOP_N`,
`OPENAI_MODEL`, the system prompt. `chunk_size` / `chunk_overlap` straddle both:
retrieval quality at a given chunk size is deterministic, but the *right* chunk
size interacts with `top_n` (bigger chunks carry more, so fewer are needed) so the
final call needs the judge.

Do the retrieval-only knobs first — they are cheap, deterministic, and they settle
the candidate set that the generator knobs are then tuned against.

## Prerequisites for a real tuning pass

1. **qrels** — for each eval question, a label of which chunks are actually
   relevant (the gold set). `ground_truth_contexts` in `test_dataset.json` is
   close but is hand-authored prose, not chunk IDs; a tuning pass needs the
   labels tied to real `docname:chunk_index` keys. **Still outstanding.**
2. **A bigger dataset** — ✅ *as large as this corpus supports.* #112 grew
   `test_dataset.json` to 92 entries (dev 39 / test 53, 8–19 per headline slice)
   and added `scripts/generate_eval_set.py` to draft candidates from the live
   corpus. This is **not** the ~150 the power analysis below wants — the
   ~38-clause demo corpus saturates with near-duplicate paraphrases past ~85
   questions. Expanding the corpus (e.g. ingesting CUAD) was considered and
   **rejected** (#129, closed won't-do): it only feeds the retrieval-only sweeps,
   which even at 150+ questions stay underpowered (~12/slice vs the ~70/slice the
   power analysis wants), and a synthetic corpus with fabricated ERPNext metadata
   would not transfer to real client corpora — which #127 validates directly.
3. **A dev/test split** — ✅ *done.* Every entry is `dev` (39) or `test` (53);
   `evaluate.py --split dev|test|all`. Tune on `dev`, freeze
   `results.baseline.json` on `test`.

### The corpus-size ceiling — what #113 can and cannot tune here

The reference corpus is ~38 clause-bearing chunks. With `RETRIEVAL_TOP_K = 20`,
the candidate pool is over half the corpus, so `recall@20` sits at the ceiling
for almost every question and the **retrieval-only sweeps cannot separate
configurations** no matter how many questions the set has. On this corpus #113
tunes:

- system prompt
- `QUERY_REWRITE_STRATEGY` — compared on the **judge headline**, not `recall@20`
- `RERANK_TOP_N`
- `OPENAI_MODEL`
- `RERANKER_MODEL` — a one-shot check (is `ms-marco-MiniLM-L-6-v2` good enough vs
  a larger cross-encoder / BGE reranker), not a sweep

`chunk_size` / `chunk_overlap`, RRF `k`, and `RETRIEVAL_TOP_K` **stay at
conventional defaults** and are not tuned here:

- RRF `k` — insensitive over {10–100}; no expected move (Step 4).
- `RETRIEVAL_TOP_K` — the recall knee shifts with corpus **size and homogeneity**,
  so it is inherently a per-client knob. A generous default (20) absorbs most
  clients; #127 nudges it up (20 → 30 → 50) when a client slice fails.
- `chunk_size` / `chunk_overlap` — driven by clause length, ~uniform across
  procurement clients.

Expanding the corpus to make these separable (#129) was **rejected** — see the
Prerequisites note above. Report retrieval at `recall@5` / `@10` rather than
`@20` if you keep any deterministic retrieval reporting — it gives the retriever
something to discriminate. (Full reasoning: #113 comment, 2026-08-28.)

## General method — applies to every knob

- **Slice by `case_class`.** The mean hides the answer. Recall and faithfulness
  degrade on the hard slices (`semantic-no-anchor`, `disambiguation`,
  `precision-multi`) — tune to those, not the aggregate.
- **Paired comparison.** Same questions across every config → **Wilcoxon
  signed-rank**, not an unpaired test. Much more sensitive.
- **Bootstrap** the per-question scores for CIs, or run each config 3× and average
  out judge noise. RAGAS scores wobble ±0.03–0.05 run-to-run on identical inputs;
  to claim config A ≠ config B the intervals must separate.
- **Power.** With per-question SD ~0.15 and wanting to detect a 0.05 gap at 80%
  power, you need ~70 questions *per slice*. Hence the ~100–200 total above.
- **Cost on the same plot.** tokens/query, $/query, p95 latency — every knob move
  has a cost axis, plot it alongside quality.
- **One knob at a time**, unless you suspect interaction (`chunk_size × top_n`) —
  then a small grid, not a line.
- **Decision rule.** The cheapest, fastest config whose quality is statistically
  within noise of the best.

## The order to tune in

Each knob's pool or behaviour feeds the next, so tune in pipeline order:

1. `chunk_size` / `chunk_overlap` — deterministic recall first, then revisit
   against `top_n` at the end (Step E).
2. `EMBEDDING_MODEL` — only if #101 / #90 make a swap real.
3. `QUERY_REWRITE_STRATEGY` (+ `REWRITE_MODEL`) — Step A.
4. RRF `k` — small deterministic sweep, k ∈ {10, 30, 60, 100}; RRF is
   insensitive to k over wide ranges, so expect no move and leave it at 60.
5. `RETRIEVAL_TOP_K` — Step B.
6. `RERANKER_MODEL` — deterministic `nDCG@5` over the fixed Step B pool; a bigger
   cross-encoder (e.g. `ms-marco-MiniLM-L-12` or a BGE reranker) trades latency
   for ranking quality.
7. `RERANK_TOP_N` — Step C.
8. `OPENAI_MODEL`, then the system prompt — Step D.

## Step A — `QUERY_REWRITE_STRATEGY` (retrieval-only)

Three arms: `hyde` (default), `step_back`, and `none` (embed the raw question).

**Priors:**

- **HyDE** searches in answer-space — helps when the question's vocabulary
  diverges from the corpus (`semantic-no-anchor`). Costs an extra `REWRITE_MODEL`
  chat call per query (~0.5–1 s on `gpt-4o-mini`; it was ~2.5 s and ~58% of
  end-to-end latency on `gpt-4o` before #120), and on a ~60-chunk corpus a
  hallucinated entity in the hypothetical doc can drag retrieval off-target.
- **Step-back** abstracts the question — helps multi-hop / "why" questions, can
  *lose* the specific anchor (supplier token, doc number) on point lookups.
- This corpus is mostly point lookups with distinctive supplier tokens, and BM25
  + reranker already handle those — so `none` is a real contender, not a
  strawman.

**Method:** for each arm, embed the (rewritten) query, run `HybridSearch.search()`,
record the fused rank of every gold chunk. Compute `recall@20` and `MRR` per
`case_class`. HyDE additionally gets a `temperature` sub-sweep (0.0 / 0.3 / 0.7) —
a hotter hypothetical doc trades precision for recall.

`REWRITE_MODEL` sub-sweep (default `gpt-4o-mini` since #120): re-run the HyDE arm
with `gpt-4o-mini` vs `gpt-4o` and diff `recall@20` / `MRR` per slice against the
`rewrite` span latency (Langfuse). #120's premise is that the scaffold only needs
to be topically close enough to embed, so the cheaper model should hold recall
while roughly halving the step. Keep the cheapest model whose recall is within
noise of the best.

**Expected outcome:** intent-dependent — HyDE for `semantic-no-anchor`, `none` for
`showcase` / `disambiguation`. If so, the fix is a cheap pre-retrieval intent
classifier that picks the strategy, not one global env var. Record the decision as
an ADR (`docs/adr/`).

## Step B — `RETRIEVAL_TOP_K` (retrieval-only)

The reranker can only reorder what it is handed; it cannot recover a chunk that
RRF ranked 25th. So the pool must be wide enough to *contain* the gold chunk, but
every extra candidate is another cross-encoder forward pass (cost linear in pool
size). On a ~60-chunk corpus, 20 is a third of everything and recall into the pool
is effectively 1.0 — which is why the current default is defensible without
tuning.

1. For each dev question, run `HybridSearch.search()` and record the fused rank of
   every gold chunk.
2. Compute **`recall@k`** for k ∈ {10, 20, 30, 50, 100} — the fraction of gold
   chunks in the top-k of the fused list. Deterministic, no API calls.
3. Plot `recall@k` and find the **knee** — the smallest k where recall plateaus
   (e.g. ≥ 0.95). Past the knee you pay reranker latency for nothing.
4. Break `recall@k` out **by `case_class`**; tune to the hard slices.
5. Cross-check against **reranker latency vs pool size** (linear in `top_k`). If
   recall is still climbing at the latency budget, take the largest k you can
   afford; otherwise stop at the knee.

Output: one `RETRIEVAL_TOP_K` value.

## Step C — `RERANK_TOP_N` (generator-dependent)

LLMs degrade with irrelevant context ("lost in the middle", distraction), and each
chunk is ~512 tokens, so 5 ≈ 2.5k context tokens/query vs ~10k for 20 — 4× the
generation cost for little gain on point-lookup questions. "3–5 passages" is the
standard for passage-grounded QA, which is why 5 is defensible without tuning.

Hold `RETRIEVAL_TOP_K` at the Step B value. Sweep `top_n ∈ {3, 5, 8, 10, 15}`. For
each, run `evaluate.py` and record **per `case_class`**:

| Signal | What it tells you |
|---|---|
| `faithfulness`, `answer_relevancy` on `showcase` | does more context *dilute* the easy cases? |
| `context_recall` on `disambiguation` / `precision-multi` | does more context *recover* multi-answer material? |
| `context_precision` | drops mechanically as `top_n` rises — watch, don't over-index |
| answer correctness (exact / human, not just RAGAS) | the ground truth |
| tokens/query, $/query, p95 latency | the cost axis, on the same plot |

The value is the inflection point: where `context_recall` on the hard slices
**stops improving** but `faithfulness` and cost **keep getting worse**.

The one place `top_n = 5` clearly hurts is enumeration ("list all cancelled
contracts" drops the 3rd match — see `docs/ARCHITECTURE.md` § Known Limitations).
Raising `top_n` is the *wrong* fix there; that needs a separate metadata-filtered
retrieval path (#45). Enumeration is scoped out rather than distorting `top_n` for
the common case.

## Step D — `OPENAI_MODEL` and the system prompt (generator-dependent)

- **Model swap** — a judge + cost pass over the same dev set (`gpt-4o` vs
  `gpt-4o-mini` vs a newer model vs a non-OpenAI model behind the #90 seam).
  Watch `faithfulness` and citation-format compliance, not just answer_relevancy;
  cheaper models tend to drop `[docname]` citations first.
- **Prompt changes** — judged on `faithfulness` and refusal behaviour. This is
  the one knob where the `expected_fail` refusal cases (`aggregation`,
  `temporal`) matter most: a prompt tweak that makes the model answer instead of
  refusing is a regression even if the headline moves up.
- Either change means refreshing `evaluation/results.baseline.json` **in the same
  PR** (per `CLAUDE.md` § CI).

## Step E — expect the answer to not be a single number

Likely conclusions:

- `RETRIEVAL_TOP_K` = one global value (recall plateaus).
- `QUERY_REWRITE_STRATEGY` and `RERANK_TOP_N` = **intent-dependent** — a rewrite
  and `top_n` of 5 for point lookups, `none`/wider (or the metadata-filter path)
  for enumeration / comparison intents. This means classifying intent before
  retrieval, or always retrieving wide and letting the generator's own filtering
  handle the common case.
- `chunk_size` **interacts** with `top_n` — bigger chunks carry more per chunk, so
  fewer are needed. A thorough pass sweeps `chunk_size × top_n` as a small grid
  rather than fixing one.

## Per-client tuning — tune once, validate per client

Everything above is the **one-time** pass on the reference corpus (#113): it
produces the shipped defaults in `config.py` / `pipeline/constants.py` and the
baseline thresholds. It is **not** re-run for every client. Per-client work is a
lighter validation pass (#127) — a small generated + spot-checked eval set on the
client's corpus, checked against the reference thresholds, with a targeted
single-knob nudge only when a `case_class` slice fails.

### Which knobs are corpus-dependent

| Knob | Per client? | Why |
|---|---|---|
| `chunk_size` / `chunk_overlap` | No | driven by document *structure* (contract / T&C clause length), ~uniform across procurement clients |
| `EMBEDDING_MODEL` | No | global product decision (#101 / #90) |
| `QUERY_REWRITE_STRATEGY` + HyDE params | No | depends on question *intent*, not the corpus — and Step A may make it an adaptive classifier anyway |
| `REWRITE_MODEL` | No | global latency/cost vs. scaffold-quality choice; corpus-independent |
| RRF `k` | No | fusion constant, insensitive over wide ranges |
| `RETRIEVAL_TOP_K` | **Weakly** | recall knee shifts with corpus **size** and **homogeneity** (more docs / more boilerplate → gold chunk buried deeper). A bigger pool costs reranker compute only, not generation — so a generous default (20–30) absorbs most clients; revisit only for a genuinely large or homogeneous corpus |
| `RERANKER_MODEL` | No | global quality / latency choice |
| `RERANK_TOP_N` | **Weakly** | sensitive to the client's *question style* (single-clause lookup vs cross-document synthesis), not their contracts |
| `OPENAI_MODEL` | No | global |
| answer system prompt | Mostly no | may get light per-client *vocabulary* edits ("agreements" not "contracts", client-specific status terms) — localization, not tuning |

Not knobs, but genuinely per-client — and set from knowing the client's ERPNext,
not from an eval sweep:

- **Metadata filter vocabulary** — `_DOCTYPE_KEYWORDS` / `_STATUS_KEYWORDS` in
  `pipeline/query_pipeline.py`, if the client uses custom doctypes or status
  values. Config.
- **Ingestion field mapping** — whether the client's custom fields / doctypes are
  wired into `ingestion/`. An ingestion problem, not tuning.

### Reach order when a per-client slice fails

Per-client validation is mostly a pass / confirm exercise — the defaults are
expected to hold. When a `case_class` slice comes in below the reference
threshold, adjust in this order and stop as soon as the slice recovers:

1. **`RETRIEVAL_TOP_K`** — raise it (e.g. 20 → 30 → 50) and re-check `recall@k`
   into the pool for the failing slice. Cheapest fix, deterministic, no judge.
2. **System prompt** — client-vocabulary or grounding edits, if the failure is
   citation format / refusal / terminology rather than retrieval.
3. **`RERANK_TOP_N`** — only if the failure is context *recall* on
   multi-document questions and step 1 did not fix it.

A full Step A–E sweep per client is almost never warranted; if one looks
necessary, the client's corpus differs from the reference enough that #113 itself
should be re-examined.

## Related

- `docs/ARCHITECTURE.md` § Evaluation — the harness and `case_class` slices
- `docs/ARCHITECTURE.md` § Known Limitations — why enumeration/temporal are out of scope
- `pipeline/query_rewriter.py` — the two rewrite strategies
- `config.py`, `pipeline/constants.py` — where the knobs live
- `docs/MODEL_PROVIDER_SWAP.md` — swapping `OPENAI_MODEL` / `EMBEDDING_MODEL` provider
- #102 — the harder eval dataset (prerequisite)
- #112 — reference tuning set (92 entries + dev/test split) + `scripts/generate_eval_set.py`
- #129 — CUAD corpus expansion, **closed won't-do**: only feeds retrieval-only
  sweeps that stay underpowered anyway, and synthetic metadata would not transfer
- #113 — the one-time reference-corpus tuning pass this doc describes
- #127 — the per-client validation workflow (uses #113's defaults + thresholds)
- #101 / #90 — local embedding model / provider-agnostic model seam
- #49 — CI threshold gating, which also wants per-`case_class` numbers
- #45 — the enumeration investigation, closed as out of scope
