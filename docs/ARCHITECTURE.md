# Architecture

## System Overview

```
ERPNext
  │  REST API (initial full ingest)
  │  Webhooks (incremental updates)
  ▼
┌─────────────────────────────────┐
│        Ingestion Layer          │
│  erpnext_client → document_     │
│  parser → chunker → embedder    │
└──────────────┬──────────────────┘
               │ upsert (idempotent, by docname+chunk_index)
               ▼
┌──────────────────────────────────┐
│           Qdrant                 │
│  Collection: contract            │
│  Indexed payload: source_doctype,│
│  supplier, status, start_date,   │
│  end_date                        │
└──────────────┬───────────────────┘
               │ queried by step 3 below (vector + BM25 lookup)
               ▼
┌──────────────────────────────────┐
│      Query Pipeline (per request)│
│  1. QueryRewriter (HyDE/step-back)│  ← entry point, runs before retrieval
│  2. filter extraction (keywords) │
│  3. HybridSearch → RRF → top-20  │
│  4. Cross-encoder rerank → top-5 │
│  5. Prompt builder → GPT-4o      │
│  6. Answer + source citations    │
└──────────┬──────────┬────────────┘
           │          │ Langfuse spans
           ▼          ▼
        FastAPI    Langfuse
        /query      (trace)
           │
           ▼
        Streamlit
```

The steps are execution order. Retrieval is *inside* the pipeline (step 3), driven
by the rewritten query from step 1 — not a separate stage that runs first and hands
the pipeline a top-5. See "Query Pipeline — Step by Step" below.

## Query Pipeline — Step by Step

1. **Query rewriting** — HyDE generates a hypothetical answer; its embedding becomes the query vector. This improves recall for abstract questions.
2. **Metadata filter extraction** — pure keyword matching (`_extract_filters()` in `query_pipeline.py`) against doctype and status keyword lists in the original question. No LLM call and no date-range parsing — only `source_doctype` and `status` are ever set this way; date/supplier filters come solely from the frontend sidebar.
3. **Hybrid search** — BM25 (lexical) and Qdrant vector search run in parallel. Results are fused with Reciprocal Rank Fusion (`k=60`), returning 20 candidates. Metadata filters (supplier, doctype, status) are honoured by **both** legs: Qdrant filters during ANN traversal; `rank_bm25` has no filter hook, so the ranked BM25 list is filtered in Python before fusion (`_passes_filter`, mirroring `VectorStore._build_filter`). Every chunk in the fused set therefore satisfies the filter — it's a hard constraint (#98).
4. **Cross-encoder reranking** — `ms-marco-MiniLM` scores all 20 `(query, chunk)` pairs and returns the top 5.
5. **Generation** — GPT-4o receives the top-5 chunks as context with a structured prompt that requires source citations.
6. **Tracing** — each step is a Langfuse child span; the full trace is linked to the question.

### Known Limitations — aggregation/enumeration queries

This pipeline is scoped to passage-grounded Q&A: retrieval is always top-20 hybrid search reranked to
a fixed top-5 before generation. It does **not** reliably compute exact sums, counts, or full
enumerations over many matching records (e.g. "total PO amount to supplier X this month," "list all
open POs for supplier Y") — any entity with more matching records than the top-5 budget will have some
silently dropped before the LLM ever sees them, since the reranker optimizes for "best passage(s) for
one question," not "every record matching a filter." Supporting that query class reliably would
require a second, parallel retrieval path (an exact metadata-filtered fetch, e.g. Qdrant `scroll` by
`supplier`/`source_doctype`, bypassing top-k semantic rerank entirely) plus aggregation-aware
generation — a different architecture, not a tuning fix. See #45 for the original investigation and
root-cause analysis; closed as out of scope.

### Known Limitations — time-relative / temporal queries

The pipeline cannot reliably answer questions that depend on "now" — "which contracts are active
today," "what expires in the next 3 months," "anything signed since March." Two independent causes:

1. **No date arithmetic anywhere in the query path.** `_extract_filters()` is keyword-only (doctype
   and status); there is no date-range parsing and no comparison of `start_date`/`end_date` against
   the current date. A question like "expiring soon" is handled as plain semantic search over clause
   text, which has no notion of the calendar.
2. **`status` goes stale, and a time-based change never reaches the index.** ERPNext derives a
   Contract's `status` (`Active`/`Inactive`/`Unsigned`) from `is_signed` + the date range. Ingestion
   copies whatever value is current into the Qdrant payload. It refreshes only when the webhook
   fires — and the webhook fires on `on_update`/`on_submit`/`on_cancel`, i.e. a **human editing and
   saving** the Contract. ERPNext's own daily refresh
   (`contract.update_status_for_contracts()`) writes with `frappe.db.set_value`, a direct DB write
   that runs no document events, so it does **not** fire the webhook — an `Active → Inactive` flip
   from a lapsed `end_date` is invisible to the RAG index. (On the demo bench the scheduler is
   disabled outright, so even ERPNext's `status` field is stale until a contract is re-saved.)

Supporting temporal queries would need date-aware filter extraction plus a retrieval path that
evaluates ranges against `datetime.now()` at query time, and a scheduled re-ingest (or a
status-change hook that actually fires the webhook) to keep `status` fresh — a different
architecture, not a tuning fix. The `temporal` slice of `evaluation/test_dataset.json` covers this
boundary as `expected_fail` cases; the reference answers describe the correct result *and* why the
pipeline can't produce it.

## Model Configuration

Generation and embedding model names are centralized in `config.py`, set via the `OPENAI_MODEL`
(default `gpt-4o`) and `EMBEDDING_MODEL` (default `text-embedding-3-small`) env vars — mirroring
the `QUERY_REWRITE_STRATEGY` pattern. Every call site (`query_rewriter.py`, `query_pipeline.py`,
`embedder.py`, `evaluate.py`) reads from `config.py` instead of hardcoding a model literal.

`REWRITE_MODEL` (default `gpt-4o-mini`) is a third, separate knob for the pre-retrieval HyDE /
step-back chat call in `query_rewriter.py`. It's deliberately not `OPENAI_MODEL`: the rewrite only
needs a scaffold paragraph to embed, and on `gpt-4o` it was ~58% of end-to-end query latency
(issue #120). `evaluate.py`'s `_run_config()` records `rewrite_model` alongside `rewrite_strategy`.

Chunking (`CHUNK_SIZE`, default `512`; `CHUNK_OVERLAP`, default `64`) is centralized the same way,
read by `ingestion/chunker.py`'s `chunk_text()` default arguments. `evaluate.py`'s `_run_config()`
records `chunk_size`/`chunk_overlap`/`embedding_model` alongside the query-time knobs it already
tracked, so a `results.json` baseline is self-describing about what produced the indexed Qdrant
collection it scored — not just what query-time settings ran. `evaluate.py` does not re-run
ingestion itself: re-chunking or re-embedding requires a deliberate `POST /ingest/full` first, then
`evaluate.py` scores whatever is currently indexed.

**Caveat — embedding model swaps:** `retrieval/vector_store.py`'s `VECTOR_DIM` is derived from
`EMBEDDING_MODEL` via `config.embedding_dimension()`, which maps model name → vector size. The
Qdrant collection's vector size is fixed at creation time (`ensure_collection`), so switching to an
embedding model with a different dimension requires recreating the collection and running a full
re-ingest — existing points are not re-embedded automatically. Unrecognized model names raise at
import time until their dimension is added to `config._EMBEDDING_DIMENSIONS`.

### Future Enhancement — provider-agnostic adapter (not implemented)

`OPENAI_MODEL`/`REWRITE_MODEL`/`EMBEDDING_MODEL` (above) solve swapping the OpenAI *model version*,
not the *provider* — every call site (`query_rewriter.py`, `query_pipeline.py`, `embedder.py`,
`evaluate.py`) still constructs a raw `openai.OpenAI()` client and reads its response shape
directly (`response.choices[0].message.content` / `response.data[i].embedding`).

A genuine provider swap would need a small hand-rolled seam: a `ChatModel`/`EmbeddingModel`
interface (two methods each) that every call site depends on instead of a concrete SDK, an
`OpenAIChatModel`/`OpenAIEmbeddingModel` adapter wrapping today's existing `openai`-SDK code behind
that interface (a mechanical move, not a rewrite), and `LLM_PROVIDER`/`EMBEDDING_PROVIDER` env vars
(default `openai`) dispatching to the right adapter class.

LangChain was considered for this instead of a hand-rolled adapter — its `init_chat_model` factory
exists in the pinned `langchain` version and would normalize the response shape across providers in
one move. Rejected: it's real added dependency/version-risk surface for a swap that isn't happening,
versus a hand-rolled adapter that stays fully within the codebase's own control and is no larger for
the one provider (OpenAI) actually in use today. `langchain` is kept only for
`RecursiveCharacterTextSplitter` in `ingestion/chunker.py`; `langchain-openai` and
`langchain-community` were declared but never imported and have been removed (ADR 0002).

Either approach touches the OpenAI-response-shape mocking in `tests/test_query_rewriter.py`,
`tests/test_query_pipeline.py`, and `tests/test_embedder.py` (currently hand-rolled `MagicMock`
chains matching `.choices[0].message.content` / `.data[i].embedding`). Deferred — see #51; not
prioritized since no provider swap is currently planned.

The full "what LangChain core provides, what this project uses instead, and why" comparison —
covering model wrappers, LCEL, loaders, retrievers, memory, callbacks, and LangGraph — lives in
**ADR 0002** (`docs/adr/0002-no-langchain-framework.md`).

For the concrete steps to switch provider with today's codebase as-is (no adapter layer), see
`docs/MODEL_PROVIDER_SWAP.md`.

## Document Indexing Strategy

Both ingested doctypes (Contract, Terms and Conditions) are unstructured text, chunked the same way:

- HTML stripped via BeautifulSoup before chunking
- `RecursiveCharacterTextSplitter`: `chunk_size=512`, `chunk_overlap=64`
- Each chunk stores `chunk_index` and `total_chunks` for reconstruction

### PDF attachments

`Contract` documents may carry attached PDFs (e.g. signed contract scans). For this doctype,
`erpnext_client.get_attached_files()` lists the document's `File` records, `.pdf`-named ones are
downloaded via `get_file_content()`, and their text is extracted via `extract_text_from_pdf()`. The
extracted text is chunked the same way as any other unstructured text and indexed as **additional
points under the same `docname`** as the parent document — `chunk_index`/`total_chunks` are
renumbered across the combined set (parent text + attachments) so `delete_by_docname` and
reconstruction stay correct. A failure to list attachments or extract one PDF's text is logged and
skipped; it never blocks indexing of the parent document or other attachments.

### Supplier Metadata Enrichment

`supplier_group` (used in the payload schema and as a filter field) is **not** present on `Contract`
records directly — it only exists on `Supplier`. `ingestion/webhook_handler.py`'s
`resolve_supplier_group()`/`_fetch_supplier_group()` fetch the `Supplier` record and join
`supplier_group` onto the metadata dict built in `prepare_doc_for_indexing()` — this fetch is **not
cached**; it hits ERPNext fresh on every call. `Terms and Conditions` has no supplier field, so this
lookup naturally no-ops for it.

### Qdrant Payload Schema

Payload is **flat** — every metadata field sits at the top level alongside `text`, not nested under
a `metadata` key (`retrieval/vector_store.py`'s `upsert_chunks()` stores
`{k: v for k, v in chunk.items() if k != "vector"}` directly as the point payload, so filter
expressions can reference fields like `supplier` or `status` without a nested path):

```json
{
  "text": "...",
  "source_doctype": "Contract",
  "docname": "CON-2024-00042",
  "supplier": "Supplier-XYZ",
  "supplier_group": "Hardware",
  "start_date": "2024-01-01",
  "end_date": "2025-01-01",
  "status": "Active",
  "company": "My Company",
  "chunk_index": 2,
  "total_chunks": 8
}
```

Point ID is derived as `uuid5(NAMESPACE_DNS, f"{docname}:{chunk_index}")` — deterministic, enabling idempotent upserts.

`Contract` carries two additional, optional payload fields — `linked_doctype`/`linked_docname` — populated from
the doctype's `document_type`/`document_name` Dynamic Link pair when the contract references a specific
transaction document (this is a generic Frappe Dynamic Link and works for any doctype, e.g. a Sales Order).
A matching sentence ("This contract is linked to Sales Order SO-2024-00123.") is also prepended to the
embedded/chunked text so the link is retrievable via semantic/BM25 search without requiring a metadata filter.

## Incremental Indexing via Webhooks

ERPNext fires webhooks on the following events:

| Doctype             | Event       | Effect                                                  |
|---------------------|-------------|----------------------------------------------------------|
| Contract            | `on_submit` | Indexed when contract is submitted (incl. attached PDFs)|
| Contract            | `on_update` | Re-indexed on any desk save (incl. attached PDFs)       |
| Contract            | `on_update_after_submit` | Re-indexed on an `allow_on_submit` field edit (e.g. `is_signed`) after submit |
| Contract            | `on_cancel` | Re-indexed with status=Cancelled                        |
| Terms and Conditions| `on_update` | Re-indexed (not a submittable doctype)                  |

> **Known gap (PO only, out of scope):** `on_update_after_submit` for Purchase Orders is not supported — Frappe does not fire this webhook event via API or desk UI saves for PO, and PO ingestion is descoped so this isn't being pursued. The same gap was also confirmed for Contract and has since been fixed (#96) — see `docs/DEPLOYMENT.md` § Future Enhancements → "Post-submit edits are now re-indexed" for the root cause and fix.

The webhook handler:
1. Verifies `X-Frappe-Webhook-Signature` (HMAC-SHA256, base64-encoded)
2. Fetches the full document via REST API
3. Calls `vector_store.delete_by_docname(docname)` — removes all old chunks
4. Re-runs the full parse → chunk → embed → upsert pipeline for that document only

This ensures updated contract terms are reflected in the next query without a full re-index.

## BM25 Index

The BM25 index is built in memory at API startup by fetching all payload texts from Qdrant. It is rebuilt after each webhook upsert to stay current. For large collections (>100k docs), consider moving to Qdrant's sparse vector support instead.

### Filter behaviour

Metadata filters (supplier, doctype, status) are applied to **both** legs of the hybrid search. Qdrant filters during ANN traversal; `rank_bm25` has no native filter hook, so `HybridSearch.search` walks the BM25-ranked list and drops chunks that fail `_passes_filter` (the same non-None exact-match AND used by `VectorStore._build_filter`) before RRF fusion, re-ranking the survivors densely so both legs feed RRF the same rank space.

- Every chunk in the fused top-20 satisfies the filter — it is a hard constraint, not a soft rank bias (fixed in #98; before that the BM25 leg was corpus-wide and a lexical hit from a non-matching supplier could survive fusion).
- Date and multi-select filters: `_passes_filter` treats a list/tuple/set condition as membership, but the Qdrant leg's `_build_filter` still uses scalar `MatchValue`, so a multi-select doctype filter isn't yet a true OR end-to-end. Date filters remain exact-equality on `start_date`/`end_date` (no range) on both legs — see Known Limitations.
- `_parse_sources` only includes documents GPT-4o explicitly cites, so even before #98 uncited leakage was discarded downstream; the fix makes the candidate pool itself correct.

## Observability

### Query trace

Every query creates a Langfuse trace with the following child observations:

| Observation | Type | Input captured | Output captured |
|---|---|---|---|
| `rewrite` | span | original question | rewritten text + query vector |
| `filter_extraction` | span | original question | `{source_doctype?, status?}` dict |
| `hybrid_search` | span | rewritten query + filters | list of 20 scored chunks |
| `rerank` | span | original question + 20 candidates | top-5 re-scored chunks |
| `generate` | generation | original question + context string | answer with `[docname]` citations |

The trace root carries `input.question` and `output.{answer, source_count}`. If any step raises, the failing span is ended with `level="ERROR"` (the root `trace.update(level=...)` call is a historical no-op — Langfuse traces have no `level` field, only observations do).

`generate` is a Langfuse **generation** (not a plain span) — it's created with `trace.generation(model=OPENAI_MODEL)` and passes `usage={"input", "output", "total"}` (from the OpenAI response's `.usage`) to `end()`. Only `generation`-type observations get token counts and cost auto-computed by Langfuse; a plain span just stores whatever input/output you hand it (see PR [#87](https://github.com/arunjoyt/contract-intelligence/pull/87), issue #81).

### Ingestion traces

Both re-indexing paths are traced the same way (issue [#123](https://github.com/arunjoyt/contract-intelligence/issues/123)). The shared helpers live in `ingestion/tracing.py` (`span()` context manager + `traced_embed()`), and every helper no-ops when no Langfuse client is wired in, so ingestion runs unchanged without credentials.

**`webhook_reindex`** — one trace per `POST /webhook/erpnext` call that hits a supported doctype (ignored doctypes create no trace). Root `input` is `{doctype, docname}`; root `output` is `{status, chunk_count}` (or `{status: "skipped", reason}`, or `{status: "error", error}` on a failure). Langfuse traces carry no `level` field — only observations do — so on a failure the ERROR signal is on the span for the step that raised; the root `output.status` is the coarse marker.

| Observation | Type | Notes |
|---|---|---|
| `fetch` | span | `get_doc` + Supplier-group join |
| `parse` | span | HTML strip + struct→NL serialization |
| `chunk` | span | split + PDF-attachment extraction; `output.chunk_count` |
| `embed` | generation | `model=EMBEDDING_MODEL`, `usage={"input", "total"}` from the embeddings API |
| `upsert` | span | `delete_by_docname` + `upsert_chunks` |

**`full_ingest`** — one trace per `POST /ingest/full` run. A `list:<doctype>` span wraps each ERPNext listing call (so a run that indexes nothing because the listing failed shows the ERROR span, not just a bare `0/0` output), then each document gets one `<doctype>:<docname>` span containing its own `embed` generation, so per-document latency and cost are both visible. Root `output` is `{documents_indexed, chunks_indexed}`. A single document (or one doctype's listing) failing is logged and its span marked `ERROR`, but the run continues and its trace is not marked — one bad doc is not a failed ingest.

The embeddings API reports no `completion_tokens`, so the `embed` generation's `usage` has no `"output"` key — this is the first place ingestion embedding cost (`text-embedding-3-small`) shows up in Langfuse; before #123 it was untracked on both paths.

The Langfuse UI is accessible at `http://localhost:3000` (default docker-compose port).

### How to verify in the Langfuse UI

1. Start all services including Langfuse: `docker compose up qdrant langfuse postgres -d`
2. Open `http://localhost:3000` → log in (credentials from `.env`: `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY`)
3. Run the API with Langfuse wired in: `uvicorn api.main:app --reload`
4. Send a query: `curl -s -X POST http://localhost:8000/query -H 'Content-Type: application/json' -d '{"question":"What are the payment terms for our active contracts?"}'`
5. In the Langfuse UI → **Traces** → select the new trace and verify:
   - Root trace `name` is `query`; `input.question` matches the sent question
   - Five child observations appear: `rewrite`, `filter_extraction`, `hybrid_search`, `rerank`, `generate`
   - Each has a non-zero duration and no error level
   - `generate` shows non-zero `Total tokens` and a computed cost (it's a `generation`, not a plain span)
   - Root trace `output.answer` is non-empty and contains at least one `[docname]`-style citation
   - On a forced error, `level` is `ERROR` on the failing span and propagates to the root trace
6. Trigger an ingestion path and check its trace:
   - Edit a Contract in ERPNext (or `curl -X POST .../ingest/full -H "X-Admin-Secret: ..."`) and refresh **Traces**
   - A `webhook_reindex` trace (or `full_ingest`) appears with `fetch` / `parse` / `chunk` / `embed` / `upsert` observations
   - `embed` is a `generation` with non-zero `Total tokens` and a computed cost

### Future Enhancements — production quality monitoring (not implemented)

Tracing tells you a query ran and how long it took; it says nothing about whether the answer was any good. RAGAS (`evaluation/evaluate.py`) only scores a fixed golden dataset on a manual run, so quality drift on real production questions currently goes undetected. Two independent options, not mutually exclusive:

1. **User feedback** — add thumbs up/down to the Streamlit chat UI (`frontend/app.py`), and on click call `Langfuse.score(trace_id=..., name="user_feedback", value=1|0)` against the trace ID of that query. Cheap to build; signal is real but low-volume (depends on users bothering to click).
2. **LLM-as-judge sampling** — a periodic job (e.g. nightly, similar cadence to the deferred RAGAS-nightly idea above) pulls a sample of recent production traces from Langfuse via its API, runs an LLM judge over each `{question, answer, sources}` triple for faithfulness/relevancy, and writes the result back as a Langfuse score. Closer to RAGAS's signal, works without user participation, but costs an extra LLM call per sampled trace and needs a judge prompt/rubric.

See issue tracking this in the GitHub roadmap (#17, Future Enhancements).

## Security Notes

- ERPNext API key/secret never leaves the server — the webhook handler only exposes an HMAC-protected endpoint
- The `/ingest/full` endpoint is protected by `X-Admin-Secret` header
- All secrets are loaded from environment variables; `.env` is gitignored
- No user input is interpolated into Qdrant filter expressions directly — filters are constructed programmatically

### Auth — implemented (Option B)

`POST /query` requires a valid JWT (`Authorization: Bearer <token>`). The JWT is minted by FastAPI after completing the ERPNext OAuth2 Authorization Code + PKCE flow. Role enforcement (`Purchase Manager`, `Purchase User`, `Accounts User`, `System Manager`) happens at the point where the access token is exchanged — unauthorized users receive `403`. JWT lifetime is controlled by `JWT_EXPIRY_HOURS` (default 8h). See `docs/DEPLOYMENT.md` § Option B for setup details.

### Known Limitation — access control is role-level, not document-level

Authorization is a single yes/no gate: does the authenticated user hold one of the four allowed
roles. Once past that gate, every user sees the same answers and sources, drawn from the **entire**
indexed corpus — the JWT's role claim is checked once at the API boundary (`api/auth/dependencies.py`)
and never passed into `QueryPipeline.run()`, `HybridSearch.search()`, or `VectorStore`. There is no
per-document, per-company, or per-department filtering: ingestion pulls every record of the five
target doctypes into one shared Qdrant collection using a single, Administrator-level ERPNext API
key (`ingestion/erpnext_client.py`), independent of whatever row-level User Permissions ERPNext
itself enforces on those documents in the Desk UI.

This is acceptable as long as the four allowed roles are already meant to see the same procurement
data in ERPNext (i.e. ERPNext isn't using User Permissions to scope some of those roles to specific
companies/departments today). It stops being acceptable the moment that assumption breaks — e.g. a
new company or restricted department is onboarded into the same ERPNext instance and expects its
data walled off from other users holding the same role. Enforcing that would require passing the
authenticated user's identity/permissions through the query pipeline and into the Qdrant filter
(`filter_conditions` in `retrieval/vector_store.py`), not just gating `/query` at the door. See #60
for the decision record — accepted conditionally, reopen if the assumption above breaks — and
`docs/DOCUMENT_LEVEL_ACCESS_CONTROL.md` for the proposed solution.

## Evaluation

`evaluation/evaluate.py` scores the full query pipeline (rewrite → hybrid search → rerank →
generate) against `evaluation/test_dataset.json` and writes `evaluation/results.json` (gitignored —
a per-run artifact). It is a **manual local run** — not wired into CI (see below). `evaluation/results.baseline.json`
is a committed, deliberately-updated frozen run: the reference for spotting regressions by eye.
Refresh it (copy a clean `results.json` over it) in the same PR as any pipeline change that is
meant to move the numbers.

### Not in CI

The eval is deliberately not a merge gate. An LLM-judge score over ~15 headline questions wobbles
±0.03–0.05 run-to-run (see Caveats), so a hard threshold would fire on noise and get muted; and the
harness only exercises retrieval-ranking + generation over an already-indexed corpus — chunking and
parsing changes don't show up until a re-ingest. Run it by hand when you touch the rewriter, fusion,
reranker, or the generation prompt, and compare against `results.baseline.json`. An automated gate is
only worth building once the dataset is substantially larger (order 150+ questions).

### The dataset

~22 entries grounded in the current `scripts/seed_data/demo_data.yaml` fixtures with real
`CON-YYYY-NNNNN` docnames. Each carries `question`, `ground_truth_contexts`, `ground_truth_answer`,
a free-text `capability` label, a `case_class` (the slice key), and an optional `expected_fail`
flag. `evaluate.py` reports one headline metric set plus a `metrics_by_case_class` breakdown.

| `case_class` | What it stresses |
|---|---|
| `showcase` | One question per row of the README **What It Does** table — a distinct component each (BM25 exact-term, vector paraphrase, reranker cross-document, HyDE abstract, plain-language↔field, PDF). Easy by design. |
| `disambiguation` | A supplier with two contracts (Alpha current vs superseded, Telstar primary vs backup, Vertex procurement vs support, Ashford live vs cancelled) — the near-duplicate must not be cited. |
| `precision-multi` | "Which suppliers do X" — retrieval must return *all and only* the matches, against a top-5 rerank budget. |
| `semantic-no-anchor` | No supplier name in the question — BM25 has nothing to grab, the dense side has to carry it. |
| `refusal` | `ground_truth_contexts: []`. Globex (never seeded) and a clause genuinely absent from a real contract (force majeure in the Telstar agreement). |
| `aggregation` | `expected_fail: true`. Counting / averaging / enumerating over the corpus — the top-5 budget drops records, and there is no arithmetic. |
| `temporal` | `expected_fail: true`. "Active now", "expires in 6 months", "most recent" — no date arithmetic in the query path, and `status` in the index is stale (see Known Limitations above). |

`aggregation` and `temporal` are both `expected_fail` — scored so the gap stays visible, but
excluded from the headline so they don't mask a real regression elsewhere.

`ground_truth_contexts` are written in the framed form the generator sees —
`[docname] (status: …; supplier: …): text` — because some answers depend on payload metadata
(e.g. `status: Unsigned`) that never appears in the chunk's own text. `evaluate.py` frames the
retrieved chunks the same way (`_frame_chunk`) so RAGAS compares like with like.

`evaluate.py` imports the system prompt, `CONTEXT_META_FIELDS`, and the `top_k` / `top_n` /
`max_tokens` values from `pipeline/constants.py` — the same module `query_pipeline.py` uses — so a
baseline run exercises the exact prompt and retrieval budget the live pipeline does. `results.json`
records a `config` block (rewrite strategy, judge model, generation model, top-k/top-n, chunking/
embedding knobs, pinned library versions) so a baseline is self-describing and two runs are
comparable (#110, #117).

### Langfuse Dataset + Experiment logging (optional, additive)

When `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` are set, `evaluate.py` also records each question as
a Langfuse trace (input/output/metadata) and pushes the four RAGAS scores onto it via
`langfuse.score()`. `evaluation/push_dataset.py` is a separate, idempotent one-off script that syncs
`test_dataset.json` into a Langfuse Dataset (`LANGFUSE_EVAL_DATASET`, default
`contract-intelligence-golden-set`) — items upsert by an id derived from the question text
(`evaluation/langfuse_dataset.py`), so re-running it after editing the dataset doesn't duplicate
items. If a question was pushed first, `evaluate.py` links its trace into a Langfuse Dataset **run**
named after the current git short SHA, so two pipeline configs' scores are comparable side-by-side in
the Langfuse UI instead of eyeballing two `results.json` files (#109).

The RAGAS judge LLM/embeddings are passed explicitly to `ragas_evaluate()`
(`ragas.llms.llm_factory` / `ragas.embeddings.embedding_factory`, pinned to `gpt-4o-mini` /
`text-embedding-ada-002` in `evaluate.py`) rather than left to ragas's implicit default, so a run's
judge is reproducible and visible. These are pinned to ragas's own existing defaults, not this app's
`OPENAI_MODEL`/`EMBEDDING_MODEL` — `answer_relevancy` is embedding-model-sensitive, and swapping in
`EMBEDDING_MODEL` (`text-embedding-3-small`) measurably shifted it against `results.baseline.json` in
testing, breaking historical comparability for no benefit.

This whole layer is additive and optional: `evaluate.py` runs identically, and `results.json`/
`results.baseline.json` are written exactly as before, with no Langfuse credentials at all — the
offline path this repo's own baseline relies on is unchanged.

**Client deployments (#118):** `test_dataset.json`/`results.baseline.json` in *this* repo are
synthetic demo data and stay committed. For a real client, point `LANGFUSE_PUBLIC_KEY`/
`LANGFUSE_SECRET_KEY`/`LANGFUSE_HOST` at their own Langfuse project and run `push_dataset.py` against
their own (uncommitted) dataset file — their golden-set questions and scores live only in their
project, never in this repo's git history. `evaluate.py`'s code path is identical either way.

`evaluate.py` runs against whatever is already in the Qdrant collection and exits with an error if
it is empty — a real run needs a live, fully-ingested collection so retrieval, chunking and parsing
are all reflected. (The `@live` integration tests in `tests/test_integration.py` seed the
`ground_truth_contexts` strings into a throwaway collection themselves, purely to smoke-test that
the script produces a valid `results.json`.)

### Metrics — what the LLM judge actually does

The ground truth is a *reference*, not an answer key: the pipeline's free-form answer and
arbitrary chunk boundaries will never string-match a hand-authored reference, so RAGAS uses an LLM
judge for semantic comparison. The four metrics use the reference differently — two not at all:

| Metric | Uses `ground_truth_answer`? | What the judge does |
|---|---|---|
| `faithfulness` | No — checks against *retrieved contexts* | Splits the generated answer into atomic claims; verifies each is entailed by the retrieved chunks. Hallucination check. |
| `answer_relevancy` | No — reference-free | Generates questions from the answer, embeds them, compares to the real question. Catches evasive / off-topic answers. |
| `context_recall` | Yes | Splits the reference answer into claims; checks each is supported by the retrieved chunks. "Did retrieval fetch enough?" |
| `context_precision` | Yes | Judges whether each retrieved chunk helped produce the reference answer, rank-weighted. "Is retrieval returning signal, well-ordered?" |

Drop the ground truth and only the two reference-free metrics remain computable.

### The refusal cases

`refusal` entries have `ground_truth_contexts: []`. RAGAS's answer/context metrics are meaningless
for "correctly answered nothing," so `evaluate.py` scores them with a plain refusal-string match
(`_is_refusal` → `refusal_handled` fraction) and keeps them out of the RAGAS set. "Did it refuse" is
a discrete outcome — string matching is both correct and free here, the same reason
`proj-ticket-rag` can use `ranx` rank-math for its retrieval-only eval.

The absent-clause case (force majeure in the Telstar contract) is harder than Globex: retrieval
*does* return relevant Telstar chunks, they just don't mention force majeure. In the baseline run
the model still emitted the canonical refusal string here rather than fabricating a clause — both
refusal cases passed. If a future change makes it answer "the provided terms don't mention force
majeure" instead, `_is_refusal` will count that as a miss; that would be a signal to loosen the
check, not necessarily a regression.

### Interpreting the numbers

Baseline run (2026-08-27, `gpt-4o`, frozen in `results.baseline.json`) — headline over the 15
non-`expected_fail` questions: faithfulness 0.85, answer_relevancy 0.93, context_recall 0.89,
context_precision 0.91, refusal 2/2. By slice:

| slice | n | what the numbers say |
|---|---|---|
| `showcase` | 6 | ~0.88 / 0.95 / 0.92 / 0.93 — high, but **easy by construction** (README questions, a distinctive supplier token each, near-verbatim ground truth, a ~60-chunk corpus where the top-20 pool holds a third of everything so `context_recall` near 1.0 is nearly free — `proj-ticket-rag` hit the same ceiling). A "not broken" signal, not a quality measurement. |
| `disambiguation` | 4 | `context_precision` **1.00** — the reranker cites the right contract (current vs superseded, primary vs backup, live vs cancelled) every time. This is the slice a retrieval regression would break first. |
| `precision-multi` | 2 | Answers list all matches correctly, but `context_precision` **0.50** — the top-5 pool carries irrelevant contracts. The known top-k limitation, made visible without the answer being wrong. |
| `semantic-no-anchor` | 3 | `context_recall`/`context_precision` **1.00** with no lexical anchor — the dense side carries it. `faithfulness` 0.71 is LLM-judge strictness on one answer (manually verified grounded), not a hallucination; small-N noise. |
| `aggregation` | 2 | `expected_fail`. Q "how many cancelled" found **2 of 3** (top-5 dropped the third). `context_recall` 0.33. |
| `temporal` | 3 | `expected_fail`. The pipeline **refuses** "which are active now" / "expiring in 6 months" rather than fabricating dates — graceful, but scores ~0 on RAGAS. "Most recent Vertex contract" it got right by comparing `start_date` in the framed metadata. |

Caveats:

- **Nondeterministic.** LLM-judge scores wobble ±0.03–0.05 run-to-run on identical inputs — one of
  the reasons the eval is not a merge gate. Compare a run against the baseline per `case_class`, not
  on the headline alone, and treat a single-slice move within that band as noise.
- **Still smallish N.** 15 headline questions, 2–4 per slice. Directional; it will not cleanly
  separate close configurations (HyDE vs step-back) — an ablation needs the slices deepened further.
  See `docs/PIPELINE_TUNING.md` for how every knob (`chunk_size`, `QUERY_REWRITE_STRATEGY`,
  `top_k`, `top_n`, model) would be tuned on a bigger dataset.
- **`expected_fail` is scored, not skipped.** `aggregation` + `temporal` land in
  `expected_fail_metrics` (baseline: f 0.57 / cr 0.23 / cp 0.23, n=5) so the known limitations stay
  measured without dragging the headline.
- **Costs OpenAI quota.** Generation (`gpt-4o`) + judge (`gpt-4o-mini`) calls per run;
  `results.json` records which generation model produced it.

## Testing Strategy

Three layers. Each layer has a different scope and cost.

### Layer 1 — Unit tests (`tests/test_*.py`, excluding `test_integration.py`)

All external services are mocked:

| Service | How it's mocked |
|---|---|
| ERPNext | `respx` intercepts at the httpx transport layer; `AsyncMock` for injected clients |
| Qdrant | `QdrantClient` class replaced with `MagicMock` via `monkeypatch` |
| OpenAI embeddings | `_client.embeddings.create` replaced with `MagicMock` |
| OpenAI GPT-4o | `_client.chat.completions.create` replaced with `MagicMock` |
| Langfuse | `MagicMock()` injected as `langfuse=` kwarg into `QueryPipeline` |
| CrossEncoder model | `CrossEncoder` class replaced with `MagicMock` via `monkeypatch` |

Run with `pytest tests/ -v`. No network. Required to pass in CI on every push.

### Layer 2 — Backend integration tests (`tests/test_integration.py`)

Nine groups that hit real services. Gated by `RUN_INTEGRATION=1`. Covers the full backend stack: ERPNext REST → ingestion → Qdrant → retrieval → pipeline → GPT-4o → Langfuse tracing. Streamlit is checked only with `httpx.get` → HTTP 200. Not in CI.

ERPNext is driven via **REST API** (`frappe.client.submit`, `frappe.client.save`) — this does not exercise the Frappe background worker queue that fires webhooks in the real Desk path.

### Layer 3 — Full-loop E2E tests (`tests/e2e/`, Playwright) — implemented, run live

Playwright drives the **ERPNext Desk UI and the real Streamlit UI in a real Chromium browser**, exercising the same paths a user takes. This layer exists to catch a class of bugs that REST-API integration tests cannot — and a live `RUN_E2E=1` run against this project's dev ERPNext site (2026-07-23) found two real, previously-unknown gaps purely by actually running it:

**Bugs found only via a real browser:**
1. Webhook record not configured in ERPNext — `terms-on-update` was missing. **Fixed**: created the record (matching the other three exactly) and verified functionally — a disposable Terms and Conditions document's update landed in Qdrant within 3 seconds.
2. Webhooks configured but not firing (URL wrong, disabled, worker queue not running).
3. `on_update` not firing on a Desk save to an already-submitted Contract — confirmed live via Frappe's own Webhook Request Log (zero delivery attempts, not a failed one). Contract's `is_signed` field is `allow_on_submit`, so toggling it via the Desk UI is a legal edit that should, but doesn't, re-index. **Fixed 2026-08-28 (#96)**: the root cause was that such an edit fires `on_update_after_submit`, not `on_update` — the registered webhook was the wrong event, and Frappe's webhook dispatcher separately hardcoded an allow-list that omitted `on_update_after_submit` (fixed upstream, present in this bench's installed frappe). See `docs/DEPLOYMENT.md` § Future Enhancements → "Post-submit edits are now re-indexed".

**Design principle — full-loop over isolated checks.** Earlier revisions had separate files that stopped at a raw Qdrant `points/scroll` call (does the point exist?) instead of the user-observable outcome (can a real person get this answer through the UI?). Those were consolidated into `test_erpnext_to_streamlit_loop.py`: each test drives the real ERPNext **Desk UI** for the action under test (REST bypasses the webhook-firing worker queue — exactly the bug class this suite exists to catch), then verifies the result by asking about it through the real Streamlit browser UI instead of querying Qdrant directly. `test_webhook_config.py` (static config, no ERPNext data) and `test_erpnext_desk_update_after_submit.py` (needs Frappe's exact Webhook Request Log and Qdrant payload values to conclusively prove the confirmed gap above) stay separate on purpose — routing either through an LLM-composed UI answer would trade away the precision that makes them work.

One thing full-loop tests must get right: BM25 only indexes chunk **text**, not `docname` (see `retrieval/hybrid_search.py`), so a bare docname-anchored question has no lexical anchor into a corpus of dozens of other, more topically-rich contracts and reliably fails retrieval — found live while building the cancel-status test. Planting the distinctive fact directly in the chunk text (not just referencing the docname in the question) fixes this.

Test files, what each catches, and the live result (gated by `RUN_E2E=1`):

| File | Browser? | Catches | Live result |
|---|---|---|---|
| `test_webhook_config.py` | No (REST) | Missing / misconfigured webhook records | 5 passed (fixed: created the missing `terms-on-update` record) |
| `test_erpnext_desk_update_after_submit.py` | Yes | Post-submit Desk save (re-)indexing, idempotently across repeated saves | 2 passed (fixed 2026-08-28, #96 — was 1 xfailed, 1 passed) |
| `test_erpnext_to_streamlit_loop.py` | Yes | Desk action (submit / submit+PDF / cancel) not reaching the real Streamlit UI | 3 passed |
| `test_streamlit_sidebar_filters.py` | Yes | Sidebar filter widgets breaking the query flow | 1 passed |

Three stale pre-rebrand webhooks (`po-*`, `scorecard-on-update`) are also still present on this site; harmless (ignored by `SUPPORTED_DOCTYPES`) but not yet cleaned up — a separate, optional follow-up. The post-submit-edit gap was originally recorded as `xfail(strict=True)`, documenting it as a known Frappe 15 platform limitation rather than leaving it as an unexplained red test (`strict=True` meant a future fix would flip it to a loud XPASS failure instead of silently staying green) — that's exactly what happened: fixed 2026-08-28 (#96), `xfail` markers removed, see `docs/DEPLOYMENT.md` § Future Enhancements → "Post-submit edits are now re-indexed".

`test_erpnext_to_streamlit_loop.py` drives a real Desk UI action, then a real running Streamlit server through a second real Chromium browser tab against the real FastAPI backend — nothing mocked, the complete browser → Desk UI → webhook → Qdrant → browser → Streamlit → FastAPI → pipeline → OpenAI → browser round trip. Login to Streamlit is bypassed by minting a JWT directly (`api.auth.jwt_handler.mint_token`, what `/auth/callback` calls after a real OAuth exchange) via `mint_test_jwt`, rather than driving ERPNext's OAuth consent screen through the browser a second time. `test_streamlit_sidebar_filters.py` covers what the full-loop tests don't — filter widget interaction against pre-existing demo data — and is a different, complementary layer from the always-on `tests/test_streamlit.py` below: that one verifies UI *logic* fast and in-process with `httpx.post` mocked; these verify the UI actually *renders and works* end-to-end.

Streamlit UI also has a separate, always-on layer tested via `streamlit.testing.v1.AppTest` (in-process, no browser, no running server, **not** gated by `RUN_E2E` — runs in the normal `pytest tests/` unit suite) — `tests/test_streamlit.py`, implemented and passing. See `docs/IMPLEMENTATION_PLAN.md` Step 17 for the full test breakdown.
