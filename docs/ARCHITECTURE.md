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
3. **Hybrid search** — BM25 (lexical) and Qdrant vector search run in parallel. Results are fused with Reciprocal Rank Fusion (`k=60`), returning 20 candidates. **Important:** metadata filters (supplier, doctype, status) are applied only to the Qdrant vector-search path — BM25 has no metadata support and searches the full corpus. Caller filters therefore narrow vector results but do not guarantee that every result in the fused set matches the filter.
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

## Model Configuration

Generation and embedding model names are centralized in `config.py`, set via the `OPENAI_MODEL`
(default `gpt-4o`) and `EMBEDDING_MODEL` (default `text-embedding-3-small`) env vars — mirroring
the `QUERY_REWRITE_STRATEGY` pattern. Every call site (`query_rewriter.py`, `query_pipeline.py`,
`embedder.py`, `evaluate.py`) reads from `config.py` instead of hardcoding a model literal.

**Caveat — embedding model swaps:** `retrieval/vector_store.py`'s `VECTOR_DIM` is derived from
`EMBEDDING_MODEL` via `config.embedding_dimension()`, which maps model name → vector size. The
Qdrant collection's vector size is fixed at creation time (`ensure_collection`), so switching to an
embedding model with a different dimension requires recreating the collection and running a full
re-ingest — existing points are not re-embedded automatically. Unrecognized model names raise at
import time until their dimension is added to `config._EMBEDDING_DIMENSIONS`.

### Future Enhancement — provider-agnostic adapter (not implemented)

`OPENAI_MODEL`/`EMBEDDING_MODEL` (above) solve swapping the OpenAI *model version*, not the
*provider* — every call site (`query_rewriter.py`, `query_pipeline.py`, `embedder.py`,
`evaluate.py`) still constructs a raw `openai.OpenAI()` client and reads its response shape
directly (`response.choices[0].message.content` / `response.data[i].embedding`).

A genuine provider swap would need a small hand-rolled seam: a `ChatModel`/`EmbeddingModel`
interface (two methods each) that every call site depends on instead of a concrete SDK, an
`OpenAIChatModel`/`OpenAIEmbeddingModel` adapter wrapping today's existing `openai`-SDK code behind
that interface (a mechanical move, not a rewrite), and `LLM_PROVIDER`/`EMBEDDING_PROVIDER` env vars
(default `openai`) dispatching to the right adapter class.

`langchain`, `langchain-openai`, and `langchain-community` are already dependencies (used today only
for `RecursiveCharacterTextSplitter` in `ingestion/chunker.py`) and were considered for this instead
of a hand-rolled adapter — LangChain's `init_chat_model` factory does exist in the pinned version
and would normalize the response shape across providers in one move. Rejected for now: it's real
added dependency/version-risk surface for a swap that isn't happening, versus a hand-rolled adapter
that stays fully within the codebase's own control and is no larger for the one provider (OpenAI)
actually in use today.

Either approach touches the OpenAI-response-shape mocking in `tests/test_query_rewriter.py`,
`tests/test_query_pipeline.py`, and `tests/test_embedder.py` (currently hand-rolled `MagicMock`
chains matching `.choices[0].message.content` / `.data[i].embedding`). Deferred — see #51; not
prioritized since no provider swap is currently planned.

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
| Contract            | `on_cancel` | Re-indexed with status=Cancelled                        |
| Terms and Conditions| `on_update` | Re-indexed (not a submittable doctype)                  |

> **Known gap:** `on_update_after_submit` for Purchase Orders is not supported. Frappe 15 does not fire this webhook event via API or desk UI saves, so edits to a submitted PO (e.g. delivery date, remarks) are not automatically re-indexed. Workaround: `POST /ingest/full`. See `docs/DEPLOYMENT.md` § Future Enhancements.

The webhook handler:
1. Verifies `X-Frappe-Webhook-Signature` (HMAC-SHA256, base64-encoded)
2. Fetches the full document via REST API
3. Calls `vector_store.delete_by_docname(docname)` — removes all old chunks
4. Re-runs the full parse → chunk → embed → upsert pipeline for that document only

This ensures updated contract terms are reflected in the next query without a full re-index.

## BM25 Index

The BM25 index is built in memory at API startup by fetching all payload texts from Qdrant. It is rebuilt after each webhook upsert to stay current. For large collections (>100k docs), consider moving to Qdrant's sparse vector support instead.

### Filter behaviour — verified in live testing (2026-06-19)

Metadata filters (supplier, doctype, status) apply **only to the Qdrant vector-search leg** of the hybrid search. BM25 is corpus-wide by design; it cannot apply field conditions. Consequences:

- A supplier filter narrows the Qdrant candidate set to that supplier but BM25 may still surface documents from other suppliers that score highly on lexical overlap.
- The pipeline's `_parse_sources` only includes documents that GPT-4o explicitly cites, so uncited BM25 hits are silently discarded — the practical leakage is lower than the raw candidate count suggests.
- If strict supplier isolation is required (e.g. a multi-tenant deployment), a post-rerank filter on `source.supplier` must be added in `query_pipeline.py` before the generation step.

## Observability

Every query creates a Langfuse trace with the following child observations:

| Observation | Type | Input captured | Output captured |
|---|---|---|---|
| `rewrite` | span | original question | rewritten text + query vector |
| `filter_extraction` | span | original question | `{source_doctype?, status?}` dict |
| `hybrid_search` | span | rewritten query + filters | list of 20 scored chunks |
| `rerank` | span | original question + 20 candidates | top-5 re-scored chunks |
| `generate` | generation | original question + context string | answer with `[docname]` citations |

The trace root carries `input.question` and `output.{answer, source_count}`. If any step raises, the failing span and the root trace are both updated with `level="ERROR"`.

`generate` is a Langfuse **generation** (not a plain span) — it's created with `trace.generation(model=OPENAI_MODEL)` and passes `usage={"input", "output", "total"}` (from the OpenAI response's `.usage`) to `end()`. Only `generation`-type observations get token counts and cost auto-computed by Langfuse; a plain span just stores whatever input/output you hand it (see PR [#87](https://github.com/arunjoyt/contract-intelligence/pull/87), issue #81).

**Not traced: embedding calls.** `ingestion/embedder.py`'s `Embedder` (used both for HyDE query embedding at query time and for chunk embedding during ingestion) is intentionally left trace-agnostic — it has no Langfuse dependency and isn't passed a trace/span object by any caller. Wiring it in would mean threading trace context through the ingestion path (full ingest, webhook re-index) as well as the query path, for comparatively little signal: embedding cost (`text-embedding-3-small`) is negligible next to GPT-4o generation cost. Revisit if embedding volume or cost grows enough to matter.

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

### Future Enhancements — production quality monitoring (not implemented)

Tracing tells you a query ran and how long it took; it says nothing about whether the answer was any good. RAGAS (`evaluation/evaluate.py`) only scores a fixed golden dataset in CI, so quality drift on real production questions currently goes undetected. Two independent options, not mutually exclusive:

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

Nine groups that hit real services. Gated by `RUN_E2E=1` → `RUN_INTEGRATION=1`. Covers the full backend stack: ERPNext REST → ingestion → Qdrant → retrieval → pipeline → GPT-4o → Langfuse tracing. Streamlit is checked only with `httpx.get` → HTTP 200. Not in CI.

ERPNext is driven via **REST API** (`frappe.client.submit`, `frappe.client.save`) — this does not exercise the Frappe background worker queue that fires webhooks in the real Desk path.

### Layer 3 — E2E Desk UI tests (`tests/e2e/`, Playwright) — planned, not yet implemented

Playwright drives the **ERPNext Desk UI in a real Chromium browser**, exercising the same code path a user takes. This layer exists to catch a class of bugs that REST-API integration tests cannot:

**Three bugs found only via Desk UI:**
1. Webhook records not configured in ERPNext.
2. Webhooks configured but not firing (URL wrong, disabled, worker queue not running).
3. `on_update_after_submit` event not triggering a webhook after a desk save on a submitted PO.

Planned test files and what each catches:

| File | Browser? | Catches |
|---|---|---|
| `test_webhook_config.py` | No (REST) | Missing / misconfigured webhook records |
| `test_erpnext_desk_submit.py` | Yes | Webhook not firing on Desk submit |
| `test_erpnext_desk_update_after_submit.py` | Yes | `on_update_after_submit` gap |
| `test_erpnext_desk_cancel.py` | Yes | Cancel not propagating to Qdrant removal |

Streamlit UI is tested separately via `streamlit.testing.v1.AppTest` (in-process, no browser). See `docs/IMPLEMENTATION_PLAN.md` Step 17 for the full implementation plan.
