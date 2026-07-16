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
│  Collection: procurement         │
│  Indexed payload: source_doctype,│
│  supplier, status, start_date,   │
│  end_date                        │
└──────────────┬───────────────────┘
               │
     ┌─────────▼──────────┐
     │   Retrieval Layer   │
     │  BM25 + vector →   │
     │  RRF fusion →      │
     │  Cross-encoder     │
     │  rerank            │
     └─────────┬──────────┘
               │ top-5 chunks + metadata
               ▼
┌──────────────────────────────────┐
│         Pipeline Layer           │
│  QueryRewriter (HyDE/step-back)  │
│  → HybridSearch                  │
│  → Reranker                      │
│  → Prompt builder                │
│  → GPT-4o                        │
│  → Answer + source citations     │
└──────────┬──────────┬────────────┘
           │          │ Langfuse spans
           ▼          ▼
        FastAPI    Langfuse
        /query      (trace)
           │
           ▼
        Streamlit
```

## Query Pipeline — Step by Step

1. **Query rewriting** — HyDE generates a hypothetical answer; its embedding becomes the query vector. This improves recall for abstract questions.
2. **Metadata filter extraction** — simple heuristics (and optionally a small LLM call) parse supplier names, date ranges, and doctype hints from the original question.
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

## Document Indexing Strategy

### Structured documents (PO, Invoice, Scorecard)

Serialized to a single natural-language string before embedding. No chunking — the entire doc is one vector. This avoids fragmenting relational fields across chunks.

```
Purchase Order PO-2024-00123 issued to Vendor ABC on 2024-03-15.
Total value: 45,000 USD. Delivery expected by 2024-04-10.
Items: Steel Pipes, Welding Rods. Payment terms: Net 30. Status: Submitted.
```

### Unstructured documents (Contract, T&C, PDFs)

- HTML stripped via BeautifulSoup before chunking
- `RecursiveCharacterTextSplitter`: `chunk_size=512`, `chunk_overlap=64`
- Each chunk stores `chunk_index` and `total_chunks` for reconstruction

### PDF attachments

`Purchase Order` and `Contract` documents may carry attached PDFs (e.g. signed contract scans,
invoice PDFs). For these two doctypes, `erpnext_client.get_attached_files()` lists the document's
`File` records, `.pdf`-named ones are downloaded via `get_file_content()`, and their text is
extracted via `extract_text_from_pdf()`. The extracted text is chunked the same way as any other
unstructured text and indexed as **additional points under the same `docname`** as the parent
document — `chunk_index`/`total_chunks` are renumbered across the combined set (parent text +
attachments) so `delete_by_docname` and reconstruction stay correct. A failure to list attachments
or extract one PDF's text is logged and skipped; it never blocks indexing of the parent document or
other attachments.

### Supplier Metadata Enrichment

`supplier_group` (used in the payload schema and as a filter field) is **not** present on `Purchase
Order` records — it only exists on `Supplier` (and is, separately, already a direct field on
`Purchase Invoice`). For `Purchase Order` and `Contract`, `erpnext_client.py` must fetch/cache
`Supplier` records and join `supplier_group` onto the record before serialization in
`document_parser.py`.

### Qdrant Payload Schema

```json
{
  "text": "...",
  "metadata": {
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
}
```

Point ID is derived as `uuid5(NAMESPACE_DNS, f"{docname}:{chunk_index}")` — deterministic, enabling idempotent upserts.

`Contract` carries two additional, optional payload fields — `linked_doctype`/`linked_docname` — populated from
the doctype's `document_type`/`document_name` Dynamic Link pair when the contract references a specific
transaction document (e.g. a Purchase Order). A matching sentence ("This contract is linked to Purchase Order
PO-2024-00123.") is also prepended to the embedded/chunked text so the link is retrievable via semantic/BM25
search without requiring a metadata filter.

## Incremental Indexing via Webhooks

ERPNext fires webhooks on the following events:

| Doctype             | Event       | Effect                                                  |
|---------------------|-------------|----------------------------------------------------------|
| Purchase Order      | `on_submit` | Indexed fresh on submission (incl. attached PDFs)       |
| Purchase Order      | `on_cancel` | Re-indexed with status=Cancelled                        |
| Purchase Invoice    | `on_submit` | Indexed fresh on submission                             |
| Purchase Invoice    | `on_cancel` | Re-indexed with status=Cancelled                        |
| Contract            | `on_submit` | Indexed when contract is submitted (incl. attached PDFs)|
| Contract            | `on_update` | Re-indexed on any desk save (incl. attached PDFs)       |
| Terms and Conditions| `on_update` | Re-indexed (not a submittable doctype)                  |
| Supplier Scorecard  | `on_update` | Re-indexed (scorecards are not submittable)             |

> **Known gap:** `on_update_after_submit` for Purchase Orders is not supported. Frappe 15 does not fire this webhook event via API or desk UI saves, so edits to a submitted PO (e.g. delivery date, remarks) are not automatically re-indexed. Workaround: `POST /ingest/full`. See `docs/DEPLOYMENT.md` § Future Enhancements.

The webhook handler:
1. Verifies `X-Frappe-Webhook-Signature` (HMAC-SHA256, base64-encoded)
2. Fetches the full document via REST API
3. Calls `vector_store.delete_by_docname(docname)` — removes all old chunks
4. Re-runs the full parse → chunk → embed → upsert pipeline for that document only

This ensures updated contract terms or re-submitted POs are reflected in the next query without a full re-index.

## BM25 Index

The BM25 index is built in memory at API startup by fetching all payload texts from Qdrant. It is rebuilt after each webhook upsert to stay current. For large collections (>100k docs), consider moving to Qdrant's sparse vector support instead.

### Filter behaviour — verified in live testing (2026-06-19)

Metadata filters (supplier, doctype, status) apply **only to the Qdrant vector-search leg** of the hybrid search. BM25 is corpus-wide by design; it cannot apply field conditions. Consequences:

- A supplier filter narrows the Qdrant candidate set to that supplier but BM25 may still surface documents from other suppliers that score highly on lexical overlap.
- The pipeline's `_parse_sources` only includes documents that GPT-4o explicitly cites, so uncited BM25 hits are silently discarded — the practical leakage is lower than the raw candidate count suggests.
- If strict supplier isolation is required (e.g. a multi-tenant deployment), a post-rerank filter on `source.supplier` must be added in `query_pipeline.py` before the generation step.

## Observability

Every query creates a Langfuse trace with the following spans:

| Span | Input captured | Output captured |
|---|---|---|
| `rewrite` | original question | rewritten text + query vector |
| `filter_extraction` | original question | `{source_doctype?, status?}` dict |
| `hybrid_search` | rewritten query + filters | list of 20 scored chunks |
| `rerank` | original question + 20 candidates | top-5 re-scored chunks |
| `generate` | original question + context string | answer with `[docname]` citations |

The trace root carries `input.question` and `output.{answer, source_count}`. If any step raises, the failing span and the root trace are both updated with `level="ERROR"`.

The Langfuse UI is accessible at `http://localhost:3000` (default docker-compose port).

### How to verify in the Langfuse UI

1. Start all services including Langfuse: `docker compose up qdrant langfuse postgres -d`
2. Open `http://localhost:3000` → log in (credentials from `.env`: `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY`)
3. Run the API with Langfuse wired in: `uvicorn api.main:app --reload`
4. Send a query: `curl -s -X POST http://localhost:8000/query -H 'Content-Type: application/json' -d '{"question":"What are the payment terms for our active contracts?"}'`
5. In the Langfuse UI → **Traces** → select the new trace and verify:
   - Root trace `name` is `query`; `input.question` matches the sent question
   - Five child spans appear: `rewrite`, `filter_extraction`, `hybrid_search`, `rerank`, `generate`
   - Each span has a non-zero duration and no error level
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
