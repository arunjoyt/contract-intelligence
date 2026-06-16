# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
Machine-specific setup (bench path, local site URL/credentials) lives in `CLAUDE.local.md`, which is
gitignored — see that file if you need to run bench commands or hit the local ERPNext site.

## Project State

This repository currently contains **only planning documentation** (`README.md`, `docs/ARCHITECTURE.md`,
`docs/DEPLOYMENT.md`, `docs/IMPLEMENTATION_PLAN.md`). No source code, `requirements.txt`, `pyproject.toml`,
or tests exist yet. When asked to implement this project, follow `docs/IMPLEMENTATION_PLAN.md` — it is an
**ordered** list of steps (Step 1 → Step 16) that explicitly states "Each step depends on the ones before
it. Do not reorder." Treat that file as the source of truth for build order, file layout, and interfaces;
treat `docs/ARCHITECTURE.md` as the source of truth for data flow and schema details; treat
`docs/DEPLOYMENT.md` as the source of truth for auth/deployment when those steps come up.

Once code exists, the commands below (drawn from the docs) are the ones to use:

```bash
# Run the API
uvicorn api.main:app --reload

# Run the frontend
streamlit run frontend/app.py

# Start infra (Qdrant + Langfuse + Postgres)
docker compose up qdrant langfuse postgres -d

# Trigger a full ingest
curl -X POST http://localhost:8000/ingest/full -H "X-Admin-Secret: <ADMIN_SECRET>"

# Lint
ruff check .

# Tests (must run with no network — OpenAI/Qdrant are mocked)
pytest tests/

# Single test
pytest tests/test_chunker.py::test_name -v

# RAGAS evaluation
python evaluation/evaluate.py
```

## Architecture

A RAG system that answers natural-language procurement questions grounded in ERPNext (Frappe) data.

```
ERPNext --REST API (full ingest) / Webhooks (incremental)--> Ingestion --> Qdrant --> Retrieval --> Pipeline --> FastAPI --> Streamlit
                                                                                                         |
                                                                                                      Langfuse (tracing)
```

**Layers** (see `docs/ARCHITECTURE.md` for full diagrams):

- **Ingestion** (`ingestion/`): `erpnext_client.py` (async httpx wrapper around Frappe REST API) →
  `document_parser.py` (HTML stripping via BeautifulSoup, PDF extraction via pypdf, struct→NL serialization)
  → `chunker.py` (`RecursiveCharacterTextSplitter`, `chunk_size=512`/`chunk_overlap=64`) → `embedder.py`
  (`text-embedding-3-small`, batched). `webhook_handler.py` is the incremental re-indexing entry point.
- **Retrieval** (`retrieval/`): `vector_store.py` (Qdrant wrapper) + `hybrid_search.py` (BM25 in-memory
  index fused with Qdrant vector search via Reciprocal Rank Fusion, `k=60`) + `reranker.py`
  (`cross-encoder/ms-marco-MiniLM-L-6-v2`, lazy singleton, loaded once at startup).
- **Pipeline** (`pipeline/`): `query_rewriter.py` (HyDE or step-back rewriting, controlled by
  `QUERY_REWRITE_STRATEGY`) → `query_pipeline.py` (orchestrates rewrite → metadata filter extraction →
  hybrid search top-20 → rerank top-5 → GPT-4o generation with required source citations). Every step is a
  Langfuse child span.
- **API** (`api/main.py`): `POST /query`, `POST /webhook/erpnext`, `POST /ingest/full` (background task,
  `X-Admin-Secret`-gated), `GET /health`. Startup hook ensures the Qdrant collection exists, rebuilds the
  BM25 index from Qdrant, and warms the reranker.
- **Frontend** (`frontend/app.py`): Streamlit chat UI with supplier/doctype/date/status filters in the
  sidebar; calls the API over `httpx`.

### Indexing strategy — two distinct paths

- **Structured docs** (Purchase Order, Purchase Invoice, Supplier Scorecard): serialized to a single
  natural-language string and embedded as **one vector, no chunking** — relational fields must not be
  fragmented across chunks.
- **Unstructured docs** (Contract, Terms and Conditions, attached PDFs): HTML-stripped, then split via
  `RecursiveCharacterTextSplitter`; each chunk carries `chunk_index`/`total_chunks` for reconstruction.

### Qdrant payload & idempotency

Point ID is deterministic: `uuid5(NAMESPACE_DNS, f"{docname}:{chunk_index}")`. This makes upserts idempotent
— re-ingesting a document overwrites its existing points instead of duplicating them. Payload includes
`source_doctype`, `docname`, `supplier`, `supplier_group`, `start_date`, `end_date`, `status`, `company`,
`chunk_index`, `total_chunks`.

### Incremental indexing (webhooks)

ERPNext fires webhooks on `on_submit`/`on_update` for `Purchase Order`, `Contract`, `Supplier Scorecard`.
The handler: verify `X-Frappe-Webhook-Signature` (HMAC-SHA256) → fetch full doc via `ERPNextClient` →
`delete_by_docname` → re-run parse → chunk → embed → upsert → rebuild BM25 index. No full re-index is
needed for routine updates.

### Query pipeline order

1. `QueryRewriter.rewrite()` — HyDE (default) embeds a hypothetical answer document instead of the raw
   query, improving recall for abstract questions; step-back rewrites the question at a higher abstraction
   level instead.
2. Metadata filter extraction (heuristic, possibly LLM-assisted) from the original question.
3. `HybridSearch.search()` — BM25 + Qdrant vector search in parallel, fused via RRF, top-20.
4. `Reranker.rerank()` — cross-encoder scores all 20 `(query, chunk)` pairs, returns top-5.
5. GPT-4o generation with a system prompt that requires `[docname]`-style citations per claim and forbids
   answering outside the provided context.

### Security invariants

- ERPNext API key/secret stay server-side; the webhook endpoint is the only externally reachable path and
  it's HMAC-protected.
- `/ingest/full` requires `X-Admin-Secret`.
- Qdrant filter expressions are built programmatically — never interpolate raw user input into them.

## Deployment

Two access-control options are documented in `docs/DEPLOYMENT.md` — pick based on ERPNext hosting:

- **Option A (Frappe custom app)**: native ERPNext desk integration; a `@frappe.whitelist()` method checks
  `frappe.get_roles()` and calls the FastAPI backend (bound to `127.0.0.1:8000`, loopback-only) using a
  short-lived HMAC token. No separate login. Requires bench access (not available on ERPNext SaaS).
- **Option B (standalone + OAuth2/JWT)**: separate app, "Login with ERPNext" via OAuth2 Authorization Code
  + PKCE; FastAPI exchanges the code, fetches roles, mints an 8-hour JWT. Works with any ERPNext hosting,
  no custom app install. Role changes apply at next login only.

Allowed roles in both options: `Purchase Manager`, `Purchase User`, `Accounts User`, `System Manager`.

`docs/DEPLOYMENT.md`'s "Implementation Sequence" table maps these auth additions onto
`IMPLEMENTATION_PLAN.md` Steps 12/13/15 — consult it before modifying those steps.

## CI

GitHub Actions (`.github/workflows/ci.yml`, not yet created): `ruff check .` and `pytest tests/` on every
push (tests run with no network — OpenAI and Qdrant are mocked); RAGAS evaluation runs only on merge to
`main`, with `evaluation/results.json` uploaded as an artifact.
