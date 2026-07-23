# Implementation Plan

Ordered list of modules to build. Each step depends on the ones before it. Do not reorder.

**Status:** all steps below (0–16) are implemented — see roadmap issue #17. This file is kept as the
historical build reference and is periodically refreshed to track implementation drift; for current
system design and data flow, `docs/ARCHITECTURE.md` is authoritative.

> **Note:** Steps below describe the system as originally built, covering Purchase Order, Purchase
> Invoice, and Supplier Scorecard ingestion alongside Contract and Terms and Conditions. The project
> was later descoped to Contract Intelligence only (dropping those three doctypes) — see #66 and
> `docs/CONTRACT_INTELLIGENCE_DESCOPE_PLAN.md` for the decision, and `docs/ARCHITECTURE.md` for the
> current, descoped design.

---

## Step 0 — ERPNext Environment Verification

Before writing any ingestion code, verify the target ERPNext instance is actually usable end-to-end.
This is a one-time local/staging setup step, not application code — see `CLAUDE.local.md` for this
machine's specific bench path, site URL, and credentials.

**Checklist:**
1. Confirm the site is reachable (`curl $ERPNEXT_URL` → `200`).
2. Generate an API key/secret for the integration user (Frappe Desk → User → API Access → Generate
   Keys, or `frappe.core.doctype.user.user.generate_keys`). Put them in `.env` as `ERPNEXT_API_KEY` /
   `ERPNEXT_API_SECRET` — never commit them.
3. Confirm data exists for every doctype the ingestion layer touches: `Purchase Order`,
   `Purchase Invoice`, `Supplier`, `Supplier Scorecard`, `Contract`, `Terms and Conditions`. Default
   ERPNext demo data typically populates the first three but **not** `Supplier Scorecard` / `Contract` /
   `Terms and Conditions` — seed a handful of realistic records for these manually so the
   unstructured-doc chunking path (Step 4) and scorecard serialization (Step 3) can be exercised
   against real data, not only mocks.
4. Note for Step 2 (`erpnext_client.py`): `supplier_group` is **not** a field on `Purchase Order`
   (confirmed: `frappe.client.get_list` rejects it with `DataError`) — it only exists on `Supplier` and
   on `Purchase Invoice` (which carries its own `supplier_group` field directly). For `Purchase Order`
   and `Contract`, the client (or `document_parser.py`) must look `supplier_group` up via a `Supplier`
   fetch/cache rather than assume it's present on the record.

**Verification:** a request to `Purchase Order` with the generated token returns `200` and a
non-empty `data` list.

---

## Step 1 — Project Scaffold

**Files:**
- `requirements.txt`
- `pyproject.toml` (ruff + mypy config)
- `.env.example`
- `.gitignore`
- Empty `__init__.py` files in each package directory

**Key Python deps to pin:**
```
fastapi
uvicorn[standard]
langchain
langchain-openai
langchain-community
qdrant-client
openai
sentence-transformers
rank-bm25
beautifulsoup4
lxml
pypdf
ragas
langfuse
streamlit
python-dotenv
httpx
pytest
ruff
```

---

## Step 2 — `ingestion/erpnext_client.py`

Async Frappe REST API wrapper using `httpx.AsyncClient`.

**Interface:**
```python
class ERPNextClient:
    async def get_list(doctype, filters, fields, limit) -> list[dict]
    async def get_doc(doctype, name) -> dict
    async def get_file_content(file_url) -> bytes
```

**Auth:** `Authorization: token {ERPNEXT_API_KEY}:{ERPNEXT_API_SECRET}` header on every request.

**Error handling:** raise typed `ERPNextNotFoundError`, `ERPNextAuthError`.

---

## Step 3 — `ingestion/document_parser.py`

**Functions:**
```python
def extract_text_from_html(html: str) -> str          # BeautifulSoup, strip tags
def extract_text_from_pdf(pdf_bytes: bytes) -> str    # pypdf, join pages
def po_to_text(po: dict) -> str                       # structured → natural language
def supplier_scorecard_to_text(sc: dict) -> str
def invoice_to_text(inv: dict) -> str
```

Attached PDFs on `Purchase Order`/`Contract` (e.g. signed contract scans) are listed via
`erpnext_client.get_attached_files()`, downloaded, and run through `extract_text_from_pdf()`; the
extracted text is chunked and indexed as additional points under the parent `docname` (see
`ingestion/webhook_handler.py`'s `ATTACHMENT_DOCTYPES` and `docs/ARCHITECTURE.md` § PDF attachments).

`po_to_text` template (from brief):
```
Purchase Order {name} issued to {supplier} on {transaction_date}.
Total value: {grand_total} {currency}.
Delivery expected by {schedule_date}.
Items: {item_names}.
Payment terms: {payment_terms_template}.
Status: {status}.
```

---

## Step 4 — `ingestion/chunker.py`

```python
def chunk_text(text: str, chunk_size=512, chunk_overlap=64) -> list[dict]
# Returns: [{"text": str, "chunk_index": int, "total_chunks": int}]
```

Uses `langchain.text_splitter.RecursiveCharacterTextSplitter`.  
Structured docs (already short serialized strings) are returned as a single chunk without splitting.

---

## Step 5 — `ingestion/embedder.py`

```python
class Embedder:
    def embed_texts(texts: list[str]) -> list[list[float]]   # batched, max 2048/call
    def embed_query(text: str) -> list[float]
```

Model: `EMBEDDING_MODEL` env var (see `config.py`), default `text-embedding-3-small` (1536 dims).

---

## Step 6 — `retrieval/vector_store.py`

```python
class VectorStore:
    def ensure_collection()                              # create if not exists
    def upsert_chunks(chunks: list[dict])               # idempotent by docname+chunk_index
    def search(query_vector, filter_conditions, top_k=20) -> list[ScoredPoint]
    def delete_by_docname(docname: str)
    def get_all_texts() -> list[dict]                    # for BM25 index rebuild
```

Point ID: `str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{docname}:{chunk_index}"))`.

---

## Step 7 — `ingestion/webhook_handler.py`

FastAPI `APIRouter`, mounted at `/webhook`.

```
POST /webhook/erpnext
  1. Verify HMAC-SHA256 signature
  2. Extract doctype + docname
  3. Fetch full doc via ERPNextClient
  4. delete_by_docname → parse → chunk → embed → upsert
  5. Rebuild BM25 index
  6. Return {"status": "indexed", "docname": ...}
```

Supported doctypes: `Purchase Order`, `Purchase Invoice`, `Contract`, `Terms and Conditions`,
`Supplier Scorecard`.

---

## Step 8 — `retrieval/hybrid_search.py`

```python
class HybridSearch:
    def build_bm25_index(docs: list[dict])       # called on startup and after upsert
    def search(query: str, filter_conditions: dict, top_k: int = 20) -> list[dict]
```

Internals:
1. BM25 scores over in-memory corpus
2. Qdrant vector search (with metadata filter)
3. RRF merge: `score(doc) = Σ 1 / (60 + rank)`
4. Return merged list sorted by RRF score

---

## Step 9 — `retrieval/reranker.py`

```python
class Reranker:
    def rerank(query: str, candidates: list[dict], top_n: int = 5) -> list[dict]
```

Model: `cross-encoder/ms-marco-MiniLM-L-6-v2` via `sentence_transformers.CrossEncoder`.  
Loaded once at startup (lazy singleton).

---

## Step 10 — `pipeline/query_rewriter.py`

```python
class QueryRewriter:
    def rewrite(query: str) -> tuple[str, list[float]]
    # Returns (rewritten_text, embedding_vector)
```

**HyDE strategy (default):**  
Prompt the model (`OPENAI_MODEL` env var, see `config.py`, default `gpt-4o`): *"Write a hypothetical
procurement document that would answer: {query}"*  
Embed the hypothetical answer; use that vector for retrieval.

**Step-back strategy:**  
Prompt the model to rewrite the question at a higher abstraction level, then embed the rewritten question.

Controlled by `QUERY_REWRITE_STRATEGY` env var.

---

## Step 11 — `pipeline/query_pipeline.py`

```python
class QueryPipeline:
    def run(question: str, filters: dict | None = None) -> dict
    # Returns {"answer": str, "sources": list[SourceDoc]}
```

Pipeline steps (all wrapped in Langfuse spans):
1. `QueryRewriter.rewrite(question)` → rewritten text + query vector
2. Extract metadata filters from original question
3. `HybridSearch.search(rewritten_text, filters, top_k=20)`
4. `Reranker.rerank(question, candidates, top_n=5)`
5. Build context string with chunk text + metadata
6. Call the raw `openai` SDK (`client.chat.completions.create(model=OPENAI_MODEL, ...)` — not
   LangChain's `ChatOpenAI`; LangChain is only used for `RecursiveCharacterTextSplitter` in
   `chunker.py`) with a system prompt requiring source citations
7. Parse response into `answer` + `sources`

**System prompt key points:**
- Answer only from the provided context
- Cite each claim with `[docname]`
- If the answer is not in context, say so explicitly — do not hallucinate

---

## Step 12 — `api/main.py`

```python
app = FastAPI()

GET  /auth/login      # redirect to ERPNext OAuth2 authorize endpoint
GET  /auth/callback   # exchange code → roles → mint JWT
POST /query           # {"question": str, "filters": dict|null} → pipeline response
                      #   requires Authorization: Bearer <jwt> (require_allowed_role dependency)
POST /webhook/erpnext # delegated to webhook_handler router
POST /ingest/full     # manual full re-index (background task, X-Admin-Secret required)
GET  /health          # {"status": "ok"}
```

Startup event:
1. Call `VectorStore.ensure_collection()`
2. Fetch all texts from Qdrant → build BM25 index
3. Warm up reranker model

Auth modules (`api/auth/`): `oauth2.py` (authorize URL + token exchange + role fetch), `pkce.py` (code verifier/challenge), `jwt_handler.py` (mint + decode), `dependencies.py` (`get_current_user`, `require_allowed_role` FastAPI dependencies).

---

## Step 13 — `frontend/app.py`

Streamlit app layout:
- **Auth gate:** `frontend/auth_ui.py` — shows "Login with ERPNext" button when no JWT in `st.session_state`; handles OAuth2 callback redirect and stores the returned JWT
- **Sidebar:** supplier filter (text), doctype filter (multiselect), date range, status filter
- **Main:** `st.chat_input` for question entry (shown only after successful login)
- **Response:** answer text block + collapsible "Sources" expander listing `docname`, `source_doctype`, `supplier` per citation
- **Session state:** `messages` list (last 10 turns displayed as chat history), `jwt` token

Calls `POST {BACKEND_URL}/query` via `httpx` with `Authorization: Bearer <jwt>` header.

---

## Step 14 — `evaluation/test_dataset.json` + `evaluation/evaluate.py`

**`test_dataset.json`** — 15 Q&A pairs, one per query pattern, with stored ground-truth contexts (so evaluation runs without a live ERPNext).

**`evaluate.py`:**
```python
# Loads test_dataset.json
# If the Qdrant collection is empty (e.g. a fresh CI service container with no live
# ERPNext to ingest from), seeds it from the dataset's own ground-truth contexts first
# Runs pipeline.run(question) for each entry
# Computes RAGAS: faithfulness, answer_relevancy, context_recall, context_precision
# Writes evaluation/results.json
```

---

## Step 15 — `docker-compose.yml`

| Service | Image | Port | Volume |
|---|---|---|---|
| `postgres` | `postgres:16-alpine` | internal only (`rag_internal`) | `pg_data` |
| `langfuse` | `langfuse/langfuse:2` | `127.0.0.1:3000` (loopback, SSH-tunnel only) | — |
| `qdrant` | `qdrant/qdrant:latest` | `127.0.0.1:6333` (loopback, SSH-tunnel only) | `qdrant_data` |
| `app` | local `Dockerfile` | internal only (`rag_internal`) | — |
| `frontend` | local `Dockerfile`, `streamlit run` | internal only (`rag_internal`) | — |
| `nginx` | `nginx:alpine` | `80`/`443` (public — sole ingress) | — |

All services run `restart: unless-stopped`. `docker-compose.frontend.yml` is a local-dev override —
it doesn't add new services, it exposes `app`/`frontend`'s ports directly to the host (`8000`/`8501`)
for the "infra in Docker, app/frontend running natively" workflow.
`docker-compose.local-prod.yml` is a separate override for testing the production nginx topology
locally with mkcert certs.

---

## Step 16 — `.github/workflows/ci.yml`

**Job: `lint-and-test`** (every push, every PR):
```yaml
- uses: actions/setup-python@v5 (3.11)
- run: pip install -r requirements.txt
- run: ruff check .
- run: pytest tests/ -v
```

**Job: `evaluate`** (push to `main` only):
```yaml
- run: python evaluation/evaluate.py
- uses: actions/upload-artifact (evaluation/results.json)
```

---

## Step 17 — `tests/e2e/` — ERPNext Desk E2E Test Suite (Playwright)

**Status: implemented and executed live** (2026-07-23, `RUN_E2E=1` against this project's dev
ERPNext site, with `playwright install chromium`). Result: **10 passed, 1 xfailed (confirmed
platform limitation, not a bug), 1 failed (real, unfixed config gap)** — see below. Two real,
previously-unknown findings surfaced purely from actually running this against live
infrastructure, not from writing the tests:

1. **Missing webhook.** `test_webhook_config.py::test_all_required_webhooks_exist` fails for
   real: the `terms-on-update` webhook record (required per `docs/DEPLOYMENT.md`'s table) does
   not exist on this site — Terms and Conditions updates currently trigger no re-indexing at
   all. Three stale pre-rebrand webhook records (`po-on-submit`, `po-on-cancel`,
   `po-on-update-submitted`, `scorecard-on-update`) also still exist; harmless
   (`SUPPORTED_DOCTYPES` ignores them) but worth cleaning up. **Not fixed as part of this
   work** — an actionable follow-up, left failing intentionally so it stays visible.
2. **Confirmed platform gap, now on Contract too.** `on_update` does not fire for a Desk
   "Update" action on an already-submitted Contract — confirmed conclusively via Frappe's own
   **Webhook Request Log** doctype showing zero delivery attempts for that action window (not a
   failed delivery, one never attempted). This was previously only verified against Purchase
   Order (`on_update_after_submit`, before PO left scope); now confirmed for Contract's plain
   `on_update` too. See `docs/DEPLOYMENT.md` § Future Enhancements for full detail.
   `test_erpnext_desk_update_after_submit.py`'s first test is marked `xfail(strict=True)`
   documenting this — `strict=True` means if a future Frappe version starts firing the webhook,
   the test flips to an XPASS *failure*, so the fix gets noticed rather than silently masked.

End-to-end tests that drive the **ERPNext Desk UI via a real Chromium browser** to catch webhook-triggering bugs that REST-API integration tests miss. Not part of CI (requires live infrastructure); gated by `RUN_E2E=1`.

**Why this layer is needed — bugs found only via Desk UI, not by `test_integration.py`:**
1. Webhooks not configured in ERPNext at all (found: `terms-on-update`, above).
2. Webhooks configured but not firing (bad URL, disabled, background queue not running).
3. A webhook event not firing at all for a Desk save on a submitted document (found: Contract's `on_update`, above).

`test_integration.py` calls `frappe.client.submit` / `frappe.client.save` via REST, which bypasses the Frappe background worker queue that fires webhooks in production. Playwright drives the actual Desk UI through the same path a real user takes — which is exactly why it caught both findings above and REST-based `test_integration.py` couldn't.

**Scope note (post-rebrand):** the original version of this plan referenced Purchase Order, from
before the project narrowed to Contract Intelligence. The suite targets the doctypes actually in
`ingestion.webhook_handler.SUPPORTED_DOCTYPES` today — **Contract** (submittable; `on_submit`,
`on_update`, `on_cancel`) and **Terms and Conditions** (not submittable; `on_update` only).
The update-after-submit test edits Contract's `is_signed` field (`allow_on_submit`, a Check box,
confirmed live to auto-flip the (hidden) `status` field from "Unsigned" to "Active" via
ERPNext's own controller) via the Desk UI, rather than PO's `on_update_after_submit` path.

**Dependency:** `pytest-playwright` (added to `requirements.txt`). One-time setup: `playwright install chromium` (~300MB; already done on this machine).

**Files:**
```
tests/e2e/
├── __init__.py
├── conftest.py                               # Playwright fixtures, ERPNext Desk login, REST setup/poll helpers
├── test_webhook_config.py                    # REST assertions: records exist, enabled, correct doctype/event/URL (5 tests)
├── test_erpnext_desk_submit.py               # Desk submit Contract (+ PDF attachment) → poll Qdrant → assert indexed (2 tests)
├── test_erpnext_desk_update_after_submit.py  # Desk is_signed toggle + resave on submitted Contract → poll Qdrant (2 tests)
├── test_erpnext_desk_cancel.py               # Desk cancel → poll Qdrant → assert status=Cancelled (webhook_handler re-indexes, doesn't delete) (1 test)
└── test_streamlit_frontend.py                # Real browser against a running Streamlit + FastAPI, nothing mocked (2 tests)
```

**Test coverage and live result:**

| File | Tests | What it catches | Live result |
|---|---|---|---|
| `test_webhook_config.py` | 5 (REST, no browser) | Missing / misconfigured webhook records | 4 passed, 1 failed (missing `terms-on-update`, real gap) |
| `test_erpnext_desk_submit.py` | 2 | Webhook not firing on Desk submit | 2 passed |
| `test_erpnext_desk_update_after_submit.py` | 2 | `on_update` not firing on a post-submit Desk save | 1 xfailed (confirmed gap), 1 passed (repeated saves stay inert, no corruption) |
| `test_erpnext_desk_cancel.py` | 1 | Cancel not propagating to Qdrant (re-index with status=Cancelled) | 1 passed |
| `test_streamlit_frontend.py` | 2 | Frontend-level regressions AppTest can't see (real render, real network, real OpenAI round trip) | 2 passed |

**`test_streamlit_frontend.py`** drives an actual running Streamlit server (`streamlit run frontend/app.py`) through a real Chromium browser against the real FastAPI backend — the full browser → Streamlit → FastAPI → pipeline → Qdrant → OpenAI → browser round trip, nothing mocked. This is a different, complementary layer from `tests/test_streamlit.py` (below): AppTest verifies UI *logic* fast and in-process with `httpx.post` mocked; this verifies the same UI actually *renders and works* end-to-end. Login is bypassed by minting a JWT directly (`api.auth.jwt_handler.mint_token`, the same function `/auth/callback` calls after a real OAuth exchange) and navigating to `{FRONTEND_URL}/?token=<jwt>`, rather than driving ERPNext's OAuth consent screen through the browser (already covered by `conftest.py`'s Desk login and by `test_integration.py`'s auth group). The sidebar-filter test intentionally only asserts the query completes without error, not that results are strictly scoped to the filter — `docs/ARCHITECTURE.md`'s "Filter behaviour" section documents that metadata filters only narrow the Qdrant leg of hybrid search, so a supplier filter doesn't guarantee every returned source matches (confirmed empirically while writing this test).

**`tests/test_streamlit.py`** (separate file, **not** gated by `RUN_E2E` — runs as part of the normal `pytest tests/` unit suite on every push) tests the same UI via `streamlit.testing.v1.AppTest` (in-process, no browser, no running server, `httpx.post` mocked): basic query renders, sidebar supplier filter wires correct param to API, connection error shows a message without crashing. 3 cases, implemented and passing.

**Run:**
```bash
playwright install chromium   # one-time

RUN_E2E=1 pytest tests/e2e/ -v --headed   # watch in browser
RUN_E2E=1 pytest tests/e2e/ -v            # headless
pytest tests/test_streamlit.py -v         # no server or browser needed
```

**Not in CI** — requires live ERPNext, Qdrant, and FastAPI. Run manually before releases.

---

## Environment Variables Reference

| Variable | Required | Description |
|---|---|---|
| `ERPNEXT_URL` | Yes | Base URL of ERPNext instance |
| `ERPNEXT_API_KEY` | Yes | Frappe API key |
| `ERPNEXT_API_SECRET` | Yes | Frappe API secret |
| `WEBHOOK_SECRET` | Yes | HMAC secret for webhook validation (base64 HMAC-SHA256) |
| `OPENAI_API_KEY` | Yes | Used for embeddings + GPT-4o |
| `QDRANT_URL` | Yes | Qdrant server URL |
| `QDRANT_API_KEY` | No | Only needed for Qdrant Cloud |
| `QDRANT_COLLECTION` | Yes | Collection name (e.g. `procurement`) |
| `LANGFUSE_PUBLIC_KEY` | Yes | Langfuse project public key |
| `LANGFUSE_SECRET_KEY` | Yes | Langfuse project secret key |
| `LANGFUSE_HOST` | Yes | Langfuse server URL |
| `BACKEND_URL` | Yes | FastAPI base URL (used by Streamlit) |
| `ADMIN_SECRET` | Yes | Protects `/ingest/full` endpoint |
| `QUERY_REWRITE_STRATEGY` | No | `hyde` (default) or `step_back` |
| `OPENAI_MODEL` | No | Generation model (default `gpt-4o`); see `config.py` |
| `EMBEDDING_MODEL` | No | Embedding model (default `text-embedding-3-small`); changing to a different-dimension model requires recreating the Qdrant collection and a full re-ingest |
| `ERPNEXT_OAUTH_CLIENT_ID` | Yes | OAuth2 client ID from ERPNext desk → Integrations → OAuth Client |
| `ERPNEXT_OAUTH_CLIENT_SECRET` | Yes | OAuth2 client secret from the same record |
| `OAUTH_REDIRECT_URI` | Yes | Callback URL registered in the OAuth client (e.g. `http://localhost:8000/auth/callback`) |
| `JWT_SECRET` | Yes | Random 256-bit hex — signs session JWTs |
| `JWT_EXPIRY_HOURS` | No | Session lifetime in hours (default `8`) |
| `ALLOWED_ROLES` | No | Comma-separated ERPNext roles (default `Purchase Manager,Purchase User,Accounts User,System Manager`) |
| `FRONTEND_URL` | Yes | Streamlit URL — used by auth callback to redirect after login (e.g. `http://localhost:8501`) |

---

## Verification Checklist

Completed end-to-end against a live ERPNext instance (`prag01.test`) — see roadmap issue #17 (#29,
"End-to-end verification against live ERPNext") and PR #35.

- [x] ERPNext API key/secret generated and verified against the live REST API (Step 0)
- [x] Demo data exists (seeded if necessary) for `Supplier Scorecard`, `Contract`, and
      `Terms and Conditions` (Step 0)
- [x] `GET /health` returns `{"status": "ok"}`
- [x] `POST /ingest/full` with valid `X-Admin-Secret` triggers background ingest without errors
- [x] Qdrant collection exists and contains points after ingest
- [x] `POST /query` without `Authorization` header returns 401
- [x] `GET /auth/login` redirects to ERPNext OAuth2 authorize endpoint
- [x] Full auth flow: log in as `Purchase Manager` → JWT issued → `POST /query` returns answer + sources
- [x] `POST /query` with JWT for a non-allowed role returns 403
- [x] Sources reference real `docname` values from Qdrant
- [x] Webhook endpoint returns 200 and re-indexes on simulated POST (on_submit / on_cancel / on_update)
- [ ] Streamlit UI renders login button when unauthenticated; renders answer and sources after login
- [ ] Streamlit chat history preserved across turns within a session
- [ ] `python evaluation/evaluate.py` completes and writes `results.json`
- [ ] `pytest tests/` — all pass, no network calls made
- [ ] `ruff check .` — no errors
- [ ] `docker compose up` — all four services healthy
