# Implementation Plan

Ordered list of modules to build. Each step depends on the ones before it. Do not reorder.

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

Model: `text-embedding-3-small` (1536 dims).

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

Supported doctypes: `Purchase Order`, `Contract`, `Supplier Scorecard`.

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
Prompt GPT-4o: *"Write a hypothetical procurement document that would answer: {query}"*  
Embed the hypothetical answer; use that vector for retrieval.

**Step-back strategy:**  
Prompt GPT-4o to rewrite the question at a higher abstraction level, then embed the rewritten question.

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
6. Call `ChatOpenAI(model="gpt-4o")` with system prompt requiring source citations
7. Parse response into `answer` + `sources`

**System prompt key points:**
- Answer only from the provided context
- Cite each claim with `[docname]`
- If the answer is not in context, say so explicitly — do not hallucinate

---

## Step 12 — `api/main.py`

```python
app = FastAPI()

POST /query           # {"question": str, "filters": dict|null} → pipeline response
POST /webhook/erpnext # delegated to webhook_handler router
POST /ingest/full     # manual full re-index (background task, X-Admin-Secret required)
GET  /health          # {"status": "ok"}
```

Startup event:
1. Call `VectorStore.ensure_collection()`
2. Fetch all texts from Qdrant → build BM25 index
3. Warm up reranker model

---

## Step 13 — `frontend/app.py`

Streamlit app layout:
- **Sidebar:** supplier filter (text), doctype filter (multiselect), date range, status filter
- **Main:** `st.chat_input` for question entry
- **Response:** answer text block + collapsible "Sources" expander listing `docname`, `source_doctype`, `supplier` per citation
- **Session state:** `messages` list (last 10 turns displayed as chat history)

Calls `POST {BACKEND_URL}/query` via `httpx`.

---

## Step 14 — `evaluation/test_dataset.json` + `evaluation/evaluate.py`

**`test_dataset.json`** — 15 Q&A pairs, one per query pattern, with stored ground-truth contexts (so evaluation runs without a live ERPNext).

**`evaluate.py`:**
```python
# Loads test_dataset.json
# Runs pipeline.run(question) for each entry
# Computes RAGAS: faithfulness, answer_relevancy, context_recall, context_precision
# Writes evaluation/results.json
```

---

## Step 15 — `docker-compose.yml`

| Service | Image | Port | Volume |
|---|---|---|---|
| `postgres` | `postgres:16-alpine` | 5432 | `pg_data` |
| `langfuse` | `langfuse/langfuse:latest` | 3000 | — |
| `qdrant` | `qdrant/qdrant:latest` | 6333 | `qdrant_data` |
| `app` | local `Dockerfile` | 8000 | — |

`docker-compose.frontend.yml` adds `streamlit` service on port 8501.

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

## Environment Variables Reference

| Variable | Required | Description |
|---|---|---|
| `ERPNEXT_URL` | Yes | Base URL of ERPNext instance |
| `ERPNEXT_API_KEY` | Yes | Frappe API key |
| `ERPNEXT_API_SECRET` | Yes | Frappe API secret |
| `WEBHOOK_SECRET` | Yes | HMAC secret for webhook validation |
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

---

## Verification Checklist

- [ ] `GET /health` returns `{"status": "ok"}`
- [ ] `POST /ingest/full` with valid `X-Admin-Secret` triggers background ingest without errors
- [ ] Qdrant collection exists and contains points after ingest
- [ ] `POST /query` with example question returns answer + non-empty sources
- [ ] Sources reference real `docname` values from Qdrant
- [ ] Webhook endpoint returns 200 and re-indexes on simulated POST
- [ ] Streamlit UI renders answer and sources, preserves chat history across turns
- [ ] `python evaluation/evaluate.py` completes and writes `results.json`
- [ ] `pytest tests/` — all pass, no network calls made
- [ ] `ruff check .` — no errors
- [ ] `docker compose up` — all four services healthy
