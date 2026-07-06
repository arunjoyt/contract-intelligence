# Procurement Intelligence Assistant

A production-grade RAG application that answers natural language procurement queries grounded in ERPNext data.

## What It Does

Query your procurement data in plain English:

- "Which vendor has the best delivery SLA for electrical components?"
- "Are we violating any contract terms with PO-2024-1892?"
- "Which suppliers have contracts expiring in the next 60 days?"
- "What is the approved price ceiling for office supplies from Vendor X?"
- "Summarize all penalty clauses across active contracts."

## Tech Stack

| Layer | Technology |
|---|---|
| ERP | ERPNext (Frappe REST API + Webhooks) |
| Vector DB | Qdrant (self-hosted) |
| Orchestration | LangChain |
| Embeddings | `text-embedding-3-small` (OpenAI) |
| Re-ranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | GPT-4o |
| Evaluation | RAGAS |
| Observability | Langfuse (self-hosted) |
| API | FastAPI |
| Frontend | Streamlit |
| Auth | ERPNext OAuth2 + JWT (Option B) |
| Infra | Docker Compose + Nginx, GitHub Actions |

## Project Structure

```
procurement-rag/
├── ingestion/
│   ├── erpnext_client.py       # Frappe REST API wrapper
│   ├── document_parser.py      # HTML stripping, PDF extraction, struct→text
│   ├── chunker.py              # Recursive character text splitter
│   ├── embedder.py             # OpenAI embedding wrapper (batched)
│   └── webhook_handler.py      # FastAPI webhook receiver + re-indexer
├── retrieval/
│   ├── vector_store.py         # Qdrant client wrapper
│   ├── hybrid_search.py        # BM25 + vector fusion via RRF
│   └── reranker.py             # Cross-encoder re-ranking
├── pipeline/
│   ├── query_pipeline.py       # End-to-end RAG chain with Langfuse tracing
│   └── query_rewriter.py       # HyDE / step-back query rewriting
├── api/
│   ├── main.py                 # FastAPI app (query, webhook, ingest, health)
│   ├── auth/
│   │   ├── oauth2.py           # ERPNext OAuth2 — authorize URL, token exchange, role fetch
│   │   ├── pkce.py             # PKCE code verifier / challenge generation
│   │   ├── jwt_handler.py      # JWT mint and decode
│   │   └── dependencies.py     # FastAPI Depends: get_current_user, require_allowed_role
│   └── routers/
│       └── auth.py             # GET /auth/login, GET /auth/callback
├── frontend/
│   ├── app.py                  # Streamlit chat UI (gated behind session JWT)
│   └── auth_ui.py              # "Login with ERPNext" page, OAuth redirect, logout
├── evaluation/
│   ├── test_dataset.json       # Q&A pairs for RAGAS evaluation
│   ├── evaluate.py             # RAGAS runner
│   └── results.json            # Latest evaluation scores
├── tests/
│   ├── test_document_parser.py
│   ├── test_chunker.py
│   ├── test_embedder.py
│   ├── test_reranker.py
│   ├── test_hybrid_search.py
│   ├── test_vector_store.py
│   ├── test_query_rewriter.py
│   ├── test_query_pipeline.py
│   ├── test_webhook_handler.py
│   ├── test_erpnext_client.py
│   ├── test_api.py
│   └── test_integration.py     # Live ERPNext + Qdrant integration tests (opt-in)
├── docs/
│   ├── ARCHITECTURE.md         # System design and data flow
│   ├── IMPLEMENTATION_PLAN.md  # Ordered implementation steps
│   ├── DEPLOYMENT.md           # Auth options, infra topology, webhook setup
│   └── SECURITY_REVIEW.md      # Security review findings and accepted risks
├── nginx/
│   └── nginx.conf              # Reverse-proxy config (Option B production)
├── .github/
│   └── workflows/
│       └── ci.yml              # Lint, test, evaluate
├── docker-compose.yml          # Production: all services on rag_internal, nginx on 80/443
├── docker-compose.frontend.yml # Dev override: exposes service ports directly to host
├── Dockerfile
├── requirements.txt
├── pyproject.toml
├── .env.example
└── .gitignore
```

## Quick Start (local development)

1. Copy `.env.example` to `.env` and fill in all values.
2. In your ERPNext desk, create an OAuth2 client (**Integrations → OAuth Client**) and set the Redirect URI to `http://localhost:8000/auth/callback`. Copy the generated `client_id` and `client_secret` into `.env`.
3. Start infrastructure:
   ```bash
   docker compose -f docker-compose.yml -f docker-compose.frontend.yml up qdrant langfuse postgres -d
   ```
4. Start the API:
   ```bash
   uvicorn api.main:app --reload
   ```
5. Trigger a full ingest (requires a running ERPNext instance):
   ```bash
   curl -X POST http://localhost:8000/ingest/full \
     -H "X-Admin-Secret: <ADMIN_SECRET>"
   ```
6. Start the frontend:
   ```bash
   streamlit run frontend/app.py
   ```
7. Open `http://localhost:8501` — click **Login with ERPNext** and sign in with an allowed role (`Purchase Manager`, `Purchase User`, `Accounts User`, or `System Manager`).

## Production Deployment (Option B)

See `docs/DEPLOYMENT.md` for the full topology and step-by-step instructions. The short version:

1. Set your domains in `.env` (bare hostnames, no scheme — substituted into nginx's config
   automatically at container start, no manual file editing needed):
   ```
   FRONTEND_DOMAIN=procurement-rag.example.com
   API_DOMAIN=api.procurement-rag.example.com
   ```
2. Provision TLS certs for those same domains:
   ```bash
   certbot certonly --standalone \
     -d procurement-rag.example.com \
     -d api.procurement-rag.example.com
   ```
3. Fill in the rest of the production values in `.env`:
   - Strong random values for `WEBHOOK_SECRET`, `ADMIN_SECRET`, `JWT_SECRET`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`
   - Set URL vars to your actual domains:
     ```
     OAUTH_REDIRECT_URI=https://api.procurement-rag.example.com/auth/callback
     FRONTEND_URL=https://procurement-rag.example.com
     PUBLIC_API_URL=https://api.procurement-rag.example.com
     ```
   - Update the ERPNext OAuth client's Redirect URI to match `OAUTH_REDIRECT_URI`
4. Start everything:
   ```bash
   docker compose up -d
   ```
   All services run on the internal `rag_internal` network; nginx is the only service with public ports (80/443).

## Environment Variables

See `.env.example` for the full list with generation instructions. Key groups:

| Group | Variables | Purpose |
|---|---|---|
| ERPNext | `ERPNEXT_URL`, `ERPNEXT_API_KEY`, `ERPNEXT_API_SECRET` | Frappe REST API access |
| Webhook | `WEBHOOK_SECRET` | HMAC-SHA256 signature verification |
| OpenAI | `OPENAI_API_KEY` | Embeddings and GPT-4o |
| Qdrant | `QDRANT_URL`, `QDRANT_COLLECTION` | Vector store |
| Langfuse | `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` | Tracing |
| Auth | `ERPNEXT_OAUTH_CLIENT_ID`, `ERPNEXT_OAUTH_CLIENT_SECRET`, `JWT_SECRET`, `ALLOWED_ROLES` | OAuth2 + JWT |
| URLs | `OAUTH_REDIRECT_URI`, `FRONTEND_URL`, `PUBLIC_API_URL` | OAuth callback and post-login redirect targets; must use public domains in production |
| Admin | `ADMIN_SECRET` | Gate for `POST /ingest/full` |
| Pipeline | `QUERY_REWRITE_STRATEGY` | `hyde` (default) or `step_back` |

## Data Sources

| Doctype | Indexing type | Notes |
|---|---|---|
| Purchase Order | Structured → natural language | Serialized fields: supplier, items, rate, dates, status |
| Contract | Unstructured chunks | `contract_terms` HTML stripped before chunking |
| Supplier Scorecard | Structured → natural language | Scoring criteria and standing |
| Purchase Invoice | Structured → natural language | Payment terms, outstanding amounts |
| Terms and Conditions | Unstructured chunks | Template text |
| Attached PDFs | Unstructured chunks | Extracted via Frappe File API |

## Incremental Indexing

ERPNext webhooks fire on `on_submit` / `on_cancel` / `on_update` events. The webhook endpoint:
1. Validates the HMAC-SHA256 signature (`X-Frappe-Webhook-Signature`)
2. Fetches the full updated document via ERPNext REST API
3. Deletes all existing Qdrant chunks for that `docname`
4. Re-parses, chunks, embeds, and upserts

See `docs/DEPLOYMENT.md` for the required ERPNext webhook records and scripted setup.

## Evaluation

```bash
python evaluation/evaluate.py
```

Runs RAGAS metrics (`faithfulness`, `answer_relevancy`, `context_recall`, `context_precision`) against `evaluation/test_dataset.json` and writes scores to `evaluation/results.json`.

## CI/CD

GitHub Actions runs on every push:
- `ruff check .` — linting
- `pytest tests/` — unit tests (no network required; OpenAI and Qdrant are mocked)
- On merge to `main`: RAGAS evaluation with results uploaded as artifact

Integration tests (`tests/test_integration.py`) require a live ERPNext + Qdrant instance and are opt-in via `RUN_INTEGRATION=1`.
