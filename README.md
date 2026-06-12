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
| Infra | Docker Compose, GitHub Actions |

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
│   └── main.py                 # FastAPI app (query, webhook, ingest, health)
├── evaluation/
│   ├── test_dataset.json       # Q&A pairs for RAGAS evaluation
│   └── evaluate.py             # RAGAS runner
├── frontend/
│   └── app.py                  # Streamlit UI
├── tests/
│   ├── test_document_parser.py
│   ├── test_chunker.py
│   ├── test_embedder.py
│   ├── test_reranker.py
│   └── test_query_pipeline.py
├── docs/
│   ├── ARCHITECTURE.md         # System design and data flow
│   └── IMPLEMENTATION_PLAN.md  # Ordered implementation steps
├── .github/
│   └── workflows/
│       └── ci.yml              # Lint, test, evaluate
├── docker-compose.yml          # Qdrant + app + Langfuse + Postgres
├── docker-compose.frontend.yml # Streamlit (optional separate service)
├── Dockerfile
├── requirements.txt
├── pyproject.toml
├── .env.example
└── .gitignore
```

## Quick Start

1. Copy `.env.example` to `.env` and fill in all values.
2. Start infrastructure:
   ```bash
   docker compose up qdrant langfuse postgres -d
   ```
3. Start the API:
   ```bash
   uvicorn api.main:app --reload
   ```
4. Trigger a full ingest (requires a running ERPNext instance):
   ```bash
   curl -X POST http://localhost:8000/ingest/full \
     -H "X-Admin-Secret: <ADMIN_SECRET>"
   ```
5. Start the frontend:
   ```bash
   streamlit run frontend/app.py
   ```

## Environment Variables

See `.env.example` for the full list. Key groups:

- `ERPNEXT_*` — ERPNext connection and credentials
- `OPENAI_API_KEY` — used for embeddings and GPT-4o
- `QDRANT_*` — vector store connection
- `LANGFUSE_*` — observability backend
- `WEBHOOK_SECRET` — HMAC secret for ERPNext webhook validation
- `QUERY_REWRITE_STRATEGY` — `hyde` (default) or `step_back`

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

ERPNext webhooks fire on `save`/`submit` events. The webhook endpoint:
1. Validates the HMAC signature
2. Fetches the full updated document
3. Deletes all existing Qdrant chunks for that docname
4. Re-parses, chunks, embeds, and upserts

No full re-index is needed for routine updates.

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
