"""FastAPI application — procurement RAG API.

Endpoints
---------
GET  /health          — liveness check
POST /query           — run the RAG pipeline
POST /ingest/full     — full re-index (background task, X-Admin-Secret gated)
POST /webhook/erpnext — incremental re-index on ERPNext webhook

All singletons (Embedder, VectorStore, HybridSearch, Reranker, QueryPipeline,
ERPNextClient) are created during startup and stored on ``app.state`` so every
request handler can reach them without module-level globals.
"""

from __future__ import annotations

import hmac
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Any

from dotenv import load_dotenv
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel

from api.auth.dependencies import require_allowed_role
from api.routers.auth import router as auth_router
from ingestion.chunker import chunk_text
from ingestion.embedder import Embedder
from ingestion.erpnext_client import ERPNextClient
from ingestion.webhook_handler import handle_webhook_request, prepare_doc_for_indexing
from pipeline.query_pipeline import QueryPipeline
from pipeline.query_rewriter import QueryRewriter
from retrieval.hybrid_search import HybridSearch
from retrieval.reranker import Reranker
from retrieval.vector_store import VectorStore

load_dotenv()

logger = logging.getLogger(__name__)

_INGEST_DOCTYPES = ("Purchase Order", "Contract", "Supplier Scorecard")


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(app: FastAPI):
    embedder = Embedder()
    vector_store = VectorStore()
    vector_store.ensure_collection()

    hybrid_search = HybridSearch(embedder=embedder, vector_store=vector_store)
    hybrid_search.build_bm25_index(vector_store.get_all_texts())

    reranker = Reranker()
    reranker.warm_up()

    rewriter = QueryRewriter(embedder=embedder)

    lf = None
    lf_public = os.environ.get("LANGFUSE_PUBLIC_KEY")
    lf_secret = os.environ.get("LANGFUSE_SECRET_KEY")
    if lf_public and lf_secret:
        from langfuse import Langfuse
        lf = Langfuse(
            public_key=lf_public,
            secret_key=lf_secret,
            host=os.environ.get("LANGFUSE_HOST"),
        )
        logger.info("Langfuse tracing enabled")
    else:
        logger.warning("LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set — tracing disabled")

    pipeline = QueryPipeline(
        rewriter=rewriter, hybrid_search=hybrid_search, reranker=reranker, langfuse=lf
    )

    erpnext_client = ERPNextClient()

    webhook_secret = os.environ.get("WEBHOOK_SECRET", "")
    if not webhook_secret:
        logger.warning("WEBHOOK_SECRET is not set — webhook endpoint will reject all requests")

    app.state.embedder = embedder
    app.state.vector_store = vector_store
    app.state.hybrid_search = hybrid_search
    app.state.pipeline = pipeline
    app.state.erpnext_client = erpnext_client
    app.state.webhook_secret = webhook_secret
    app.state.oauth_state: dict[str, str] = {}

    logger.info("Startup complete — collection ready, BM25 built, reranker warm")

    yield

    if lf:
        lf.flush()
    await erpnext_client.aclose()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Procurement RAG API", lifespan=_lifespan)
app.include_router(auth_router)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    question: str
    filters: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/query")
def query(req: QueryRequest, _user: dict = Depends(require_allowed_role)) -> dict[str, Any]:  # noqa: B008
    result = app.state.pipeline.run(req.question, filters=req.filters)
    return {
        "answer": result["answer"],
        "sources": [asdict(s) for s in result["sources"]],
    }


@app.post("/ingest/full", status_code=202)
def ingest_full(
    background_tasks: BackgroundTasks,
    x_admin_secret: str | None = Header(default=None),
) -> dict[str, str]:
    _check_admin_secret(x_admin_secret)
    background_tasks.add_task(
        _run_full_ingest,
        app.state.embedder,
        app.state.vector_store,
        app.state.hybrid_search,
    )
    return {"status": "accepted"}


@app.post("/webhook/erpnext")
async def webhook_erpnext(request: Request) -> dict[str, str]:
    return await handle_webhook_request(
        request=request,
        erpnext_client=app.state.erpnext_client,
        embedder=app.state.embedder,
        vector_store=app.state.vector_store,
        rebuild_bm25=lambda: app.state.hybrid_search.build_bm25_index(
            app.state.vector_store.get_all_texts()
        ),
        webhook_secret=app.state.webhook_secret,
    )


# ---------------------------------------------------------------------------
# Full ingest background task
# ---------------------------------------------------------------------------


async def _run_full_ingest(
    embedder: Embedder,
    vector_store: VectorStore,
    hybrid_search: HybridSearch,
) -> None:
    logger.info("Full ingest started")
    async with ERPNextClient() as client:
        for doctype in _INGEST_DOCTYPES:
            try:
                doc_list = await client.get_list(doctype, limit=0)
            except Exception:
                logger.exception("Failed to list %s", doctype)
                continue

            logger.info("Ingesting %d %s documents", len(doc_list), doctype)
            for entry in doc_list:
                name: str = entry["name"]
                try:
                    doc = await client.get_doc(doctype, name)
                    supplier_name = doc.get("supplier") or doc.get("party_name")
                    supplier_group: str | None = None
                    if supplier_name:
                        try:
                            supplier = await client.get_doc("Supplier", supplier_name)
                            supplier_group = supplier.get("supplier_group")
                        except Exception:
                            pass

                    text, metadata, force_single = prepare_doc_for_indexing(
                        doctype, doc, supplier_group
                    )
                    if not text.strip():
                        continue

                    chunks = chunk_text(text, force_single_chunk=force_single)
                    vectors = embedder.embed_texts([c["text"] for c in chunks])
                    enriched = [
                        {**chunk, **metadata, "vector": vector}
                        for chunk, vector in zip(chunks, vectors, strict=True)
                    ]
                    vector_store.upsert_chunks(enriched)
                except Exception:
                    logger.exception("Failed to ingest %s %s", doctype, name)

    hybrid_search.build_bm25_index(vector_store.get_all_texts())
    logger.info("Full ingest complete")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_admin_secret(provided: str | None) -> None:
    expected = os.environ.get("ADMIN_SECRET") or ""
    if not expected or not provided or not hmac.compare_digest(expected, provided):
        raise HTTPException(status_code=403, detail="Invalid or missing admin secret")
