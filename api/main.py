"""FastAPI application — Contract Intelligence RAG API.

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

import asyncio
import hmac
import logging
import os
from contextlib import asynccontextmanager, nullcontext
from dataclasses import asdict
from typing import Any
from urllib.parse import quote

from dotenv import load_dotenv
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel

from api.auth.dependencies import require_allowed_role
from api.routers.auth import router as auth_router
from ingestion.embedder import Embedder
from ingestion.erpnext_client import ERPNextClient
from ingestion.tracing import span, traced_embed
from ingestion.webhook_handler import (
    gather_chunks_for_doc,
    handle_webhook_request,
    prepare_doc_for_indexing,
    resolve_supplier_group,
)
from pipeline.query_pipeline import QueryPipeline
from pipeline.query_rewriter import QueryRewriter
from retrieval.hybrid_search import HybridSearch
from retrieval.reranker import Reranker
from retrieval.vector_store import VectorStore

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

logger = logging.getLogger(__name__)

_INGEST_DOCTYPES = ("Contract", "Terms and Conditions")


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
    app.state.langfuse = lf
    app.state.erpnext_client = erpnext_client
    app.state.webhook_secret = webhook_secret
    # Serializes full-ingest runs: overlapping runs are harmless but wasteful
    # (each re-embeds the whole corpus), so a second request while one is in
    # flight is rejected rather than queued (see issue #125).
    app.state.ingest_lock = asyncio.Lock()
    app.state.oauth_state: dict[str, str] = {}
    app.state.oauth_completed: dict[str, tuple[float, str, str]] = {}

    logger.info("Startup complete — collection ready, BM25 built, reranker warm")

    yield

    if lf:
        lf.flush()
    await erpnext_client.aclose()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Contract Intelligence API", lifespan=_lifespan)
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
    sources = []
    for s in result["sources"]:
        source = asdict(s)
        source["erpnext_url"] = _erpnext_record_url(s.source_doctype, s.docname)
        sources.append(source)
    return {"answer": result["answer"], "sources": sources}


@app.post("/ingest/full", status_code=202)
def ingest_full(
    background_tasks: BackgroundTasks,
    x_admin_secret: str | None = Header(default=None),
) -> dict[str, str]:
    _check_admin_secret(x_admin_secret)
    if app.state.ingest_lock.locked():
        raise HTTPException(status_code=409, detail="A full ingest is already running")
    background_tasks.add_task(
        _run_full_ingest,
        app.state.embedder,
        app.state.vector_store,
        app.state.hybrid_search,
        app.state.langfuse,
        app.state.ingest_lock,
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
        langfuse=app.state.langfuse,
    )


# ---------------------------------------------------------------------------
# Full ingest background task
# ---------------------------------------------------------------------------


async def _run_full_ingest(
    embedder: Embedder,
    vector_store: VectorStore,
    hybrid_search: HybridSearch,
    langfuse: Any | None = None,
    lock: asyncio.Lock | None = None,
) -> None:
    if lock is not None and lock.locked():
        # A run slipped past the endpoint's `locked()` check before this task
        # started (issue #125). Skip rather than pile on a redundant re-embed.
        logger.warning("Full ingest already running — skipping duplicate run")
        return
    async with lock or nullcontext():
        await _do_full_ingest(embedder, vector_store, hybrid_search, langfuse)


async def _do_full_ingest(
    embedder: Embedder,
    vector_store: VectorStore,
    hybrid_search: HybridSearch,
    langfuse: Any | None,
) -> None:
    logger.info("Full ingest started")
    trace = langfuse.trace(name="full_ingest") if langfuse else None
    docs_indexed = 0
    chunks_indexed = 0
    failed_listings: list[str] = []

    async with ERPNextClient() as client:
        for doctype in _INGEST_DOCTYPES:
            # Traced (not just logged) so a run that indexes nothing because the
            # ERPNext listing failed shows *why* in the trace, not just a 0/0
            # output with no observations.
            try:
                with span(trace, f"list:{doctype}") as list_span:
                    doc_list = await client.get_list(doctype, limit=0)
                    if list_span:
                        list_span.update(output={"count": len(doc_list)})
            except Exception:
                logger.exception("Failed to list %s", doctype)
                failed_listings.append(doctype)
                continue

            logger.info("Ingesting %d %s documents", len(doc_list), doctype)
            for entry in doc_list:
                name: str = entry["name"]
                try:
                    with span(trace, f"{doctype}:{name}"):
                        doc = await client.get_doc(doctype, name)
                        supplier_group = await resolve_supplier_group(doctype, doc, client)

                        text, metadata, force_single = prepare_doc_for_indexing(
                            doctype, doc, supplier_group
                        )
                        chunks = await gather_chunks_for_doc(
                            doctype, doc, text, force_single, client
                        )
                        if not chunks:
                            continue

                        vectors = traced_embed(trace, embedder, [c["text"] for c in chunks])
                        enriched = [
                            {**chunk, **metadata, "vector": vector}
                            for chunk, vector in zip(chunks, vectors, strict=True)
                        ]
                        vector_store.upsert_chunks(enriched)
                        docs_indexed += 1
                        chunks_indexed += len(chunks)
                except Exception:
                    logger.exception("Failed to ingest %s %s", doctype, name)

    hybrid_search.build_bm25_index(vector_store.get_all_texts())

    # A run that indexes nothing, or where a doctype listing errored, is never a
    # legitimate outcome for this app -- surface it instead of reporting a clean
    # "complete" (issue #125).
    failed = bool(failed_listings) or docs_indexed == 0
    output: dict[str, Any] = {
        "documents_indexed": docs_indexed,
        "chunks_indexed": chunks_indexed,
    }
    if failed:
        output["status"] = "error"
    if failed_listings:
        output["failed_listings"] = failed_listings
    if trace:
        trace.update(output=output)

    if failed:
        logger.error(
            "Full ingest finished abnormally: indexed %d docs, failed listings: %s",
            docs_indexed,
            failed_listings or "none",
        )
    else:
        logger.info("Full ingest complete")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _erpnext_record_url(source_doctype: str, docname: str) -> str:
    """Build the ERPNext desk URL for a cited source record.

    Doctype slugs follow Frappe's own routing convention: lowercased, spaces
    replaced with hyphens (e.g. "Terms and Conditions" -> "terms-and-conditions").
    """
    slug = source_doctype.lower().replace(" ", "-")
    base_url = os.environ["ERPNEXT_URL"].rstrip("/")
    return f"{base_url}/app/{slug}/{quote(docname, safe='')}"


def _check_admin_secret(provided: str | None) -> None:
    expected = os.environ.get("ADMIN_SECRET") or ""
    if not expected or not provided or not hmac.compare_digest(expected, provided):
        raise HTTPException(status_code=403, detail="Invalid or missing admin secret")
