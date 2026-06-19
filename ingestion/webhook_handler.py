"""Incremental re-indexing via ERPNext webhooks.

POST /webhook/erpnext — verifies HMAC-SHA256 signature, fetches the updated
document, deletes its existing Qdrant points, and re-indexes end-to-end.

Only Purchase Order, Contract, and Supplier Scorecard trigger re-indexing;
all other doctypes return {"status": "ignored"} so ERPNext can freely fire
webhooks for any doctype without breaking anything.

Design note: dependencies (erpnext_client, embedder, vector_store, rebuild_bm25)
are injected via the `create_webhook_router` factory rather than imported at
module level. This keeps the router testable without a live Qdrant/OpenAI and
lets api/main.py wire up shared singletons in Step 12.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from ingestion.chunker import chunk_text
from ingestion.document_parser import (
    extract_text_from_html,
    po_to_text,
    supplier_scorecard_to_text,
)
from ingestion.embedder import Embedder
from ingestion.erpnext_client import ERPNextClient
from retrieval.vector_store import VectorStore

SUPPORTED_DOCTYPES = frozenset({"Purchase Order", "Contract", "Supplier Scorecard"})


async def handle_webhook_request(
    request: Request,
    erpnext_client: ERPNextClient,
    embedder: Embedder,
    vector_store: VectorStore,
    rebuild_bm25: Callable[[], None],
    webhook_secret: str,
) -> dict[str, str]:
    """Process a single ERPNext webhook request.

    Verifies the HMAC signature, fetches and re-indexes the document.
    Called by both ``create_webhook_router`` (test-friendly factory) and
    ``api/main.py`` (which registers the route at module level).
    """
    body = await request.body()
    _verify_signature(body, request.headers.get("X-Frappe-Webhook-Signature", ""), webhook_secret)

    payload = json.loads(body)
    doctype: str = payload.get("doctype", "")
    docname: str = payload.get("docname", "")

    if doctype not in SUPPORTED_DOCTYPES:
        return {"status": "ignored", "doctype": doctype}

    doc = await erpnext_client.get_doc(doctype, docname)
    supplier_name = doc.get("supplier") or doc.get("party_name")
    supplier_group = await _fetch_supplier_group(erpnext_client, supplier_name)

    text, metadata, force_single = prepare_doc_for_indexing(doctype, doc, supplier_group)

    if not text.strip():
        return {"status": "skipped", "docname": docname, "reason": "empty text"}

    chunks = chunk_text(text, force_single_chunk=force_single)
    vectors = embedder.embed_texts([c["text"] for c in chunks])

    enriched = [
        {**chunk, **metadata, "vector": vector}
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]

    vector_store.delete_by_docname(docname)
    vector_store.upsert_chunks(enriched)
    rebuild_bm25()

    return {"status": "indexed", "docname": docname}


def create_webhook_router(
    erpnext_client: ERPNextClient,
    embedder: Embedder,
    vector_store: VectorStore,
    rebuild_bm25: Callable[[], None],
    webhook_secret: str | None = None,
) -> APIRouter:
    """Return an APIRouter with POST /erpnext wired to the given dependencies.

    Mount at ``/webhook`` in api/main.py so the full path is
    ``POST /webhook/erpnext``.
    """
    _secret = webhook_secret or os.environ["WEBHOOK_SECRET"]
    router = APIRouter()

    @router.post("/erpnext")
    async def _handler(request: Request) -> dict[str, str]:
        return await handle_webhook_request(
            request=request,
            erpnext_client=erpnext_client,
            embedder=embedder,
            vector_store=vector_store,
            rebuild_bm25=rebuild_bm25,
            webhook_secret=_secret,
        )

    return router


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _verify_signature(body: bytes, signature: str, secret: str) -> None:
    expected = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, signature):
        raise HTTPException(status_code=401, detail="Invalid webhook signature")


async def _fetch_supplier_group(client: ERPNextClient, supplier_name: str | None) -> str | None:
    """Fetch ``supplier_group`` from the Supplier record.

    PO and Contract don't carry supplier_group directly — it lives on the
    Supplier doctype and must be joined here (see ARCHITECTURE.md "Supplier
    Metadata Enrichment"). Returns None silently on any lookup failure so a
    bad/missing supplier record never blocks indexing.
    """
    if not supplier_name:
        return None
    try:
        supplier = await client.get_doc("Supplier", supplier_name)
        return supplier.get("supplier_group")
    except Exception:
        return None


def prepare_doc_for_indexing(
    doctype: str, doc: dict[str, Any], supplier_group: str | None
) -> tuple[str, dict[str, Any], bool]:
    """Return ``(text, metadata, force_single_chunk)`` for the given doctype.

    Structured docs (PO, Scorecard) pass force_single_chunk=True so they are
    never fragmented across vectors. Contracts are unstructured HTML and split
    normally via RecursiveCharacterTextSplitter.
    """
    if doctype == "Purchase Order":
        return (
            po_to_text(doc),
            {
                "source_doctype": "Purchase Order",
                "docname": doc["name"],
                "supplier": doc.get("supplier"),
                "supplier_group": supplier_group,
                "start_date": doc.get("transaction_date"),
                "end_date": doc.get("schedule_date"),
                "status": doc.get("status"),
                "company": doc.get("company"),
            },
            True,
        )

    if doctype == "Contract":
        return (
            extract_text_from_html(doc.get("contract_terms")),
            {
                "source_doctype": "Contract",
                "docname": doc["name"],
                "supplier": doc.get("party_name"),
                "supplier_group": supplier_group,
                "start_date": doc.get("start_date"),
                "end_date": doc.get("end_date"),
                "status": doc.get("status"),
                "company": doc.get("company"),
            },
            False,
        )

    # Supplier Scorecard — scorecards have no company or date range fields
    return (
        supplier_scorecard_to_text(doc),
        {
            "source_doctype": "Supplier Scorecard",
            "docname": doc["name"],
            "supplier": doc.get("supplier"),
            "supplier_group": supplier_group,
            "start_date": None,
            "end_date": None,
            "status": doc.get("status"),
            "company": None,
        },
        True,
    )
