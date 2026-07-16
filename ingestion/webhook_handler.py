"""Incremental re-indexing via ERPNext webhooks.

POST /webhook/erpnext — verifies HMAC-SHA256 signature, fetches the updated
document, deletes its existing Qdrant points, and re-indexes end-to-end.

Only doctypes in SUPPORTED_DOCTYPES trigger re-indexing; all other doctypes
return {"status": "ignored"} so ERPNext can freely fire webhooks for any
doctype without breaking anything.

Design note: dependencies (erpnext_client, embedder, vector_store, rebuild_bm25)
are injected via the `create_webhook_router` factory rather than imported at
module level. This keeps the router testable without a live Qdrant/OpenAI and
lets api/main.py wire up shared singletons in Step 12. `prepare_doc_for_indexing`,
`resolve_supplier_group`, and `gather_chunks_for_doc` are also imported directly
by api/main.py's full-ingest task so both indexing paths share one implementation
per doctype and can't drift apart.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from ingestion.chunker import Chunk, chunk_text
from ingestion.document_parser import (
    extract_text_from_html,
    extract_text_from_pdf,
    invoice_to_text,
    po_to_text,
    supplier_scorecard_to_text,
)
from ingestion.embedder import Embedder
from ingestion.erpnext_client import ERPNextClient
from retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)

SUPPORTED_DOCTYPES = frozenset(
    {
        "Purchase Order",
        "Purchase Invoice",
        "Contract",
        "Terms and Conditions",
        "Supplier Scorecard",
    }
)

# Doctypes whose attached PDF files get extracted and indexed alongside the
# parent document's own text (see docs/ARCHITECTURE.md "Document Indexing
# Strategy" and issue #52).
ATTACHMENT_DOCTYPES = frozenset({"Purchase Order", "Contract"})


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
    # Frappe sends full-document payloads with "name"; webhook_data mapping sends "docname".
    docname: str = payload.get("docname") or payload.get("name", "")

    if doctype not in SUPPORTED_DOCTYPES:
        return {"status": "ignored", "doctype": doctype}

    doc = await erpnext_client.get_doc(doctype, docname)
    supplier_group = await resolve_supplier_group(doctype, doc, erpnext_client)

    text, metadata, force_single = prepare_doc_for_indexing(doctype, doc, supplier_group)
    chunks = await gather_chunks_for_doc(doctype, doc, text, force_single, erpnext_client)

    if not chunks:
        return {"status": "skipped", "docname": docname, "reason": "empty text"}

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
    if not secret:
        raise HTTPException(status_code=500, detail="Webhook secret not configured")
    # Frappe sends HMAC-SHA256 as base64 (base64.b64encode(...digest())), not hex.
    import base64
    expected = base64.b64encode(
        hmac.new(secret.encode("utf-8"), body, hashlib.sha256).digest()
    ).decode()
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


async def resolve_supplier_group(
    doctype: str, doc: dict[str, Any], client: ERPNextClient
) -> str | None:
    """Return ``supplier_group`` for ``doc``, joining via Supplier when needed.

    Purchase Invoice carries `supplier_group` directly on the record; PO and
    Contract don't and require a Supplier lookup (see ARCHITECTURE.md "Supplier
    Metadata Enrichment"). Terms and Conditions has no supplier at all, so the
    Supplier lookup below is skipped automatically (no `supplier`/`party_name`).
    """
    if doctype == "Purchase Invoice":
        return doc.get("supplier_group")
    supplier_name = doc.get("supplier") or doc.get("party_name")
    return await _fetch_supplier_group(client, supplier_name)


async def gather_chunks_for_doc(
    doctype: str,
    doc: dict[str, Any],
    text: str,
    force_single: bool,
    erpnext_client: ERPNextClient,
) -> list[Chunk]:
    """Return every chunk to index for ``doc``: its own serialized/HTML text plus,
    for doctypes in ATTACHMENT_DOCTYPES, any attached PDFs' extracted text.

    All chunks share the same docname (the caller merges in `metadata` separately),
    so `chunk_index`/`total_chunks` are renumbered across the combined set — this
    keeps `delete_by_docname` and reconstruction correct even when the parent doc's
    own text is force_single_chunk and the attachments are split into many pieces.
    """
    chunks: list[Chunk] = chunk_text(text, force_single_chunk=force_single) if text.strip() else []

    if doctype in ATTACHMENT_DOCTYPES:
        chunks.extend(await _pdf_chunks_for_attachments(doctype, doc["name"], erpnext_client))

    if not chunks:
        return []

    total = len(chunks)
    return [{**c, "chunk_index": i, "total_chunks": total} for i, c in enumerate(chunks)]


async def _pdf_chunks_for_attachments(
    doctype: str, docname: str, client: ERPNextClient
) -> list[Chunk]:
    try:
        files = await client.get_attached_files(doctype, docname)
    except Exception:
        logger.exception("Failed to list attachments for %s %s", doctype, docname)
        return []

    chunks: list[Chunk] = []
    for f in files:
        file_name = f.get("file_name") or ""
        if not file_name.lower().endswith(".pdf"):
            continue
        try:
            pdf_bytes = await client.get_file_content(f["file_url"])
            pdf_text = extract_text_from_pdf(pdf_bytes)
        except Exception:
            logger.exception(
                "Failed to extract PDF attachment %s for %s %s", file_name, doctype, docname
            )
            continue
        if pdf_text.strip():
            chunks.extend(chunk_text(pdf_text))
    return chunks


def prepare_doc_for_indexing(
    doctype: str, doc: dict[str, Any], supplier_group: str | None
) -> tuple[str, dict[str, Any], bool]:
    """Return ``(text, metadata, force_single_chunk)`` for the given doctype.

    Structured docs (PO, Invoice, Scorecard) pass force_single_chunk=True so they
    are never fragmented across vectors. Contract and Terms and Conditions are
    unstructured HTML and split normally via RecursiveCharacterTextSplitter.
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

    if doctype == "Purchase Invoice":
        return (
            invoice_to_text(doc),
            {
                "source_doctype": "Purchase Invoice",
                "docname": doc["name"],
                "supplier": doc.get("supplier"),
                "supplier_group": supplier_group,
                "start_date": doc.get("posting_date"),
                "end_date": doc.get("due_date"),
                "status": doc.get("status"),
                "company": doc.get("company"),
            },
            True,
        )

    if doctype == "Terms and Conditions":
        # No supplier, company, or date-range fields on this doctype; "status" is
        # derived from the `disabled` checkbox so status filters (e.g. "active")
        # still work.
        return (
            extract_text_from_html(doc.get("terms")),
            {
                "source_doctype": "Terms and Conditions",
                "docname": doc["name"],
                "supplier": None,
                "supplier_group": None,
                "start_date": None,
                "end_date": None,
                "status": "Disabled" if doc.get("disabled") else "Active",
                "company": None,
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
