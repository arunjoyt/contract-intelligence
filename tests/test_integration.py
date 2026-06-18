"""Integration tests — require live external services and are skipped in CI.

These tests exercise the full end-to-end stack that unit tests cannot cover
because all external I/O is mocked there.  Run them manually against a local
docker-compose stack (Qdrant + Langfuse + Postgres) and a reachable ERPNext
site, with real env vars set.

To run a specific group:
    pytest tests/test_integration.py -m "ingestion" -v
    pytest tests/test_integration.py -m "retrieval" -v
    pytest tests/test_integration.py -m "pipeline" -v
    pytest tests/test_integration.py -m "api" -v

All tests carry @pytest.mark.skip so the normal `pytest tests/` run (CI) is
never affected.  Remove the skip decorator to execute a test.

Environment variables required for all tests:
    ERPNEXT_URL, ERPNEXT_API_KEY, ERPNEXT_API_SECRET
    OPENAI_API_KEY
    QDRANT_URL, QDRANT_COLLECTION
    WEBHOOK_SECRET
    ADMIN_SECRET
    BACKEND_URL               (for API-layer tests)
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os

import pytest

# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.integration


# ===========================================================================
# GROUP 1 — Ingestion pipeline
# Covers: erpnext_client → document_parser → chunker → embedder → vector_store
# ===========================================================================


@pytest.mark.skip(reason="requires live ERPNext + OpenAI + Qdrant")
def test_ingest_purchase_order_end_to_end() -> None:
    """
    Fetch a real Purchase Order from ERPNext, serialize it, embed it, upsert
    to Qdrant, then verify the point is retrievable by docname.

    What to assert:
    - po_to_text() produces non-empty text containing the supplier name
    - chunk_text(force_single_chunk=True) returns exactly 1 chunk
    - Embedder.embed_texts() returns a list[list[float]] with length 1 and
      inner length 1536
    - VectorStore.upsert_chunks() completes without error
    - VectorStore.search(embed_query("Purchase Order"), top_k=5) returns at
      least one ScoredPoint whose payload["docname"] matches the PO name
    """
    import asyncio

    from ingestion.chunker import chunk_text
    from ingestion.document_parser import po_to_text
    from ingestion.embedder import Embedder
    from ingestion.erpnext_client import ERPNextClient
    from retrieval.vector_store import VectorStore

    async def run() -> None:
        async with ERPNextClient() as client:
            pos = await client.get_list("Purchase Order", limit=1)
            assert pos, "No Purchase Orders found on the ERPNext site"
            docname = pos[0]["name"]
            po = await client.get_doc("Purchase Order", docname)

            supplier = await client.get_doc("Supplier", po["supplier"])
            supplier_group = supplier.get("supplier_group")

        text = po_to_text(po)
        assert po["supplier"] in text

        chunks = chunk_text(text, force_single_chunk=True)
        assert len(chunks) == 1

        embedder = Embedder()
        vectors = embedder.embed_texts([chunks[0]["text"]])
        assert len(vectors) == 1
        assert len(vectors[0]) == 1536

        vs = VectorStore()
        vs.ensure_collection()
        enriched = [{
            **chunks[0],
            "source_doctype": "Purchase Order",
            "docname": docname,
            "supplier": po.get("supplier"),
            "supplier_group": supplier_group,
            "start_date": po.get("transaction_date"),
            "end_date": po.get("schedule_date"),
            "status": po.get("status"),
            "company": po.get("company"),
            "vector": vectors[0],
        }]
        vs.upsert_chunks(enriched)

        results = vs.search(embedder.embed_query("purchase order"), top_k=5)
        docnames = [r.payload["docname"] for r in results if r.payload]
        assert docname in docnames

    asyncio.run(run())


@pytest.mark.skip(reason="requires live ERPNext + OpenAI + Qdrant")
def test_ingest_contract_chunks_html_correctly() -> None:
    """
    Fetch a real Contract, strip HTML from contract_terms, chunk it, and
    verify each chunk carries the correct metadata.

    What to assert:
    - extract_text_from_html() removes all '<' characters
    - chunk_text() returns >= 1 chunk, all with chunk_index / total_chunks set
    - Every enriched chunk dict has source_doctype == "Contract"
    - After upsert, VectorStore.delete_by_docname() removes all chunks
      (search returns nothing for that docname afterwards)
    """
    ...


@pytest.mark.skip(reason="requires live ERPNext + OpenAI + Qdrant")
def test_idempotent_upsert_does_not_duplicate_points() -> None:
    """
    Ingest the same Purchase Order twice and confirm point count stays the same.

    What to assert:
    - Upsert once → count N points for the docname
    - Upsert again (same doc, same chunk_index) → count is still N
      (deterministic uuid5 IDs overwrite, not append)
    - Use VectorStore.get_all_texts() and filter by docname to count
    """
    ...


@pytest.mark.skip(reason="requires live ERPNext + OpenAI + Qdrant")
def test_supplier_group_enriched_in_po_payload() -> None:
    """
    Confirm that the supplier_group field on the ingested PO payload comes
    from the Supplier record (not from the PO itself, which doesn't have it).

    What to assert:
    - The ERPNext Supplier for the PO has a non-null supplier_group
    - The Qdrant payload for the ingested PO chunk has supplier_group set
      to the same value
    """
    ...


# ===========================================================================
# GROUP 2 — Retrieval layer
# Covers: vector_store + hybrid_search + reranker working together
# ===========================================================================


@pytest.mark.skip(reason="requires live OpenAI + Qdrant (with data already ingested)")
def test_hybrid_search_returns_more_relevant_than_vector_alone() -> None:
    """
    For a query containing an exact term that appears in a known document
    (e.g. a specific PO number), verify that hybrid search surfaces that doc
    in its top-5 while pure vector search may not.

    What to assert:
    - HybridSearch.search("PO-XXXX payment terms") returns a result whose
      docname is "PO-XXXX" in the first 5 positions
    - The BM25 component is demonstrably contributing (manually check that
      the BM25 index was built from get_all_texts())
    """
    ...


@pytest.mark.skip(reason="requires live OpenAI + Qdrant (with data already ingested)")
def test_reranker_with_real_model_orders_by_relevance() -> None:
    """
    Load the cross-encoder model for real (no mock), run rerank() over a set
    of candidates where one is clearly more relevant, and verify ordering.

    What to assert:
    - Reranker.warm_up() completes without error
    - Given candidates ["payment terms net 30 days", "unrelated supplier delivery"],
      rerank("what are the payment terms") ranks the first candidate first
    - The model is not reloaded on a second rerank() call (check _model identity)
    """
    ...


@pytest.mark.skip(reason="requires live OpenAI + Qdrant (with data already ingested)")
def test_metadata_filter_restricts_vector_search_to_supplier() -> None:
    """
    Ingest POs from two different suppliers, then search with a supplier filter
    and verify results only include the filtered supplier.

    What to assert:
    - After ingesting POs for supplier A and supplier B,
      VectorStore.search(vector, filter_conditions={"supplier": "Supplier A"})
      returns only results with payload["supplier"] == "Supplier A"
    """
    ...


@pytest.mark.skip(reason="requires live OpenAI + Qdrant")
def test_bm25_index_rebuilt_after_webhook_upsert() -> None:
    """
    Upsert a new document via the webhook handler, then confirm the BM25 index
    reflects it (a search for a term unique to that document finds it via BM25).

    What to assert:
    - Before upsert: HybridSearch.search("unique-term-XYZ") returns 0 results
    - Trigger the rebuild_bm25 callback (as the webhook handler would)
    - After rebuild: the same search returns >= 1 result containing the new doc
    """
    ...


# ===========================================================================
# GROUP 3 — Full query pipeline (end-to-end RAG)
# Covers: query_rewriter + hybrid_search + reranker + GPT-4o generation
# ===========================================================================


@pytest.mark.skip(reason="requires live OpenAI + Qdrant (with data ingested) + Langfuse")
def test_query_pipeline_returns_answer_with_citations() -> None:
    """
    Run QueryPipeline.run() with a question answerable from the indexed data
    and verify the answer contains docname-style citations.

    What to assert:
    - result["answer"] is a non-empty string
    - result["sources"] is a non-empty list
    - Each source has docname, source_doctype, supplier fields
    - The answer text contains at least one "[docname]" citation pattern
    - No hallucinated docnames appear in citations (all cited docnames exist
      in result["sources"])
    """
    ...


@pytest.mark.skip(reason="requires live OpenAI + Qdrant + Langfuse")
def test_query_pipeline_respects_supplier_filter() -> None:
    """
    Run the pipeline with a supplier filter and verify the answer only
    references that supplier's documents.

    What to assert:
    - With filters={"supplier": "Supplier A"}, sources contain only
      docs where payload["supplier"] == "Supplier A"
    """
    ...


@pytest.mark.skip(reason="requires live OpenAI + Qdrant + Langfuse")
def test_query_pipeline_refuses_to_hallucinate_when_no_context() -> None:
    """
    Ask a question with no matching documents in Qdrant and verify the model
    says it cannot answer rather than hallucinating.

    What to assert:
    - Qdrant returns 0 results for an obscure query
    - result["answer"] contains a phrase like "not found" or "no information"
    - result["sources"] is empty
    """
    ...


@pytest.mark.skip(reason="requires live OpenAI + Qdrant + Langfuse")
def test_hyde_and_step_back_strategies_both_return_answers() -> None:
    """
    Run the pipeline twice — once with QUERY_REWRITE_STRATEGY=hyde and once
    with step_back — and verify both produce non-empty answers.

    What to assert:
    - Both strategies produce result["answer"] with len > 0
    - The HyDE strategy embeds a hypothetical document (check that
      QueryRewriter.rewrite() returns a vector of length 1536)
    - The step_back strategy returns a rewritten string at a higher
      abstraction level than the original question
    """
    ...


# ===========================================================================
# GROUP 4 — API layer
# Covers: FastAPI endpoints hit over HTTP (app must be running)
# ===========================================================================


@pytest.mark.skip(reason="requires running FastAPI server at BACKEND_URL")
def test_health_endpoint_returns_ok() -> None:
    """
    GET /health → {"status": "ok"}
    """
    import httpx

    backend_url = os.environ["BACKEND_URL"]
    response = httpx.get(f"{backend_url}/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.skip(reason="requires running FastAPI server + live Qdrant + OpenAI")
def test_ingest_full_endpoint_triggers_background_ingest() -> None:
    """
    POST /ingest/full with a valid X-Admin-Secret header should return 202
    and eventually populate Qdrant with points.

    What to assert:
    - Response is 202 (accepted, background task queued)
    - After a short wait, VectorStore.get_all_texts() returns > 0 docs
    - POST /ingest/full with a wrong Admin-Secret returns 401
    """
    ...


@pytest.mark.skip(reason="requires running FastAPI server + live ERPNext + Qdrant")
def test_webhook_endpoint_reindexes_document() -> None:
    """
    POST /webhook/erpnext with a valid HMAC signature for a known PO, then
    verify Qdrant was updated.

    What to assert:
    - Response is 200 with {"status": "indexed", "docname": "PO-XXX"}
    - The Qdrant point for PO-XXX is present (search finds it)
    - POST with a bad signature returns 401
    - POST for an unsupported doctype returns {"status": "ignored"}
    """
    import httpx

    backend_url = os.environ["BACKEND_URL"]
    secret = os.environ["WEBHOOK_SECRET"]
    docname = "PO-001"  # replace with a real docname from the test site

    payload = json.dumps({"doctype": "Purchase Order", "docname": docname}).encode()
    signature = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()

    response = httpx.post(
        f"{backend_url}/webhook/erpnext",
        content=payload,
        headers={
            "Content-Type": "application/json",
            "X-Frappe-Webhook-Signature": signature,
        },
    )
    assert response.status_code == 200
    assert response.json()["status"] == "indexed"


@pytest.mark.skip(reason="requires running FastAPI server + live OpenAI + Qdrant")
def test_query_endpoint_returns_answer_and_sources() -> None:
    """
    POST /query with a question answerable from indexed data.

    What to assert:
    - Response is 200
    - Body contains "answer" (non-empty string) and "sources" (list)
    - POST /query with invalid JSON returns 422
    """
    import httpx

    backend_url = os.environ["BACKEND_URL"]
    response = httpx.post(
        f"{backend_url}/query",
        json={"question": "What purchase orders are pending delivery?"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["answer"]
    assert isinstance(body["sources"], list)
