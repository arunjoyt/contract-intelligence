"""Integration tests — require live external services.

Skipped in CI automatically; run manually with:

    RUN_INTEGRATION=1 pytest tests/test_integration.py -v
    RUN_INTEGRATION=1 pytest tests/test_integration.py -m ingestion -v
    RUN_INTEGRATION=1 pytest tests/test_integration.py -m retrieval -v

Services needed for Groups 1 & 2:
    ERPNext  — http://127.0.0.1:8005  (credentials in .env)
    Qdrant   — http://localhost:6333
    OpenAI   — OPENAI_API_KEY in .env

Groups 3 & 4 remain as stubs until pipeline/api modules are built (Steps 10-12).

A dedicated Qdrant collection (procurement_integration_test) is created at
session start and deleted on teardown — the production collection is untouched.
"""

from __future__ import annotations

import asyncio
import os

import pytest
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Skip helpers
# ---------------------------------------------------------------------------

_run = os.getenv("RUN_INTEGRATION")

live = pytest.mark.skipif(not _run, reason="set RUN_INTEGRATION=1 to run")
needs_pipeline = pytest.mark.skip(reason="pipeline/api not yet built (Steps 10-12)")

# ---------------------------------------------------------------------------
# Shared test collection — isolated from production data
# ---------------------------------------------------------------------------

TEST_COLLECTION = "procurement_integration_test"

# ---------------------------------------------------------------------------
# Session-scoped fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def embedder():
    from ingestion.embedder import Embedder
    return Embedder()


@pytest.fixture(scope="session")
def vs(embedder):
    from retrieval.vector_store import VectorStore
    store = VectorStore(collection=TEST_COLLECTION)
    store.ensure_collection()
    yield store
    try:
        store._client.delete_collection(TEST_COLLECTION)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _first_po_name() -> str:
    from ingestion.erpnext_client import ERPNextClient
    async with ERPNextClient() as client:
        pos = await client.get_list("Purchase Order", limit=1)
    assert pos, "No Purchase Orders on the ERPNext site"
    return pos[0]["name"]


async def _ingest_po(docname: str, embedder, vs) -> dict:
    """Full PO ingestion; returns the enriched payload dict that was upserted."""
    from ingestion.chunker import chunk_text
    from ingestion.document_parser import po_to_text
    from ingestion.erpnext_client import ERPNextClient

    async with ERPNextClient() as client:
        po = await client.get_doc("Purchase Order", docname)
        supplier = await client.get_doc("Supplier", po["supplier"])

    text = po_to_text(po)
    chunks = chunk_text(text, force_single_chunk=True)
    vectors = embedder.embed_texts([c["text"] for c in chunks])

    payload = {
        **chunks[0],
        "source_doctype": "Purchase Order",
        "docname": docname,
        "supplier": po.get("supplier"),
        "supplier_group": supplier.get("supplier_group"),
        "start_date": po.get("transaction_date"),
        "end_date": po.get("schedule_date"),
        "status": po.get("status"),
        "company": po.get("company"),
        "vector": vectors[0],
    }
    vs.upsert_chunks([payload])
    return payload


# ===========================================================================
# GROUP 1 — ERPNext client connectivity
# ===========================================================================


@live
@pytest.mark.ingestion
async def test_erpnext_client_lists_purchase_orders() -> None:
    from ingestion.erpnext_client import ERPNextClient

    async with ERPNextClient() as client:
        pos = await client.get_list("Purchase Order", limit=5)

    assert len(pos) > 0
    assert all("name" in p for p in pos)


@live
@pytest.mark.ingestion
async def test_erpnext_client_fetches_full_po_doc() -> None:
    from ingestion.erpnext_client import ERPNextClient

    async with ERPNextClient() as client:
        pos = await client.get_list("Purchase Order", limit=1)
        po = await client.get_doc("Purchase Order", pos[0]["name"])

    assert po["name"] == pos[0]["name"]
    assert "supplier" in po
    assert "items" in po


@live
@pytest.mark.ingestion
async def test_erpnext_client_fetches_supplier_doc() -> None:
    """supplier_group lives on the Supplier record, not on Purchase Order."""
    from ingestion.erpnext_client import ERPNextClient

    async with ERPNextClient() as client:
        pos = await client.get_list("Purchase Order", limit=1)
        po = await client.get_doc("Purchase Order", pos[0]["name"])
        supplier = await client.get_doc("Supplier", po["supplier"])

    assert "supplier_group" in supplier
    assert supplier["supplier_group"]  # must be non-null on demo data


@live
@pytest.mark.ingestion
async def test_erpnext_client_raises_not_found_for_bad_docname() -> None:
    from ingestion.erpnext_client import ERPNextClient, ERPNextNotFoundError

    async with ERPNextClient() as client:
        with pytest.raises(ERPNextNotFoundError):
            await client.get_doc("Purchase Order", "DOES-NOT-EXIST-99999")


# ===========================================================================
# GROUP 1 — Document parser against real ERPNext data
# ===========================================================================


@live
@pytest.mark.ingestion
async def test_po_to_text_contains_supplier_and_status() -> None:
    from ingestion.document_parser import po_to_text
    from ingestion.erpnext_client import ERPNextClient

    async with ERPNextClient() as client:
        pos = await client.get_list("Purchase Order", limit=1)
        po = await client.get_doc("Purchase Order", pos[0]["name"])

    text = po_to_text(po)
    assert po["supplier"] in text
    assert po["name"] in text
    assert po.get("status", "") in text
    assert "<" not in text  # no raw HTML


@live
@pytest.mark.ingestion
async def test_contract_html_stripped_to_plain_text() -> None:
    from ingestion.document_parser import extract_text_from_html
    from ingestion.erpnext_client import ERPNextClient

    async with ERPNextClient() as client:
        contracts = await client.get_list("Contract", limit=1)

    if not contracts:
        pytest.skip("No Contracts on the ERPNext site — seed one per Step 0")

    async with ERPNextClient() as client:
        contract = await client.get_doc("Contract", contracts[0]["name"])

    raw_html = contract.get("contract_terms", "")
    if not raw_html:
        pytest.skip("Contract has no contract_terms HTML")

    text = extract_text_from_html(raw_html)
    assert "<" not in text
    assert len(text) > 0


# ===========================================================================
# GROUP 1 — Full ingestion pipeline (ERPNext → Qdrant)
# ===========================================================================


@live
@pytest.mark.ingestion
async def test_ingest_purchase_order_end_to_end(embedder, vs) -> None:
    docname = await _first_po_name()
    payload = await _ingest_po(docname, embedder, vs)

    # Embedding dimensions
    assert len(payload["vector"]) == 1536

    # Searchable in Qdrant
    query_vector = embedder.embed_query("purchase order supplier payment terms")
    results = vs.search(query_vector, top_k=10)
    retrieved = [r.payload["docname"] for r in results if r.payload]
    assert docname in retrieved


@live
@pytest.mark.ingestion
async def test_ingest_contract_end_to_end(embedder, vs) -> None:
    from ingestion.chunker import chunk_text
    from ingestion.document_parser import extract_text_from_html
    from ingestion.erpnext_client import ERPNextClient

    async with ERPNextClient() as client:
        contracts = await client.get_list("Contract", limit=1)

    if not contracts:
        pytest.skip("No Contracts on the ERPNext site — seed one per Step 0")

    docname = contracts[0]["name"]

    async with ERPNextClient() as client:
        contract = await client.get_doc("Contract", docname)
        supplier_name = contract.get("party_name")
        supplier = await client.get_doc("Supplier", supplier_name) if supplier_name else {}

    text = extract_text_from_html(contract.get("contract_terms", ""))
    if not text.strip():
        pytest.skip("Contract has no extractable text")

    chunks = chunk_text(text)
    assert len(chunks) >= 1
    assert all("chunk_index" in c and "total_chunks" in c for c in chunks)

    vectors = embedder.embed_texts([c["text"] for c in chunks])
    assert len(vectors) == len(chunks)

    enriched = [
        {
            **chunk,
            "source_doctype": "Contract",
            "docname": docname,
            "supplier": supplier_name,
            "supplier_group": supplier.get("supplier_group"),
            "start_date": contract.get("start_date"),
            "end_date": contract.get("end_date"),
            "status": contract.get("status"),
            "company": contract.get("company"),
            "vector": vector,
        }
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]
    vs.upsert_chunks(enriched)

    # All chunks must carry source_doctype
    assert all(c["source_doctype"] == "Contract" for c in enriched)

    # Cleanup: delete and confirm gone
    vs.delete_by_docname(docname)
    query_vector = embedder.embed_query(text[:100])
    results = vs.search(query_vector, filter_conditions={"docname": docname}, top_k=5)
    assert len(results) == 0


@live
@pytest.mark.ingestion
async def test_idempotent_upsert_does_not_duplicate_points(embedder, vs) -> None:
    docname = await _first_po_name()

    await _ingest_po(docname, embedder, vs)
    await _ingest_po(docname, embedder, vs)  # second upsert — same IDs

    all_docs = vs.get_all_texts()
    count = sum(1 for d in all_docs if d.get("docname") == docname)
    # PO is force_single_chunk → exactly 1 point regardless of how many times ingested
    assert count == 1


@live
@pytest.mark.ingestion
async def test_supplier_group_enriched_in_po_payload(embedder, vs) -> None:
    from ingestion.erpnext_client import ERPNextClient

    async with ERPNextClient() as client:
        pos = await client.get_list("Purchase Order", limit=1)
        po = await client.get_doc("Purchase Order", pos[0]["name"])
        supplier = await client.get_doc("Supplier", po["supplier"])

    expected_group = supplier.get("supplier_group")
    assert expected_group, "Supplier on demo data must have supplier_group set"

    docname = po["name"]
    await _ingest_po(docname, embedder, vs)

    all_docs = vs.get_all_texts()
    payload = next((d for d in all_docs if d.get("docname") == docname), None)
    assert payload is not None
    assert payload["supplier_group"] == expected_group


# ===========================================================================
# GROUP 2 — Retrieval layer
# ===========================================================================


@live
@pytest.mark.retrieval
async def test_vector_store_metadata_filter_restricts_to_supplier(embedder, vs) -> None:
    from ingestion.erpnext_client import ERPNextClient

    # Collect POs across (potentially) multiple suppliers
    async with ERPNextClient() as client:
        pos = await client.get_list("Purchase Order", limit=5)

    if len(pos) < 2:
        pytest.skip("Need at least 2 Purchase Orders for filter test")

    # Ingest first two POs
    p1 = await _ingest_po(pos[0]["name"], embedder, vs)
    p2 = await _ingest_po(pos[1]["name"], embedder, vs)

    if p1["supplier"] == p2["supplier"]:
        pytest.skip("Both POs share the same supplier — need two different suppliers")

    query_vector = embedder.embed_query("purchase order")
    results = vs.search(
        query_vector,
        filter_conditions={"supplier": p1["supplier"]},
        top_k=10,
    )

    assert all(r.payload["supplier"] == p1["supplier"] for r in results if r.payload)


@live
@pytest.mark.retrieval
async def test_hybrid_search_bm25_surfaces_exact_docname(embedder, vs) -> None:
    from retrieval.hybrid_search import HybridSearch

    docname = await _first_po_name()
    await _ingest_po(docname, embedder, vs)

    hs = HybridSearch(embedder=embedder, vector_store=vs)
    hs.build_bm25_index(vs.get_all_texts())

    # The docname is an exact term — BM25 should pick it up even if the
    # vector alone might not rank it first
    results = hs.search(docname, top_k=5)
    retrieved = [r.get("docname") for r in results]
    assert docname in retrieved


@live
@pytest.mark.retrieval
async def test_hybrid_search_bm25_rebuilt_after_new_upsert(embedder, vs) -> None:
    from ingestion.erpnext_client import ERPNextClient
    from retrieval.hybrid_search import HybridSearch

    hs = HybridSearch(embedder=embedder, vector_store=vs)

    # Build index with current state
    hs.build_bm25_index(vs.get_all_texts())
    before_count = len(hs._corpus_docs)

    # Ingest a second PO (may already be there — idempotent)
    async with ERPNextClient() as client:
        pos = await client.get_list("Purchase Order", limit=2)
    second_docname = pos[-1]["name"]
    await _ingest_po(second_docname, embedder, vs)

    # Rebuild — corpus must grow (or stay same if already ingested)
    hs.build_bm25_index(vs.get_all_texts())
    assert len(hs._corpus_docs) >= before_count


@live
@pytest.mark.retrieval
def test_reranker_real_model_orders_candidates_by_relevance() -> None:
    """Load the actual cross-encoder; no Qdrant or ERPNext needed."""
    from retrieval.reranker import Reranker

    r = Reranker()
    r.warm_up()  # downloads model if not cached

    candidates = [
        {"docname": "PO-001", "chunk_index": 0, "text": "payment terms net 30 days outstanding invoice supplier"},
        {"docname": "CON-001", "chunk_index": 0, "text": "delivery schedule lead time logistics warehouse"},
        {"docname": "SSC-001", "chunk_index": 0, "text": "supplier scorecard quality rating performance criteria"},
    ]
    results = r.rerank("what are the payment terms", candidates, top_n=3)

    assert len(results) == 3
    assert results[0]["docname"] == "PO-001"  # payment-terms doc must rank first

    # Model not reloaded on second call
    model_before = r._model
    r.rerank("delivery schedule", candidates, top_n=2)
    assert r._model is model_before


# ===========================================================================
# GROUP 3 — Full query pipeline  (stubs — Steps 10-11 not yet built)
# ===========================================================================


@needs_pipeline
def test_query_pipeline_returns_answer_with_citations() -> None: ...


@needs_pipeline
def test_query_pipeline_respects_supplier_filter() -> None: ...


@needs_pipeline
def test_query_pipeline_refuses_to_hallucinate_when_no_context() -> None: ...


@needs_pipeline
def test_hyde_and_step_back_strategies_both_return_answers() -> None: ...


# ===========================================================================
# GROUP 4 — API layer  (stubs — Step 12 not yet built)
# ===========================================================================


@needs_pipeline
def test_health_endpoint_returns_ok() -> None: ...


@needs_pipeline
def test_ingest_full_endpoint_triggers_background_ingest() -> None: ...


@needs_pipeline
def test_webhook_endpoint_reindexes_document() -> None: ...


@needs_pipeline
def test_query_endpoint_returns_answer_and_sources() -> None: ...
