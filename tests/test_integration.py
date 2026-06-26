"""Integration tests — require live external services.

Skipped in CI automatically; run manually with:

    RUN_INTEGRATION=1 pytest tests/test_integration.py -v
    RUN_INTEGRATION=1 pytest tests/test_integration.py -m ingestion -v
    RUN_INTEGRATION=1 pytest tests/test_integration.py -m retrieval -v
    RUN_INTEGRATION=1 pytest tests/test_integration.py -m pipeline -v
    RUN_INTEGRATION=1 pytest tests/test_integration.py -m api -v
    RUN_INTEGRATION=1 pytest tests/test_integration.py -m evaluation -v
    RUN_INTEGRATION=1 pytest tests/test_integration.py -m langfuse -v
    RUN_INTEGRATION=1 pytest tests/test_integration.py -m auth -v

Services needed for Groups 1-3:
    ERPNext  — http://127.0.0.1:8005  (credentials in .env)
    Qdrant   — http://localhost:6333
    OpenAI   — OPENAI_API_KEY in .env

Group 4 tests the API layer (Step 12 — api/main.py) via the containerised app.
Services needed: Docker app (localhost:8000) + Qdrant (localhost:6333) + ERPNext + OpenAI.

Group 5 tests the evaluation script (Step 14 — evaluation/evaluate.py).
Services needed: Qdrant + OpenAI (ERPNext not required — test dataset is self-contained).

Group 7 tests Langfuse observability end-to-end (see docs/ARCHITECTURE.md § Observability).
Services needed: Qdrant + OpenAI + a live Langfuse server (LANGFUSE_PUBLIC_KEY,
LANGFUSE_SECRET_KEY, LANGFUSE_HOST in .env).

A dedicated Qdrant collection (procurement_integration_test) is created at
session start and deleted on teardown — the production collection is untouched.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta

import pytest
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Skip helpers
# ---------------------------------------------------------------------------

_run = os.getenv("RUN_INTEGRATION")

live = pytest.mark.skipif(not _run, reason="set RUN_INTEGRATION=1 to run")


# ---------------------------------------------------------------------------
# Auth helper — mint a short-lived JWT signed with the env JWT_SECRET.
# Used by groups that hit /query on a running Docker stack.
# ---------------------------------------------------------------------------


def _integration_auth_headers() -> dict[str, str]:
    """Return Authorization header using the JWT_SECRET from .env."""
    import jwt

    secret = os.environ.get("JWT_SECRET", "")
    if not secret:
        pytest.skip("JWT_SECRET not set in .env — needed to mint test JWTs")
    token = jwt.encode(
        {
            "sub": "integration-test",
            "roles": ["System Manager"],
            "exp": datetime.now(tz=UTC) + timedelta(hours=1),
        },
        secret,
        algorithm="HS256",
    )
    return {"Authorization": f"Bearer {token}"}

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
    import contextlib
    with contextlib.suppress(Exception):
        store._client.delete_collection(TEST_COLLECTION)


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
        {"docname": "PO-001", "chunk_index": 0,
         "text": "payment terms net 30 days outstanding invoice supplier"},
        {"docname": "CON-001", "chunk_index": 0,
         "text": "delivery schedule lead time logistics warehouse"},
        {"docname": "SSC-001", "chunk_index": 0,
         "text": "supplier scorecard quality rating performance criteria"},
    ]
    results = r.rerank("what are the payment terms", candidates, top_n=3)

    assert len(results) == 3
    assert results[0]["docname"] == "PO-001"  # payment-terms doc must rank first

    # Model not reloaded on second call
    model_before = r._model
    r.rerank("delivery schedule", candidates, top_n=2)
    assert r._model is model_before


# ===========================================================================
# GROUP 3 helpers
# ===========================================================================


def _make_pipeline(embedder, vs, strategy: str = "hyde"):
    """Build a fully-wired QueryPipeline against the current test collection state.

    BM25 index is rebuilt from whatever is in ``vs`` at call time, so call
    this *after* any setup upserts to ensure lexical search is current.
    """
    from pipeline.query_pipeline import QueryPipeline
    from pipeline.query_rewriter import QueryRewriter
    from retrieval.hybrid_search import HybridSearch
    from retrieval.reranker import Reranker

    hs = HybridSearch(embedder=embedder, vector_store=vs)
    hs.build_bm25_index(vs.get_all_texts())
    reranker = Reranker()
    rewriter = QueryRewriter(embedder=embedder, strategy=strategy)
    return QueryPipeline(rewriter=rewriter, hybrid_search=hs, reranker=reranker)


# ===========================================================================
# GROUP 3 — Full query pipeline
# ===========================================================================


@live
@pytest.mark.pipeline
async def test_query_pipeline_returns_answer_with_citations(embedder, vs) -> None:
    """Full chain: HyDE rewrite → embed → hybrid search → rerank → GPT-4o generation.

    Verifies the answer is non-empty, contains at least one [docname] citation,
    and the sources list is populated with matching SourceDoc objects.
    """
    import re

    docname = await _first_po_name()
    payload = await _ingest_po(docname, embedder, vs)
    supplier = payload["supplier"]

    pipeline = _make_pipeline(embedder, vs)
    result = pipeline.run(f"What purchase orders have been issued to {supplier}?")

    assert result["answer"], "Expected a non-empty answer"
    citations = re.findall(r"\[([^\]]+)\]", result["answer"])
    assert citations, f"Expected at least one [docname] citation; got: {result['answer']!r}"
    assert result["sources"], "Expected at least one SourceDoc in sources"


@live
@pytest.mark.pipeline
async def test_query_pipeline_respects_supplier_filter(embedder, vs) -> None:
    """Explicit supplier filter reaches Qdrant and surfaces target-supplier documents.

    The filter applies to the vector-search side of HybridSearch; BM25 is
    intentionally unfiltered (it has no metadata support).  So the guarantee
    is that at least one source from the target supplier appears in the results
    — not that *all* sources are from that supplier.
    """
    from ingestion.erpnext_client import ERPNextClient

    async with ERPNextClient() as client:
        pos = await client.get_list("Purchase Order", limit=5)

    if len(pos) < 2:
        pytest.skip("Need at least 2 Purchase Orders to test supplier isolation")

    p1 = await _ingest_po(pos[0]["name"], embedder, vs)
    p2 = await _ingest_po(pos[1]["name"], embedder, vs)

    if p1["supplier"] == p2["supplier"]:
        pytest.skip("Both POs share the same supplier — seed data with two distinct suppliers")

    target_supplier = p1["supplier"]
    pipeline = _make_pipeline(embedder, vs)
    result = pipeline.run(
        f"Show me purchase orders from {target_supplier}",
        filters={"supplier": target_supplier},
    )

    # BM25 is unfiltered by design, so the result set may include docs from
    # other suppliers.  The Qdrant filter is verified by confirming the target
    # supplier's documents are present (i.e. the filter did not exclude them).
    supplier_names = {s.supplier for s in result["sources"]}
    assert target_supplier in supplier_names, (
        f"Expected at least one source from {target_supplier!r}; "
        f"got suppliers: {supplier_names}"
    )


@live
@pytest.mark.pipeline
async def test_query_pipeline_refuses_to_hallucinate_when_no_context(embedder, vs) -> None:
    """GPT-4o must not fabricate an answer when the context has nothing relevant.

    The test collection contains procurement documents.  A question with zero
    overlap with those documents should trigger the exact fallback phrase from
    the system prompt rather than a hallucinated answer.
    """
    # Ensure there is *something* indexed so the pipeline actually retrieves
    # chunks — we are testing the "irrelevant context" branch, not "empty collection".
    await _ingest_po(await _first_po_name(), embedder, vs)

    pipeline = _make_pipeline(embedder, vs)
    result = pipeline.run(
        "What is the chemical formula for water and how is it used in nuclear reactors?"
    )

    assert "could not find" in result["answer"].lower(), (
        f"Expected the fallback 'could not find' phrase, got: {result['answer']!r}"
    )


@live
@pytest.mark.pipeline
async def test_hyde_and_step_back_strategies_both_return_answers(embedder, vs) -> None:
    """Both QUERY_REWRITE_STRATEGY values produce a non-empty answer and sources list.

    Verifies the strategy env-var wiring and that the step-back prompt path
    reaches GPT-4o generation without error.
    """
    await _ingest_po(await _first_po_name(), embedder, vs)

    question = "What are the payment terms in the purchase orders?"

    for strategy in ("hyde", "step_back"):
        pipeline = _make_pipeline(embedder, vs, strategy=strategy)
        result = pipeline.run(question)
        assert result["answer"], f"Strategy {strategy!r} returned an empty answer"
        assert isinstance(result["sources"], list), (
            f"Strategy {strategy!r} did not return a sources list"
        )


# ===========================================================================
# GROUP 4 — API layer (Step 12 — api/main.py)
# Services: Docker app (localhost:8000) + Qdrant (localhost:6333) + ERPNext + OpenAI
# All tests use plain httpx against the running Docker stack — no TestClient.
# ===========================================================================

_G4_APP_URL = os.getenv("API_URL", "http://localhost:8000")
_G4_QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
_G4_COLLECTION = os.getenv("QDRANT_COLLECTION", "procurement")


def _g4_collection_count() -> int:
    """Return the point count for the production Qdrant collection, or 0 on error."""
    import httpx

    try:
        r = httpx.get(f"{_G4_QDRANT_URL}/collections/{_G4_COLLECTION}", timeout=5)
        if r.status_code == 200:
            return r.json().get("result", {}).get("points_count", 0)
    except Exception:
        pass
    return 0


@pytest.fixture(scope="module")
def api_http():
    """Skip the group if the containerised app is not reachable."""
    import httpx

    try:
        httpx.get(f"{_G4_APP_URL}/health", timeout=3).raise_for_status()
    except Exception:
        pytest.skip("App not running — start with: docker compose up -d")
    return _G4_APP_URL


@live
@pytest.mark.api
def test_health_endpoint_returns_ok(api_http) -> None:
    import httpx

    r = httpx.get(f"{api_http}/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


@live
@pytest.mark.api
def test_ingest_full_endpoint_triggers_background_ingest(api_http) -> None:
    """POST /ingest/full returns 202; data appears in Qdrant after polling."""
    import time

    import httpx

    admin_secret = os.getenv("ADMIN_SECRET", "")
    assert admin_secret, "ADMIN_SECRET must be set in .env"

    r = httpx.post(
        f"{api_http}/ingest/full",
        headers={"X-Admin-Secret": admin_secret},
        timeout=10,
    )
    assert r.status_code == 202
    assert r.json() == {"status": "accepted"}

    # Poll until the background ingest lands data in the production collection
    deadline = time.time() + 120
    while time.time() < deadline:
        if _g4_collection_count() > 0:
            return
        time.sleep(5)
    pytest.fail(
        f"No points in Qdrant collection '{_G4_COLLECTION}' after 120 s — "
        "check that ERPNEXT_URL is routable from inside the Docker container"
    )


@live
@pytest.mark.api
def test_webhook_endpoint_reindexes_document(api_http) -> None:
    """POST /webhook/erpnext re-indexes a real PO from ERPNext."""
    import hashlib
    import hmac
    import json

    import httpx

    erpnext_url = os.getenv("ERPNEXT_URL", "")
    api_key = os.getenv("ERPNEXT_API_KEY", "")
    api_secret = os.getenv("ERPNEXT_API_SECRET", "")
    if not all([erpnext_url, api_key, api_secret]):
        pytest.skip("ERPNEXT_URL / ERPNEXT_API_KEY / ERPNEXT_API_SECRET not set in .env")

    resp = httpx.get(
        f"{erpnext_url}/api/resource/Purchase Order",
        params={"limit_page_length": 1, "fields": '["name"]'},
        headers={"Authorization": f"token {api_key}:{api_secret}"},
        timeout=10,
    )
    resp.raise_for_status()
    pos = resp.json().get("data", [])
    assert pos, "Need at least one Purchase Order on the ERPNext site"

    docname = pos[0]["name"]
    webhook_secret = os.getenv("WEBHOOK_SECRET", "")
    assert webhook_secret, "WEBHOOK_SECRET must be set in .env"

    body = json.dumps({"doctype": "Purchase Order", "docname": docname}).encode()
    sig = hmac.new(webhook_secret.encode(), body, hashlib.sha256).hexdigest()

    r = httpx.post(
        f"{api_http}/webhook/erpnext",
        content=body,
        headers={
            "Content-Type": "application/json",
            "X-Frappe-Webhook-Signature": sig,
        },
        timeout=60,
    )
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "indexed"
    assert data["docname"] == docname


@live
@pytest.mark.api
def test_query_endpoint_returns_answer_and_sources(api_http) -> None:
    """POST /query returns a grounded answer with [docname] citations."""
    import re

    import httpx

    r = httpx.post(
        f"{api_http}/query",
        json={"question": "What purchase orders are active?"},
        headers=_integration_auth_headers(),
        timeout=60,
    )
    assert r.status_code == 200
    data = r.json()
    assert data["answer"], "Expected a non-empty answer"

    citations = re.findall(r"\[([^\]]+)\]", data["answer"])
    assert citations or "could not find" in data["answer"].lower(), (
        f"Expected citations or fallback phrase; got: {data['answer']!r}"
    )
    assert isinstance(data["sources"], list)


# ===========================================================================
# GROUP 5 — Evaluation script (Step 14 — evaluation/evaluate.py)
# Services: Qdrant + OpenAI  (ERPNext not required — dataset is self-contained)
# ===========================================================================


@pytest.fixture(scope="session")
def eval_results(tmp_path_factory, vs):
    """Run evaluate.py once per session and return the output path.

    Depends on ``vs`` only to guarantee the test Qdrant collection exists.
    No manual seeding is needed: evaluate.py falls back to ground-truth
    contexts when the collection is empty, so the script still produces a
    valid results.json in all cases.
    """
    from pathlib import Path

    from evaluation.evaluate import evaluate as run_evaluate

    dataset_path = Path("evaluation/test_dataset.json")
    output_path = tmp_path_factory.mktemp("eval") / "results.json"

    original = os.environ.get("QDRANT_COLLECTION")
    os.environ["QDRANT_COLLECTION"] = TEST_COLLECTION
    try:
        run_evaluate(dataset_path, output_path)
    finally:
        if original is None:
            os.environ.pop("QDRANT_COLLECTION", None)
        else:
            os.environ["QDRANT_COLLECTION"] = original

    return output_path


@live
@pytest.mark.evaluation
def test_evaluate_writes_results_json(eval_results) -> None:
    """evaluate.py must produce a non-empty results.json at the requested path."""
    assert eval_results.exists(), "evaluate.py did not write results.json"
    assert eval_results.stat().st_size > 0, "results.json is empty"


@live
@pytest.mark.evaluation
def test_evaluate_results_json_has_required_keys(eval_results) -> None:
    """results.json must contain the expected top-level structure."""
    import json

    data = json.loads(eval_results.read_text())
    for key in ("timestamp", "num_questions", "metrics", "per_question"):
        assert key in data, f"Missing top-level key '{key}' in results.json"


@live
@pytest.mark.evaluation
def test_evaluate_metrics_are_in_valid_range(eval_results) -> None:
    """All four RAGAS metrics must be floats in [0.0, 1.0]."""
    import json

    metrics = json.loads(eval_results.read_text())["metrics"]
    expected = {"faithfulness", "answer_relevancy", "context_recall", "context_precision"}

    assert set(metrics.keys()) == expected, (
        f"Expected metric keys {expected}, got {set(metrics.keys())}"
    )
    for name, value in metrics.items():
        assert isinstance(value, float), f"Metric '{name}' is not a float: {value!r}"
        assert 0.0 <= value <= 1.0, f"Metric '{name}' = {value} is outside [0.0, 1.0]"


@live
@pytest.mark.evaluation
def test_evaluate_per_question_count_matches_dataset(eval_results) -> None:
    """per_question length must match num_questions, and at least one question answered."""
    import json
    from pathlib import Path

    data = json.loads(eval_results.read_text())
    dataset_len = len(json.loads(Path("evaluation/test_dataset.json").read_text()))

    assert 0 < data["num_questions"] <= dataset_len, (
        f"num_questions={data['num_questions']} not in (0, {dataset_len}]"
    )
    assert len(data["per_question"]) == data["num_questions"], (
        "per_question list length does not match num_questions"
    )


@live
@pytest.mark.evaluation
def test_evaluate_per_question_entries_have_required_fields(eval_results) -> None:
    """Every per_question entry must have non-empty question and answer fields."""
    import json

    for i, entry in enumerate(json.loads(eval_results.read_text())["per_question"]):
        assert "question" in entry, f"Entry {i} missing 'question'"
        assert "answer" in entry, f"Entry {i} missing 'answer'"
        assert "retrieved_context_count" in entry, f"Entry {i} missing 'retrieved_context_count'"
        assert entry["answer"], f"Entry {i} has an empty answer"


# ---------------------------------------------------------------------------
# Group 6 — Docker Compose full-stack
# ---------------------------------------------------------------------------

_DOCKER_APP_URL = "http://localhost:8000"
_DOCKER_FRONTEND_URL = "http://localhost:8501"


@pytest.fixture(scope="module")
def docker_app():
    """Skip the group if the containerised app is not reachable."""
    import httpx

    try:
        httpx.get(f"{_DOCKER_APP_URL}/health", timeout=3).raise_for_status()
    except Exception:
        pytest.skip("Docker stack not running — start with: docker compose up -d")
    return _DOCKER_APP_URL


@live
@pytest.mark.docker
def test_docker_health_returns_ok(docker_app) -> None:
    import httpx

    r = httpx.get(f"{docker_app}/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


@live
@pytest.mark.docker
def test_docker_query_returns_answer_and_sources(docker_app) -> None:
    import httpx

    r = httpx.post(
        f"{docker_app}/query",
        json={"question": "What purchase orders are active?"},
        headers=_integration_auth_headers(),
        timeout=60,
    )
    assert r.status_code == 200
    data = r.json()
    assert "answer" in data and isinstance(data["answer"], str) and data["answer"]
    assert "sources" in data and isinstance(data["sources"], list)


@live
@pytest.mark.docker
def test_docker_ingest_full_rejects_missing_secret(docker_app) -> None:
    import httpx

    r = httpx.post(f"{docker_app}/ingest/full")
    assert r.status_code == 403


@live
@pytest.mark.docker
def test_docker_ingest_full_accepts_valid_secret(docker_app) -> None:
    import httpx

    admin_secret = os.getenv("ADMIN_SECRET", "")
    if not admin_secret:
        pytest.skip("ADMIN_SECRET not set in .env")
    r = httpx.post(f"{docker_app}/ingest/full", headers={"X-Admin-Secret": admin_secret})
    # Background ingest is accepted; full completion depends on ERPNext being
    # reachable from inside the container (ERPNEXT_URL must be a routable host,
    # not 127.0.0.1, when running via docker compose).
    assert r.status_code == 202
    assert r.json() == {"status": "accepted"}


@live
@pytest.mark.docker
def test_docker_frontend_returns_200() -> None:
    import httpx

    try:
        r = httpx.get(_DOCKER_FRONTEND_URL, timeout=10)
    except Exception:
        pytest.skip(
            "Frontend not running — start with: "
            "docker compose -f docker-compose.yml -f docker-compose.frontend.yml up -d"
        )
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# Group 7 — Langfuse observability (pure HTTP — targets containerised stack)
# ---------------------------------------------------------------------------
# All tests POST to the running app container (localhost:8000) and verify
# traces via the Langfuse REST API (localhost:3000).  No local SDK or
# qdrant_client import is needed.

_APP_HOST = os.getenv("API_URL", "http://localhost:8000")
_LF_HOST = os.getenv("LANGFUSE_HOST", "http://localhost:3000")
_LF_PUBLIC = os.getenv("LANGFUSE_PUBLIC_KEY", "")
_LF_SECRET = os.getenv("LANGFUSE_SECRET_KEY", "")

_EXPECTED_SPANS = {"rewrite", "filter_extraction", "hybrid_search", "rerank", "generate"}
_LF_QUESTION = "What are the payment terms for our active contracts?"


def _lf_headers() -> dict:
    import base64

    creds = base64.b64encode(f"{_LF_PUBLIC}:{_LF_SECRET}".encode()).decode()
    return {"Authorization": f"Basic {creds}"}


def _fetch_trace(trace_id: str) -> dict:
    import time

    import httpx

    for _ in range(8):
        try:
            r = httpx.get(
                f"{_LF_HOST}/api/public/traces/{trace_id}",
                headers=_lf_headers(),
                timeout=10,
            )
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        time.sleep(1)
    pytest.fail(f"Trace {trace_id} not found in Langfuse after retries")


def _fetch_observations(trace_id: str) -> list[dict]:
    import httpx

    r = httpx.get(
        f"{_LF_HOST}/api/public/observations",
        params={"traceId": trace_id},
        headers=_lf_headers(),
        timeout=10,
    )
    r.raise_for_status()
    return r.json().get("data", [])


@pytest.fixture(scope="module")
def lf_trace_result():
    """POST a query to the containerised app and find the resulting trace in Langfuse.

    Skips the whole module if either service is not reachable.
    Returns (trace_id, question, api_result).
    """
    import time

    import httpx

    if not _LF_PUBLIC or not _LF_SECRET:
        pytest.skip(
            "LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set in .env"
        )

    for url, label in [
        (f"{_APP_HOST}/health", "App (localhost:8000)"),
        (f"{_LF_HOST}/api/public/health", "Langfuse (localhost:3000)"),
    ]:
        try:
            httpx.get(url, timeout=3).raise_for_status()
        except Exception:
            pytest.skip(f"{label} not running — start with: docker compose up -d")

    r = httpx.post(
        f"{_APP_HOST}/query",
        json={"question": _LF_QUESTION},
        headers=_integration_auth_headers(),
        timeout=60,
    )
    r.raise_for_status()
    result = r.json()

    # Allow Langfuse time to ingest the event
    time.sleep(4)

    # Find the trace by matching the question in the input field
    r2 = httpx.get(
        f"{_LF_HOST}/api/public/traces",
        params={"name": "query", "limit": 20},
        headers=_lf_headers(),
        timeout=10,
    )
    r2.raise_for_status()
    traces = r2.json().get("data", [])
    matching = [
        t for t in traces
        if t.get("input", {}).get("question") == _LF_QUESTION
    ]
    if not matching:
        pytest.fail(f"No Langfuse trace found for: {_LF_QUESTION!r}")

    return matching[0]["id"], _LF_QUESTION, result


@live
@pytest.mark.langfuse
def test_langfuse_trace_created_for_query(lf_trace_result) -> None:
    trace_id, question, _ = lf_trace_result
    trace = _fetch_trace(trace_id)
    assert trace["name"] == "query"
    assert trace.get("input", {}).get("question") == question


@live
@pytest.mark.langfuse
def test_langfuse_trace_has_all_five_spans(lf_trace_result) -> None:
    trace_id, _, _ = lf_trace_result
    obs = _fetch_observations(trace_id)
    span_names = {o["name"] for o in obs}
    missing = _EXPECTED_SPANS - span_names
    assert not missing, f"Missing spans in trace: {missing}"


@live
@pytest.mark.langfuse
def test_langfuse_trace_output_contains_answer_and_source_count(lf_trace_result) -> None:
    trace_id, _, result = lf_trace_result
    trace = _fetch_trace(trace_id)
    output = trace.get("output") or {}
    assert "answer" in output, "Trace output missing 'answer'"
    assert "source_count" in output, "Trace output missing 'source_count'"
    assert isinstance(output["source_count"], int) and output["source_count"] >= 0


@live
@pytest.mark.langfuse
def test_langfuse_generate_span_output_matches_api_response(lf_trace_result) -> None:
    """The 'generate' span's output should contain the same answer the API returned."""
    trace_id, _, api_result = lf_trace_result
    obs = _fetch_observations(trace_id)
    gen_span = next((o for o in obs if o["name"] == "generate"), None)
    assert gen_span is not None, "'generate' span not found in observations"
    api_answer = api_result.get("answer", "")
    assert api_answer, "API returned empty answer"
    span_output = gen_span.get("output") or {}
    span_answer = (
        span_output.get("answer", span_output)
        if isinstance(span_output, dict)
        else str(span_output)
    )
    assert api_answer in str(span_answer) or str(span_answer) in api_answer, (
        f"'generate' span output does not match API response.\n"
        f"  API:  {api_answer[:120]!r}\n"
        f"  Span: {str(span_answer)[:120]!r}"
    )


@live
@pytest.mark.langfuse
def test_langfuse_each_query_gets_distinct_trace(lf_trace_result) -> None:
    """A second, different query must create a new distinct trace in Langfuse."""
    import time

    import httpx

    second_question = "List our top suppliers by purchase volume."
    r = httpx.post(
        f"{_APP_HOST}/query",
        json={"question": second_question},
        headers=_integration_auth_headers(),
        timeout=60,
    )
    r.raise_for_status()
    time.sleep(4)

    r2 = httpx.get(
        f"{_LF_HOST}/api/public/traces",
        params={"name": "query", "limit": 20},
        headers=_lf_headers(),
        timeout=10,
    )
    r2.raise_for_status()
    traces = r2.json().get("data", [])

    first_id, _, _ = lf_trace_result
    second_matches = [
        t for t in traces
        if t.get("input", {}).get("question") == second_question
    ]
    assert second_matches, f"No trace found for second query: {second_question!r}"
    assert second_matches[0]["id"] != first_id, "Second query reused the same trace ID"


# ===========================================================================
# GROUP 8 — Auth (Option B: ERPNext OAuth2 + JWT)
# Services: Docker app (localhost:8000)
# Run: RUN_INTEGRATION=1 pytest tests/test_integration.py -m auth -v
#
# Note: the full OAuth browser flow (login → ERPNext → callback → JWT) requires
# a configured ERPNext OAuth client and is covered by manual verification.
# These tests cover the API-level auth surface that can be exercised without a
# live browser session.
# ===========================================================================


@pytest.fixture(scope="module")
def auth_app():
    """Skip the group if the containerised app is not reachable."""
    import httpx

    try:
        httpx.get(f"{_DOCKER_APP_URL}/health", timeout=3).raise_for_status()
    except Exception:
        pytest.skip("Docker stack not running — start with: docker compose up -d")
    return _DOCKER_APP_URL


@live
@pytest.mark.auth
def test_auth_query_without_token_returns_401(auth_app) -> None:
    import httpx

    r = httpx.post(
        f"{auth_app}/query",
        json={"question": "What POs exist?"},
    )
    assert r.status_code == 401


@live
@pytest.mark.auth
def test_auth_query_with_invalid_token_returns_401(auth_app) -> None:
    import httpx

    r = httpx.post(
        f"{auth_app}/query",
        json={"question": "What POs exist?"},
        headers={"Authorization": "Bearer not-a-valid-jwt"},
    )
    assert r.status_code == 401


@live
@pytest.mark.auth
def test_auth_query_with_disallowed_role_returns_403(auth_app) -> None:
    import jwt

    secret = os.environ.get("JWT_SECRET", "")
    if not secret:
        pytest.skip("JWT_SECRET not set in .env")

    import httpx

    bad_token = jwt.encode(
        {
            "sub": "test-user",
            "roles": ["Sales User"],
            "exp": datetime.now(tz=UTC) + timedelta(hours=1),
        },
        secret,
        algorithm="HS256",
    )
    r = httpx.post(
        f"{auth_app}/query",
        json={"question": "What POs exist?"},
        headers={"Authorization": f"Bearer {bad_token}"},
    )
    assert r.status_code == 403


@live
@pytest.mark.auth
def test_auth_login_redirects_to_erpnext_authorize(auth_app) -> None:
    """GET /auth/login must redirect to the ERPNext OAuth2 authorize endpoint."""
    import httpx

    r = httpx.get(f"{auth_app}/auth/login", follow_redirects=False)
    assert r.status_code in (302, 307)
    location = r.headers.get("location", "")
    assert "frappe.integrations.oauth2.authorize" in location
    assert "client_id=" in location
    assert "code_challenge=" in location


@live
@pytest.mark.auth
def test_auth_callback_with_invalid_state_returns_400(auth_app) -> None:
    """GET /auth/callback with an unknown state must return 400."""
    import httpx

    r = httpx.get(
        f"{auth_app}/auth/callback",
        params={"code": "somecode", "state": "nonexistent-state"},
    )
    assert r.status_code == 400


@live
@pytest.mark.auth
def test_auth_query_with_valid_jwt_returns_200(auth_app) -> None:
    """POST /query with a valid signed JWT must reach the pipeline (not be rejected by auth)."""
    import httpx

    r = httpx.post(
        f"{auth_app}/query",
        json={"question": "What purchase orders are active?"},
        headers=_integration_auth_headers(),
        timeout=60,
    )
    # 200 means auth passed; the pipeline may return any valid response
    assert r.status_code == 200
    data = r.json()
    assert "answer" in data
    assert "sources" in data
