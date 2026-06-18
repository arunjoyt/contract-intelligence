"""Tests for retrieval.hybrid_search. No network calls — Embedder and VectorStore mocked."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from retrieval.hybrid_search import HybridSearch, _chunk_key, _tokenize

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

QUERY_VECTOR = [0.1] * 1536


def _make_scored_point(payload: dict) -> MagicMock:
    sp = MagicMock()
    sp.payload = payload
    return sp


def _doc(docname: str, chunk_index: int, text: str) -> dict:
    return {"docname": docname, "chunk_index": chunk_index, "total_chunks": 1, "text": text}


# A three-doc corpus gives BM25 enough docs for IDF to produce non-zero scores.
DOC_A = _doc("PO-001", 0, "payment terms net 30 days outstanding invoice")
DOC_B = _doc("CON-001", 0, "contract clause liability indemnification penalty")
DOC_C = _doc("SSC-001", 0, "supplier scorecard delivery quality performance")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_embedder() -> MagicMock:
    e = MagicMock()
    e.embed_query.return_value = QUERY_VECTOR
    return e


@pytest.fixture
def mock_vector_store() -> MagicMock:
    vs = MagicMock()
    vs.search.return_value = []
    return vs


@pytest.fixture
def hs(mock_embedder: MagicMock, mock_vector_store: MagicMock) -> HybridSearch:
    return HybridSearch(embedder=mock_embedder, vector_store=mock_vector_store)


# ---------------------------------------------------------------------------
# build_bm25_index
# ---------------------------------------------------------------------------


def test_build_bm25_index_empty_list_sets_bm25_to_none(hs: HybridSearch) -> None:
    hs.build_bm25_index([])
    assert hs._bm25 is None
    assert hs._corpus_docs == []


def test_build_bm25_index_stores_docs(hs: HybridSearch) -> None:
    hs.build_bm25_index([DOC_A, DOC_B, DOC_C])
    assert hs._corpus_docs == [DOC_A, DOC_B, DOC_C]
    assert hs._bm25 is not None


def test_build_bm25_index_replaces_previous_index(hs: HybridSearch) -> None:
    hs.build_bm25_index([DOC_A, DOC_B, DOC_C])
    first_bm25 = hs._bm25

    hs.build_bm25_index([DOC_C])
    assert hs._bm25 is not first_bm25
    assert len(hs._corpus_docs) == 1


# ---------------------------------------------------------------------------
# search — embedder and vector store wiring
# ---------------------------------------------------------------------------


def test_search_calls_embed_query_with_query_string(
    hs: HybridSearch, mock_embedder: MagicMock
) -> None:
    hs.search("payment terms")
    mock_embedder.embed_query.assert_called_once_with("payment terms")


def test_search_passes_query_vector_to_vector_store(
    hs: HybridSearch, mock_vector_store: MagicMock, mock_embedder: MagicMock
) -> None:
    custom_vector = [0.9] * 1536
    mock_embedder.embed_query.return_value = custom_vector

    hs.search("payment terms")

    call_args = mock_vector_store.search.call_args
    assert call_args[0][0] == custom_vector


def test_search_passes_filter_conditions_to_vector_store(
    hs: HybridSearch, mock_vector_store: MagicMock
) -> None:
    filters = {"supplier": "Acme Corp", "status": "Active"}
    hs.search("payment terms", filter_conditions=filters)

    call_args = mock_vector_store.search.call_args
    assert call_args[0][1] == filters


def test_search_passes_top_k_to_vector_store(
    hs: HybridSearch, mock_vector_store: MagicMock
) -> None:
    hs.search("payment terms", top_k=10)
    call_args = mock_vector_store.search.call_args
    assert call_args[1]["top_k"] == 10


def test_search_passes_none_filter_when_not_provided(
    hs: HybridSearch, mock_vector_store: MagicMock
) -> None:
    hs.search("payment terms")
    call_args = mock_vector_store.search.call_args
    assert call_args[0][1] is None


# ---------------------------------------------------------------------------
# search — RRF fusion
# ---------------------------------------------------------------------------


def test_search_returns_qdrant_results_when_no_bm25_index(
    hs: HybridSearch, mock_vector_store: MagicMock
) -> None:
    # No build_bm25_index call → _bm25 is None
    mock_vector_store.search.return_value = [
        _make_scored_point(DOC_A),
        _make_scored_point(DOC_B),
    ]

    results = hs.search("payment terms")

    assert len(results) == 2
    assert results[0]["docname"] == "PO-001"
    assert results[1]["docname"] == "CON-001"


def test_search_returns_bm25_results_when_qdrant_empty(
    hs: HybridSearch, mock_vector_store: MagicMock
) -> None:
    hs.build_bm25_index([DOC_A, DOC_B, DOC_C])
    mock_vector_store.search.return_value = []  # Qdrant returns nothing

    results = hs.search("payment net 30")

    # BM25 should rank DOC_A first (its text matches the query)
    assert len(results) > 0
    assert results[0]["docname"] == "PO-001"


def test_search_doc_in_both_lists_ranks_above_qdrant_only(
    hs: HybridSearch, mock_vector_store: MagicMock
) -> None:
    """RRF core property: a doc in both lists outranks a doc in only one list."""
    hs.build_bm25_index([DOC_A, DOC_B, DOC_C])

    # Qdrant returns DOC_A at rank 0, DOC_B at rank 1
    mock_vector_store.search.return_value = [
        _make_scored_point(DOC_A),
        _make_scored_point(DOC_B),
    ]

    # Query matches DOC_A via BM25 (has "payment" and "net")
    results = hs.search("payment net 30")

    # DOC_A is in BM25 (rank 0) + Qdrant (rank 0) → higher RRF than DOC_B (Qdrant only)
    assert results[0]["docname"] == "PO-001"


def test_search_doc_only_in_bm25_still_included(
    hs: HybridSearch, mock_vector_store: MagicMock
) -> None:
    """BM25-only hits are included even if Qdrant (with filters) didn't return them."""
    hs.build_bm25_index([DOC_A, DOC_B, DOC_C])

    # Qdrant returns only DOC_B (e.g. due to a supplier filter)
    mock_vector_store.search.return_value = [_make_scored_point(DOC_B)]

    # Query matches DOC_A via BM25
    results = hs.search("payment net 30")

    docnames = [r["docname"] for r in results]
    assert "PO-001" in docnames  # from BM25
    assert "CON-001" in docnames  # from Qdrant


def test_search_respects_top_k_limit(
    hs: HybridSearch, mock_vector_store: MagicMock
) -> None:
    many_docs = [_doc(f"PO-{i:03d}", 0, f"unique term word{i} procurement") for i in range(30)]
    hs.build_bm25_index(many_docs)
    mock_vector_store.search.return_value = [_make_scored_point(d) for d in many_docs[:30]]

    results = hs.search("procurement", top_k=5)

    assert len(results) <= 5


def test_search_multi_chunk_doc_each_chunk_ranked_independently(
    hs: HybridSearch, mock_vector_store: MagicMock
) -> None:
    chunk0 = _doc("CON-001", 0, "payment terms net 30 days contract")
    chunk1 = _doc("CON-001", 1, "liability clause indemnification penalty")
    filler = _doc("PO-999", 0, "unrelated supplier delivery schedule")

    hs.build_bm25_index([chunk0, chunk1, filler])
    mock_vector_store.search.return_value = []

    results = hs.search("payment net contract")

    # Both chunks are separate entries; chunk0 should rank above chunk1
    keys = [_chunk_key(r) for r in results]
    assert "CON-001:0" in keys
    assert keys.index("CON-001:0") < keys.index("CON-001:1")


def test_search_rrf_scores_decrease_with_rank(
    hs: HybridSearch, mock_vector_store: MagicMock
) -> None:
    """Qdrant rank-0 doc should beat rank-1 doc when neither is in BM25."""
    # No BM25 index → pure Qdrant RRF (rank 0 > rank 1 > rank 2)
    mock_vector_store.search.return_value = [
        _make_scored_point(DOC_A),
        _make_scored_point(DOC_B),
        _make_scored_point(DOC_C),
    ]

    results = hs.search("anything")

    assert results[0]["docname"] == "PO-001"
    assert results[1]["docname"] == "CON-001"
    assert results[2]["docname"] == "SSC-001"


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_tokenize_lowercases_and_splits() -> None:
    assert _tokenize("Payment Terms NET 30") == ["payment", "terms", "net", "30"]


def test_tokenize_empty_string() -> None:
    assert _tokenize("") == []


def test_chunk_key_format() -> None:
    assert _chunk_key({"docname": "PO-001", "chunk_index": 3}) == "PO-001:3"


def test_chunk_key_missing_fields_uses_defaults() -> None:
    assert _chunk_key({}) == ":0"
