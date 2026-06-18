"""Tests for retrieval.vector_store. No network calls — QdrantClient is mocked."""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, call

import pytest

from retrieval.vector_store import VECTOR_DIM, VectorStore


@pytest.fixture
def mock_qdrant_client(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    client = MagicMock()
    monkeypatch.setattr("retrieval.vector_store.QdrantClient", MagicMock(return_value=client))
    return client


@pytest.fixture
def store(mock_qdrant_client: MagicMock) -> VectorStore:
    return VectorStore(url="http://localhost:6333", api_key=None, collection="test_col")


# ---------------------------------------------------------------------------
# ensure_collection
# ---------------------------------------------------------------------------


def test_ensure_collection_creates_when_missing(
    store: VectorStore, mock_qdrant_client: MagicMock
) -> None:
    mock_qdrant_client.get_collections.return_value.collections = []

    store.ensure_collection()

    mock_qdrant_client.create_collection.assert_called_once()
    kwargs = mock_qdrant_client.create_collection.call_args.kwargs
    assert kwargs["collection_name"] == "test_col"
    assert kwargs["vectors_config"].size == VECTOR_DIM


def test_ensure_collection_skips_when_exists(
    store: VectorStore, mock_qdrant_client: MagicMock
) -> None:
    existing = MagicMock()
    existing.name = "test_col"
    mock_qdrant_client.get_collections.return_value.collections = [existing]

    store.ensure_collection()

    mock_qdrant_client.create_collection.assert_not_called()


# ---------------------------------------------------------------------------
# upsert_chunks
# ---------------------------------------------------------------------------


def test_upsert_chunks_generates_deterministic_ids(
    store: VectorStore, mock_qdrant_client: MagicMock
) -> None:
    chunk = {
        "docname": "PO-001",
        "chunk_index": 0,
        "total_chunks": 1,
        "text": "some text",
        "source_doctype": "Purchase Order",
        "vector": [0.1] * 1536,
    }

    store.upsert_chunks([chunk])

    points = mock_qdrant_client.upsert.call_args.kwargs["points"]
    expected_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, "PO-001:0"))
    assert points[0].id == expected_id


def test_upsert_chunks_excludes_vector_from_payload(
    store: VectorStore, mock_qdrant_client: MagicMock
) -> None:
    chunk = {
        "docname": "PO-001",
        "chunk_index": 0,
        "total_chunks": 1,
        "text": "some text",
        "vector": [0.1] * 1536,
    }

    store.upsert_chunks([chunk])

    points = mock_qdrant_client.upsert.call_args.kwargs["points"]
    assert "vector" not in points[0].payload
    assert points[0].payload["text"] == "some text"
    assert points[0].payload["docname"] == "PO-001"


def test_upsert_chunks_empty_list_is_noop(
    store: VectorStore, mock_qdrant_client: MagicMock
) -> None:
    store.upsert_chunks([])
    mock_qdrant_client.upsert.assert_not_called()


def test_upsert_chunks_multiple_chunks_same_doc(
    store: VectorStore, mock_qdrant_client: MagicMock
) -> None:
    chunks = [
        {"docname": "CON-001", "chunk_index": i, "total_chunks": 3, "text": f"t{i}", "vector": [float(i)] * 1536}
        for i in range(3)
    ]

    store.upsert_chunks(chunks)

    points = mock_qdrant_client.upsert.call_args.kwargs["points"]
    assert len(points) == 3
    ids = {p.id for p in points}
    assert len(ids) == 3  # all distinct


def test_upsert_chunks_same_docname_chunk_produces_same_id(
    store: VectorStore, mock_qdrant_client: MagicMock
) -> None:
    chunk = {"docname": "PO-999", "chunk_index": 2, "total_chunks": 5, "text": "x", "vector": [0.0] * 1536}

    store.upsert_chunks([chunk])
    first_id = mock_qdrant_client.upsert.call_args.kwargs["points"][0].id

    mock_qdrant_client.reset_mock()
    store.upsert_chunks([chunk])
    second_id = mock_qdrant_client.upsert.call_args.kwargs["points"][0].id

    assert first_id == second_id


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


def test_search_calls_qdrant_with_vector_and_limit(
    store: VectorStore, mock_qdrant_client: MagicMock
) -> None:
    mock_qdrant_client.search.return_value = []
    vec = [0.5] * 1536

    store.search(vec, top_k=10)

    mock_qdrant_client.search.assert_called_once()
    kwargs = mock_qdrant_client.search.call_args.kwargs
    assert kwargs["query_vector"] == vec
    assert kwargs["limit"] == 10
    assert kwargs["collection_name"] == "test_col"


def test_search_passes_none_filter_when_no_conditions(
    store: VectorStore, mock_qdrant_client: MagicMock
) -> None:
    mock_qdrant_client.search.return_value = []

    store.search([0.1] * 1536)

    kwargs = mock_qdrant_client.search.call_args.kwargs
    assert kwargs["query_filter"] is None


def test_search_passes_none_filter_for_empty_conditions(
    store: VectorStore, mock_qdrant_client: MagicMock
) -> None:
    mock_qdrant_client.search.return_value = []

    store.search([0.1] * 1536, filter_conditions={})

    kwargs = mock_qdrant_client.search.call_args.kwargs
    assert kwargs["query_filter"] is None


def test_search_builds_filter_from_conditions(
    store: VectorStore, mock_qdrant_client: MagicMock
) -> None:
    mock_qdrant_client.search.return_value = []

    store.search([0.1] * 1536, filter_conditions={"supplier": "Acme Corp"})

    kwargs = mock_qdrant_client.search.call_args.kwargs
    qdrant_filter = kwargs["query_filter"]
    assert qdrant_filter is not None
    assert qdrant_filter.must[0].key == "supplier"
    assert qdrant_filter.must[0].match.value == "Acme Corp"


def test_search_skips_none_valued_conditions(
    store: VectorStore, mock_qdrant_client: MagicMock
) -> None:
    mock_qdrant_client.search.return_value = []

    store.search([0.1] * 1536, filter_conditions={"supplier": None, "status": "Active"})

    kwargs = mock_qdrant_client.search.call_args.kwargs
    qdrant_filter = kwargs["query_filter"]
    assert len(qdrant_filter.must) == 1
    assert qdrant_filter.must[0].key == "status"


def test_search_returns_qdrant_results(
    store: VectorStore, mock_qdrant_client: MagicMock
) -> None:
    fake_point = MagicMock()
    mock_qdrant_client.search.return_value = [fake_point]

    result = store.search([0.1] * 1536)

    assert result == [fake_point]


# ---------------------------------------------------------------------------
# delete_by_docname
# ---------------------------------------------------------------------------


def test_delete_by_docname_calls_delete_with_filter(
    store: VectorStore, mock_qdrant_client: MagicMock
) -> None:
    store.delete_by_docname("PO-001")

    mock_qdrant_client.delete.assert_called_once()
    kwargs = mock_qdrant_client.delete.call_args.kwargs
    assert kwargs["collection_name"] == "test_col"
    selector = kwargs["points_selector"]
    assert selector.filter.must[0].key == "docname"
    assert selector.filter.must[0].match.value == "PO-001"


# ---------------------------------------------------------------------------
# get_all_texts
# ---------------------------------------------------------------------------


def _make_record(payload: dict) -> MagicMock:
    r = MagicMock()
    r.payload = payload
    return r


def test_get_all_texts_returns_payloads_from_single_page(
    store: VectorStore, mock_qdrant_client: MagicMock
) -> None:
    payloads = [{"text": "a", "docname": "PO-1"}, {"text": "b", "docname": "PO-2"}]
    mock_qdrant_client.scroll.return_value = ([_make_record(p) for p in payloads], None)

    result = store.get_all_texts()

    assert result == payloads
    assert mock_qdrant_client.scroll.call_count == 1


def test_get_all_texts_paginates_until_no_offset(
    store: VectorStore, mock_qdrant_client: MagicMock
) -> None:
    page1 = [_make_record({"text": "a", "docname": "PO-1"})]
    page2 = [_make_record({"text": "b", "docname": "PO-2"})]
    mock_qdrant_client.scroll.side_effect = [
        (page1, "offset-token"),
        (page2, None),
    ]

    result = store.get_all_texts()

    assert len(result) == 2
    assert mock_qdrant_client.scroll.call_count == 2
    # Second call must pass the offset token
    second_call_kwargs = mock_qdrant_client.scroll.call_args_list[1].kwargs
    assert second_call_kwargs["offset"] == "offset-token"


def test_get_all_texts_skips_records_with_no_payload(
    store: VectorStore, mock_qdrant_client: MagicMock
) -> None:
    r_with = _make_record({"text": "a"})
    r_without = MagicMock()
    r_without.payload = None
    mock_qdrant_client.scroll.return_value = ([r_with, r_without], None)

    result = store.get_all_texts()

    assert result == [{"text": "a"}]


# ---------------------------------------------------------------------------
# constructor
# ---------------------------------------------------------------------------


def test_constructor_reads_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QDRANT_URL", "http://qdrant.example.com")
    monkeypatch.setenv("QDRANT_COLLECTION", "my_col")
    mock_cls = MagicMock(return_value=MagicMock())
    monkeypatch.setattr("retrieval.vector_store.QdrantClient", mock_cls)

    vs = VectorStore()

    assert vs._collection == "my_col"
    mock_cls.assert_called_once_with(url="http://qdrant.example.com", api_key=None)
