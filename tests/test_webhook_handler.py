"""Tests for ingestion.webhook_handler. No network calls — all I/O is mocked."""

from __future__ import annotations

import hashlib
import hmac
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ingestion.webhook_handler import create_webhook_router

WEBHOOK_SECRET = "test-webhook-secret"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sign(body: bytes, secret: str = WEBHOOK_SECRET) -> str:
    return hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()


def _post(client: TestClient, payload: dict, *, secret: str = WEBHOOK_SECRET) -> object:
    body = json.dumps(payload).encode()
    return client.post(
        "/webhook/erpnext",
        content=body,
        headers={
            "Content-Type": "application/json",
            "X-Frappe-Webhook-Signature": _sign(body, secret),
        },
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_erpnext() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def mock_embedder() -> MagicMock:
    e = MagicMock()
    e.embed_texts.side_effect = lambda texts: [[0.1] * 1536] * len(texts)
    return e


@pytest.fixture
def mock_vector_store() -> MagicMock:
    return MagicMock()


@pytest.fixture
def mock_rebuild() -> MagicMock:
    return MagicMock()


@pytest.fixture
def http_client(
    mock_erpnext: AsyncMock,
    mock_embedder: MagicMock,
    mock_vector_store: MagicMock,
    mock_rebuild: MagicMock,
) -> TestClient:
    app = FastAPI()
    router = create_webhook_router(
        erpnext_client=mock_erpnext,
        embedder=mock_embedder,
        vector_store=mock_vector_store,
        rebuild_bm25=mock_rebuild,
        webhook_secret=WEBHOOK_SECRET,
    )
    app.include_router(router, prefix="/webhook")
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Signature verification
# ---------------------------------------------------------------------------


def test_invalid_signature_returns_401(http_client: TestClient, mock_erpnext: AsyncMock) -> None:
    body = json.dumps({"doctype": "Purchase Order", "docname": "PO-001"}).encode()
    response = http_client.post(
        "/webhook/erpnext",
        content=body,
        headers={
            "Content-Type": "application/json",
            "X-Frappe-Webhook-Signature": "bad-signature",
        },
    )
    assert response.status_code == 401
    mock_erpnext.get_doc.assert_not_called()


def test_missing_signature_header_returns_401(
    http_client: TestClient, mock_erpnext: AsyncMock
) -> None:
    body = json.dumps({"doctype": "Purchase Order", "docname": "PO-001"}).encode()
    response = http_client.post(
        "/webhook/erpnext",
        content=body,
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# Unsupported doctype
# ---------------------------------------------------------------------------


def test_unsupported_doctype_is_ignored(
    http_client: TestClient, mock_erpnext: AsyncMock, mock_vector_store: MagicMock
) -> None:
    response = _post(http_client, {"doctype": "Sales Order", "docname": "SO-001"})
    assert response.status_code == 200
    assert response.json()["status"] == "ignored"
    mock_erpnext.get_doc.assert_not_called()
    mock_vector_store.upsert_chunks.assert_not_called()


# ---------------------------------------------------------------------------
# Purchase Order
# ---------------------------------------------------------------------------

PO_DOC = {
    "name": "PO-001",
    "supplier": "Acme Corp",
    "transaction_date": "2024-01-15",
    "schedule_date": "2024-02-15",
    "grand_total": 10000.0,
    "currency": "USD",
    "items": [{"item_name": "Widget A"}, {"item_name": "Widget B"}],
    "payment_terms_template": "Net 30",
    "status": "To Receive and Bill",
    "company": "My Company",
}

SUPPLIER_DOC = {"name": "Acme Corp", "supplier_group": "Electronics"}


def test_purchase_order_indexed_with_correct_metadata(
    http_client: TestClient,
    mock_erpnext: AsyncMock,
    mock_vector_store: MagicMock,
    mock_rebuild: MagicMock,
) -> None:
    mock_erpnext.get_doc.side_effect = [PO_DOC, SUPPLIER_DOC]

    response = _post(http_client, {"doctype": "Purchase Order", "docname": "PO-001"})

    assert response.status_code == 200
    assert response.json() == {"status": "indexed", "docname": "PO-001"}

    mock_vector_store.delete_by_docname.assert_called_once_with("PO-001")
    mock_vector_store.upsert_chunks.assert_called_once()
    mock_rebuild.assert_called_once()

    chunks = mock_vector_store.upsert_chunks.call_args[0][0]
    assert len(chunks) == 1  # force_single_chunk — PO is one vector
    c = chunks[0]
    assert c["source_doctype"] == "Purchase Order"
    assert c["docname"] == "PO-001"
    assert c["supplier"] == "Acme Corp"
    assert c["supplier_group"] == "Electronics"
    assert c["start_date"] == "2024-01-15"
    assert c["end_date"] == "2024-02-15"
    assert c["status"] == "To Receive and Bill"
    assert c["company"] == "My Company"
    assert "vector" in c


def test_purchase_order_text_contains_key_fields(
    http_client: TestClient, mock_erpnext: AsyncMock, mock_embedder: MagicMock
) -> None:
    mock_erpnext.get_doc.side_effect = [PO_DOC, SUPPLIER_DOC]
    _post(http_client, {"doctype": "Purchase Order", "docname": "PO-001"})

    embedded_texts = mock_embedder.embed_texts.call_args[0][0]
    assert len(embedded_texts) == 1
    text = embedded_texts[0]
    assert "PO-001" in text
    assert "Acme Corp" in text
    assert "Widget A" in text


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------

CONTRACT_DOC = {
    "name": "CON-001",
    "party_name": "BuildRight Ltd",
    "contract_terms": "<p>Clause 1: delivery within 30 days. Clause 2: payment on receipt.</p>",
    "start_date": "2024-01-01",
    "end_date": "2025-01-01",
    "status": "Active",
    "company": "My Company",
}

CONTRACT_SUPPLIER_DOC = {"name": "BuildRight Ltd", "supplier_group": "Construction"}


def test_contract_indexed_with_correct_metadata(
    http_client: TestClient,
    mock_erpnext: AsyncMock,
    mock_vector_store: MagicMock,
) -> None:
    mock_erpnext.get_doc.side_effect = [CONTRACT_DOC, CONTRACT_SUPPLIER_DOC]

    response = _post(http_client, {"doctype": "Contract", "docname": "CON-001"})

    assert response.status_code == 200
    assert response.json() == {"status": "indexed", "docname": "CON-001"}

    chunks = mock_vector_store.upsert_chunks.call_args[0][0]
    c = chunks[0]
    assert c["source_doctype"] == "Contract"
    assert c["supplier"] == "BuildRight Ltd"  # from party_name
    assert c["supplier_group"] == "Construction"
    assert c["start_date"] == "2024-01-01"
    assert c["end_date"] == "2025-01-01"


def test_contract_html_is_stripped_before_embedding(
    http_client: TestClient, mock_erpnext: AsyncMock, mock_embedder: MagicMock
) -> None:
    mock_erpnext.get_doc.side_effect = [CONTRACT_DOC, CONTRACT_SUPPLIER_DOC]
    _post(http_client, {"doctype": "Contract", "docname": "CON-001"})

    texts = mock_embedder.embed_texts.call_args[0][0]
    assert all("<" not in t for t in texts)
    assert any("delivery within 30 days" in t for t in texts)


def test_empty_contract_terms_returns_skipped(
    http_client: TestClient, mock_erpnext: AsyncMock, mock_vector_store: MagicMock
) -> None:
    empty_contract = {**CONTRACT_DOC, "contract_terms": ""}
    mock_erpnext.get_doc.side_effect = [empty_contract, CONTRACT_SUPPLIER_DOC]

    response = _post(http_client, {"doctype": "Contract", "docname": "CON-001"})

    assert response.status_code == 200
    assert response.json()["status"] == "skipped"
    mock_vector_store.upsert_chunks.assert_not_called()


# ---------------------------------------------------------------------------
# Supplier Scorecard
# ---------------------------------------------------------------------------

SCORECARD_DOC = {
    "name": "SSC-001",
    "supplier": "Summit Traders",
    "period": "Per Month",
    "supplier_score": 88.0,
    "indicator_color": "Green",
    "status": "Active",
    "criteria": [{"criteria_name": "Delivery", "score": 18, "max_score": 20}],
}

SCORECARD_SUPPLIER_DOC = {"name": "Summit Traders", "supplier_group": "Wholesale"}


def test_supplier_scorecard_indexed_with_correct_metadata(
    http_client: TestClient,
    mock_erpnext: AsyncMock,
    mock_vector_store: MagicMock,
) -> None:
    mock_erpnext.get_doc.side_effect = [SCORECARD_DOC, SCORECARD_SUPPLIER_DOC]

    response = _post(http_client, {"doctype": "Supplier Scorecard", "docname": "SSC-001"})

    assert response.status_code == 200
    chunks = mock_vector_store.upsert_chunks.call_args[0][0]
    assert len(chunks) == 1  # force_single_chunk
    c = chunks[0]
    assert c["source_doctype"] == "Supplier Scorecard"
    assert c["supplier"] == "Summit Traders"
    assert c["supplier_group"] == "Wholesale"
    assert c["start_date"] is None
    assert c["end_date"] is None
    assert c["company"] is None


# ---------------------------------------------------------------------------
# Supplier group enrichment
# ---------------------------------------------------------------------------


def test_supplier_group_lookup_failure_does_not_block_indexing(
    http_client: TestClient,
    mock_erpnext: AsyncMock,
    mock_vector_store: MagicMock,
) -> None:
    from ingestion.erpnext_client import ERPNextNotFoundError

    mock_erpnext.get_doc.side_effect = [PO_DOC, ERPNextNotFoundError("not found")]

    response = _post(http_client, {"doctype": "Purchase Order", "docname": "PO-001"})

    assert response.status_code == 200
    assert response.json()["status"] == "indexed"
    chunks = mock_vector_store.upsert_chunks.call_args[0][0]
    assert chunks[0]["supplier_group"] is None


# ---------------------------------------------------------------------------
# Operation ordering
# ---------------------------------------------------------------------------


def test_delete_is_called_before_upsert(
    http_client: TestClient,
    mock_erpnext: AsyncMock,
    mock_vector_store: MagicMock,
) -> None:
    mock_erpnext.get_doc.side_effect = [PO_DOC, SUPPLIER_DOC]

    manager = MagicMock()
    manager.attach_mock(mock_vector_store.delete_by_docname, "delete")
    manager.attach_mock(mock_vector_store.upsert_chunks, "upsert")

    _post(http_client, {"doctype": "Purchase Order", "docname": "PO-001"})

    method_names = [c[0] for c in manager.mock_calls]
    assert method_names.index("delete") < method_names.index("upsert")


def test_rebuild_bm25_called_after_upsert(
    http_client: TestClient,
    mock_erpnext: AsyncMock,
    mock_vector_store: MagicMock,
    mock_rebuild: MagicMock,
) -> None:
    mock_erpnext.get_doc.side_effect = [PO_DOC, SUPPLIER_DOC]

    manager = MagicMock()
    manager.attach_mock(mock_vector_store.upsert_chunks, "upsert")
    manager.attach_mock(mock_rebuild, "rebuild")

    _post(http_client, {"doctype": "Purchase Order", "docname": "PO-001"})

    method_names = [c[0] for c in manager.mock_calls]
    assert method_names.index("upsert") < method_names.index("rebuild")
