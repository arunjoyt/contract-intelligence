"""Tests for ingestion.webhook_handler. No network calls — all I/O is mocked."""

from __future__ import annotations

import base64
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
    return base64.b64encode(
        hmac.new(secret.encode("utf-8"), body, hashlib.sha256).digest()
    ).decode()


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
    client = AsyncMock()
    client.get_attached_files.return_value = []
    return client


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


def test_contract_linked_document_captured_in_metadata_and_text(
    http_client: TestClient,
    mock_erpnext: AsyncMock,
    mock_vector_store: MagicMock,
    mock_embedder: MagicMock,
) -> None:
    linked_contract = {
        **CONTRACT_DOC,
        "document_type": "Purchase Order",
        "document_name": "PO-2024-00123",
    }
    mock_erpnext.get_doc.side_effect = [linked_contract, CONTRACT_SUPPLIER_DOC]

    response = _post(http_client, {"doctype": "Contract", "docname": "CON-001"})

    assert response.status_code == 200

    chunks = mock_vector_store.upsert_chunks.call_args[0][0]
    c = chunks[0]
    assert c["linked_doctype"] == "Purchase Order"
    assert c["linked_docname"] == "PO-2024-00123"

    texts = mock_embedder.embed_texts.call_args[0][0]
    assert any("Purchase Order PO-2024-00123" in t for t in texts)


def test_contract_without_linked_document_has_null_metadata(
    http_client: TestClient,
    mock_erpnext: AsyncMock,
    mock_vector_store: MagicMock,
) -> None:
    mock_erpnext.get_doc.side_effect = [CONTRACT_DOC, CONTRACT_SUPPLIER_DOC]

    response = _post(http_client, {"doctype": "Contract", "docname": "CON-001"})

    assert response.status_code == 200

    chunks = mock_vector_store.upsert_chunks.call_args[0][0]
    c = chunks[0]
    assert c["linked_doctype"] is None
    assert c["linked_docname"] is None


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
# Purchase Invoice
# ---------------------------------------------------------------------------

INVOICE_DOC = {
    "name": "PINV-001",
    "supplier": "Acme Corp",
    "supplier_group": "Electronics",  # direct field, no Supplier join needed
    "posting_date": "2024-03-01",
    "due_date": "2024-03-31",
    "grand_total": 5000.0,
    "outstanding_amount": 5000.0,
    "currency": "USD",
    "items": [{"item_name": "Widget A"}],
    "payment_terms_template": "Net 30",
    "status": "Unpaid",
    "company": "My Company",
}


def test_purchase_invoice_indexed_with_correct_metadata(
    http_client: TestClient,
    mock_erpnext: AsyncMock,
    mock_vector_store: MagicMock,
) -> None:
    mock_erpnext.get_doc.side_effect = [INVOICE_DOC]

    response = _post(http_client, {"doctype": "Purchase Invoice", "docname": "PINV-001"})

    assert response.status_code == 200
    assert response.json() == {"status": "indexed", "docname": "PINV-001"}

    # supplier_group comes straight off the invoice doc — no Supplier lookup
    mock_erpnext.get_doc.assert_called_once_with("Purchase Invoice", "PINV-001")

    chunks = mock_vector_store.upsert_chunks.call_args[0][0]
    assert len(chunks) == 1  # force_single_chunk
    c = chunks[0]
    assert c["source_doctype"] == "Purchase Invoice"
    assert c["docname"] == "PINV-001"
    assert c["supplier"] == "Acme Corp"
    assert c["supplier_group"] == "Electronics"
    assert c["start_date"] == "2024-03-01"
    assert c["end_date"] == "2024-03-31"
    assert c["status"] == "Unpaid"
    assert c["company"] == "My Company"


def test_purchase_invoice_text_contains_key_fields(
    http_client: TestClient, mock_erpnext: AsyncMock, mock_embedder: MagicMock
) -> None:
    mock_erpnext.get_doc.side_effect = [INVOICE_DOC]
    _post(http_client, {"doctype": "Purchase Invoice", "docname": "PINV-001"})

    text = mock_embedder.embed_texts.call_args[0][0][0]
    assert "PINV-001" in text
    assert "Acme Corp" in text
    assert "Widget A" in text


# ---------------------------------------------------------------------------
# Terms and Conditions
# ---------------------------------------------------------------------------

TERMS_DOC = {
    "name": "Standard Terms",
    "title": "Standard Terms",
    "terms": "<p>Payment due within 30 days of delivery.</p>",
    "disabled": 0,
}


def test_terms_and_conditions_indexed_with_correct_metadata(
    http_client: TestClient,
    mock_erpnext: AsyncMock,
    mock_vector_store: MagicMock,
) -> None:
    mock_erpnext.get_doc.side_effect = [TERMS_DOC]

    response = _post(
        http_client, {"doctype": "Terms and Conditions", "docname": "Standard Terms"}
    )

    assert response.status_code == 200
    assert response.json() == {"status": "indexed", "docname": "Standard Terms"}

    # no supplier/party_name field on this doctype — no Supplier lookup performed
    mock_erpnext.get_doc.assert_called_once_with("Terms and Conditions", "Standard Terms")

    chunks = mock_vector_store.upsert_chunks.call_args[0][0]
    c = chunks[0]
    assert c["source_doctype"] == "Terms and Conditions"
    assert c["supplier"] is None
    assert c["supplier_group"] is None
    assert c["status"] == "Active"


def test_disabled_terms_and_conditions_get_disabled_status(
    http_client: TestClient, mock_erpnext: AsyncMock, mock_vector_store: MagicMock
) -> None:
    mock_erpnext.get_doc.side_effect = [{**TERMS_DOC, "disabled": 1}]

    _post(http_client, {"doctype": "Terms and Conditions", "docname": "Standard Terms"})

    chunks = mock_vector_store.upsert_chunks.call_args[0][0]
    assert chunks[0]["status"] == "Disabled"


def test_terms_and_conditions_html_is_stripped_before_embedding(
    http_client: TestClient, mock_erpnext: AsyncMock, mock_embedder: MagicMock
) -> None:
    mock_erpnext.get_doc.side_effect = [TERMS_DOC]
    _post(http_client, {"doctype": "Terms and Conditions", "docname": "Standard Terms"})

    texts = mock_embedder.embed_texts.call_args[0][0]
    assert all("<" not in t for t in texts)
    assert any("Payment due within 30 days" in t for t in texts)


# ---------------------------------------------------------------------------
# PDF attachments (Purchase Order, Contract)
# ---------------------------------------------------------------------------


def test_po_with_pdf_attachment_indexes_extra_chunk(
    http_client: TestClient,
    mock_erpnext: AsyncMock,
    mock_vector_store: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_erpnext.get_doc.side_effect = [PO_DOC, SUPPLIER_DOC]
    mock_erpnext.get_attached_files.return_value = [
        {"name": "FILE-001", "file_url": "/private/files/po-001.pdf", "file_name": "po-001.pdf"}
    ]
    mock_erpnext.get_file_content.return_value = b"%PDF-fake-bytes"
    monkeypatch.setattr(
        "ingestion.webhook_handler.extract_text_from_pdf",
        lambda _bytes: "Attachment clause: delivery within 10 days.",
    )

    response = _post(http_client, {"doctype": "Purchase Order", "docname": "PO-001"})

    assert response.status_code == 200
    mock_erpnext.get_attached_files.assert_called_once_with("Purchase Order", "PO-001")
    mock_erpnext.get_file_content.assert_called_once_with("/private/files/po-001.pdf")

    chunks = mock_vector_store.upsert_chunks.call_args[0][0]
    # PO's own serialized text (1 chunk, force_single) + 1 PDF-derived chunk
    assert len(chunks) == 2
    assert all(c["total_chunks"] == 2 for c in chunks)
    assert {c["chunk_index"] for c in chunks} == {0, 1}
    texts = [c["text"] for c in chunks]
    assert any("Attachment clause" in t for t in texts)
    # attachment chunks still carry the parent PO's metadata (same docname)
    assert all(c["docname"] == "PO-001" for c in chunks)
    assert all(c["source_doctype"] == "Purchase Order" for c in chunks)


def test_non_pdf_attachments_are_skipped(
    http_client: TestClient,
    mock_erpnext: AsyncMock,
    mock_vector_store: MagicMock,
) -> None:
    mock_erpnext.get_doc.side_effect = [PO_DOC, SUPPLIER_DOC]
    mock_erpnext.get_attached_files.return_value = [
        {"name": "FILE-001", "file_url": "/private/files/photo.png", "file_name": "photo.png"}
    ]

    _post(http_client, {"doctype": "Purchase Order", "docname": "PO-001"})

    mock_erpnext.get_file_content.assert_not_called()
    chunks = mock_vector_store.upsert_chunks.call_args[0][0]
    assert len(chunks) == 1  # only the PO's own text


def test_attachment_fetch_failure_does_not_block_indexing(
    http_client: TestClient,
    mock_erpnext: AsyncMock,
    mock_vector_store: MagicMock,
) -> None:
    mock_erpnext.get_doc.side_effect = [PO_DOC, SUPPLIER_DOC]
    mock_erpnext.get_attached_files.side_effect = RuntimeError("ERPNext unreachable")

    response = _post(http_client, {"doctype": "Purchase Order", "docname": "PO-001"})

    assert response.status_code == 200
    assert response.json()["status"] == "indexed"
    chunks = mock_vector_store.upsert_chunks.call_args[0][0]
    assert len(chunks) == 1  # PO's own text still indexed


def test_pdf_extraction_failure_for_one_file_does_not_block_others(
    http_client: TestClient,
    mock_erpnext: AsyncMock,
    mock_vector_store: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_erpnext.get_doc.side_effect = [PO_DOC, SUPPLIER_DOC]
    mock_erpnext.get_attached_files.return_value = [
        {"name": "FILE-001", "file_url": "/private/files/bad.pdf", "file_name": "bad.pdf"},
        {"name": "FILE-002", "file_url": "/private/files/good.pdf", "file_name": "good.pdf"},
    ]
    mock_erpnext.get_file_content.side_effect = [
        RuntimeError("corrupt download"),
        b"%PDF-fake-bytes",
    ]
    monkeypatch.setattr(
        "ingestion.webhook_handler.extract_text_from_pdf",
        lambda _bytes: "Good PDF text.",
    )

    response = _post(http_client, {"doctype": "Purchase Order", "docname": "PO-001"})

    assert response.status_code == 200
    chunks = mock_vector_store.upsert_chunks.call_args[0][0]
    texts = [c["text"] for c in chunks]
    assert any("Good PDF text" in t for t in texts)
    assert len(chunks) == 2  # PO text + the one PDF that succeeded


def test_attachments_not_fetched_for_non_attachment_doctypes(
    http_client: TestClient,
    mock_erpnext: AsyncMock,
) -> None:
    mock_erpnext.get_doc.side_effect = [SCORECARD_DOC, SCORECARD_SUPPLIER_DOC]

    _post(http_client, {"doctype": "Supplier Scorecard", "docname": "SSC-001"})

    mock_erpnext.get_attached_files.assert_not_called()


def test_contract_with_empty_terms_but_pdf_attachment_is_not_skipped(
    http_client: TestClient,
    mock_erpnext: AsyncMock,
    mock_vector_store: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    empty_contract = {**CONTRACT_DOC, "contract_terms": ""}
    mock_erpnext.get_doc.side_effect = [empty_contract, CONTRACT_SUPPLIER_DOC]
    mock_erpnext.get_attached_files.return_value = [
        {"name": "FILE-001", "file_url": "/private/files/con-001.pdf", "file_name": "con-001.pdf"}
    ]
    mock_erpnext.get_file_content.return_value = b"%PDF-fake-bytes"
    monkeypatch.setattr(
        "ingestion.webhook_handler.extract_text_from_pdf",
        lambda _bytes: "Contract PDF body text.",
    )

    response = _post(http_client, {"doctype": "Contract", "docname": "CON-001"})

    assert response.status_code == 200
    assert response.json()["status"] == "indexed"
    chunks = mock_vector_store.upsert_chunks.call_args[0][0]
    assert len(chunks) == 1
    assert "Contract PDF body text." in chunks[0]["text"]


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


# ---------------------------------------------------------------------------
# Frappe full-doc payload — primary key sent as "name", not "docname"
# ---------------------------------------------------------------------------


def test_purchase_order_indexed_via_name_field(
    http_client: TestClient,
    mock_erpnext: AsyncMock,
    mock_vector_store: MagicMock,
) -> None:
    """Frappe's full-doc webhook format sends the PK as 'name', not 'docname'.

    webhook_handler.py falls back to payload.get('name') when 'docname' is absent.
    A missing or empty 'name' would cause a silent empty-string lookup and a skipped
    or errored index — this test ensures the fallback path is exercised.
    """
    mock_erpnext.get_doc.side_effect = [PO_DOC, SUPPLIER_DOC]

    body = json.dumps({"doctype": "Purchase Order", "name": "PO-001"}).encode()
    response = http_client.post(
        "/webhook/erpnext",
        content=body,
        headers={
            "Content-Type": "application/json",
            "X-Frappe-Webhook-Signature": _sign(body),
        },
    )

    assert response.status_code == 200
    assert response.json() == {"status": "indexed", "docname": "PO-001"}
    mock_vector_store.upsert_chunks.assert_called_once()


# ---------------------------------------------------------------------------
# Delete-then-upsert failure — leaves document absent from index
# ---------------------------------------------------------------------------


def test_upsert_failure_after_delete_propagates_error(
    http_client: TestClient,
    mock_erpnext: AsyncMock,
    mock_vector_store: MagicMock,
) -> None:
    """If upsert_chunks raises after delete_by_docname, the document is gone from the
    index with no rollback.  The handler must surface the error (5xx) rather than
    silently returning 'indexed' while leaving the index in a broken state.
    """
    mock_erpnext.get_doc.side_effect = [PO_DOC, SUPPLIER_DOC]
    mock_vector_store.upsert_chunks.side_effect = RuntimeError("Qdrant connection refused")

    response = _post(http_client, {"doctype": "Purchase Order", "docname": "PO-001"})

    # delete ran — document is now absent from the index
    mock_vector_store.delete_by_docname.assert_called_once_with("PO-001")
    # error must propagate as 5xx, not be swallowed as a 200 "indexed"
    assert response.status_code == 500
