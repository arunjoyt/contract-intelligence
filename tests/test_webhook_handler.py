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
    e.embed_texts_with_usage.side_effect = lambda texts: (
        [[0.1] * 1536] * len(texts),
        {"input": 7 * len(texts), "total": 7 * len(texts)},
    )
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


@pytest.fixture
def mock_langfuse() -> MagicMock:
    lf = MagicMock()
    lf.trace.return_value = MagicMock()
    return lf


@pytest.fixture
def traced_http_client(
    mock_erpnext: AsyncMock,
    mock_embedder: MagicMock,
    mock_vector_store: MagicMock,
    mock_rebuild: MagicMock,
    mock_langfuse: MagicMock,
) -> TestClient:
    app = FastAPI()
    router = create_webhook_router(
        erpnext_client=mock_erpnext,
        embedder=mock_embedder,
        vector_store=mock_vector_store,
        rebuild_bm25=mock_rebuild,
        webhook_secret=WEBHOOK_SECRET,
        langfuse=mock_langfuse,
    )
    app.include_router(router, prefix="/webhook")
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Signature verification
# ---------------------------------------------------------------------------


def test_invalid_signature_returns_401(http_client: TestClient, mock_erpnext: AsyncMock) -> None:
    body = json.dumps({"doctype": "Contract", "docname": "CON-001"}).encode()
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
    body = json.dumps({"doctype": "Contract", "docname": "CON-001"}).encode()
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


def test_cancelled_contract_gets_cancelled_status_override(
    http_client: TestClient,
    mock_erpnext: AsyncMock,
    mock_vector_store: MagicMock,
) -> None:
    """Contract's own `status` field has no Cancelled option (only Unsigned/Active/
    Inactive) -- docstatus=2 must override it so a "cancelled" status filter can
    actually match (see issue #76)."""
    mock_erpnext.get_doc.side_effect = [
        {**CONTRACT_DOC, "docstatus": 2},
        CONTRACT_SUPPLIER_DOC,
    ]

    _post(http_client, {"doctype": "Contract", "docname": "CON-001"})

    chunks = mock_vector_store.upsert_chunks.call_args[0][0]
    assert chunks[0]["status"] == "Cancelled"


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
# PDF attachments (Contract only)
# ---------------------------------------------------------------------------


def test_contract_with_pdf_attachment_indexes_extra_chunk(
    http_client: TestClient,
    mock_erpnext: AsyncMock,
    mock_vector_store: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_erpnext.get_doc.side_effect = [CONTRACT_DOC, CONTRACT_SUPPLIER_DOC]
    mock_erpnext.get_attached_files.return_value = [
        {"name": "FILE-001", "file_url": "/private/files/con-001.pdf", "file_name": "con-001.pdf"}
    ]
    mock_erpnext.get_file_content.return_value = b"%PDF-fake-bytes"
    monkeypatch.setattr(
        "ingestion.webhook_handler.extract_text_from_pdf",
        lambda _bytes: "Attachment clause: delivery within 10 days.",
    )

    response = _post(http_client, {"doctype": "Contract", "docname": "CON-001"})

    assert response.status_code == 200
    mock_erpnext.get_attached_files.assert_called_once_with("Contract", "CON-001")
    mock_erpnext.get_file_content.assert_called_once_with("/private/files/con-001.pdf")

    chunks = mock_vector_store.upsert_chunks.call_args[0][0]
    # Contract's own HTML-stripped text + 1 PDF-derived chunk
    texts = [c["text"] for c in chunks]
    assert any("Attachment clause" in t for t in texts)
    # attachment chunks still carry the parent Contract's metadata (same docname)
    assert all(c["docname"] == "CON-001" for c in chunks)
    assert all(c["source_doctype"] == "Contract" for c in chunks)
    total = len(chunks)
    assert all(c["total_chunks"] == total for c in chunks)
    assert {c["chunk_index"] for c in chunks} == set(range(total))


def test_non_pdf_attachments_are_skipped(
    http_client: TestClient,
    mock_erpnext: AsyncMock,
    mock_vector_store: MagicMock,
) -> None:
    mock_erpnext.get_doc.side_effect = [CONTRACT_DOC, CONTRACT_SUPPLIER_DOC]
    mock_erpnext.get_attached_files.return_value = [
        {"name": "FILE-001", "file_url": "/private/files/photo.png", "file_name": "photo.png"}
    ]

    _post(http_client, {"doctype": "Contract", "docname": "CON-001"})

    mock_erpnext.get_file_content.assert_not_called()
    chunks = mock_vector_store.upsert_chunks.call_args[0][0]
    assert len(chunks) == 1  # only the Contract's own text


def test_attachment_fetch_failure_does_not_block_indexing(
    http_client: TestClient,
    mock_erpnext: AsyncMock,
    mock_vector_store: MagicMock,
) -> None:
    mock_erpnext.get_doc.side_effect = [CONTRACT_DOC, CONTRACT_SUPPLIER_DOC]
    mock_erpnext.get_attached_files.side_effect = RuntimeError("ERPNext unreachable")

    response = _post(http_client, {"doctype": "Contract", "docname": "CON-001"})

    assert response.status_code == 200
    assert response.json()["status"] == "indexed"
    chunks = mock_vector_store.upsert_chunks.call_args[0][0]
    assert len(chunks) == 1  # Contract's own text still indexed


def test_pdf_extraction_failure_for_one_file_does_not_block_others(
    http_client: TestClient,
    mock_erpnext: AsyncMock,
    mock_vector_store: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_erpnext.get_doc.side_effect = [CONTRACT_DOC, CONTRACT_SUPPLIER_DOC]
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

    response = _post(http_client, {"doctype": "Contract", "docname": "CON-001"})

    assert response.status_code == 200
    chunks = mock_vector_store.upsert_chunks.call_args[0][0]
    texts = [c["text"] for c in chunks]
    assert any("Good PDF text" in t for t in texts)


def test_attachments_not_fetched_for_non_attachment_doctypes(
    http_client: TestClient,
    mock_erpnext: AsyncMock,
) -> None:
    mock_erpnext.get_doc.side_effect = [TERMS_DOC]

    _post(http_client, {"doctype": "Terms and Conditions", "docname": "Standard Terms"})

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

    mock_erpnext.get_doc.side_effect = [CONTRACT_DOC, ERPNextNotFoundError("not found")]

    response = _post(http_client, {"doctype": "Contract", "docname": "CON-001"})

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
    mock_erpnext.get_doc.side_effect = [CONTRACT_DOC, CONTRACT_SUPPLIER_DOC]

    manager = MagicMock()
    manager.attach_mock(mock_vector_store.delete_by_docname, "delete")
    manager.attach_mock(mock_vector_store.upsert_chunks, "upsert")

    _post(http_client, {"doctype": "Contract", "docname": "CON-001"})

    method_names = [c[0] for c in manager.mock_calls]
    assert method_names.index("delete") < method_names.index("upsert")


def test_rebuild_bm25_called_after_upsert(
    http_client: TestClient,
    mock_erpnext: AsyncMock,
    mock_vector_store: MagicMock,
    mock_rebuild: MagicMock,
) -> None:
    mock_erpnext.get_doc.side_effect = [CONTRACT_DOC, CONTRACT_SUPPLIER_DOC]

    manager = MagicMock()
    manager.attach_mock(mock_vector_store.upsert_chunks, "upsert")
    manager.attach_mock(mock_rebuild, "rebuild")

    _post(http_client, {"doctype": "Contract", "docname": "CON-001"})

    method_names = [c[0] for c in manager.mock_calls]
    assert method_names.index("upsert") < method_names.index("rebuild")


# ---------------------------------------------------------------------------
# Frappe full-doc payload — primary key sent as "name", not "docname"
# ---------------------------------------------------------------------------


def test_contract_indexed_via_name_field(
    http_client: TestClient,
    mock_erpnext: AsyncMock,
    mock_vector_store: MagicMock,
) -> None:
    """Frappe's full-doc webhook format sends the PK as 'name', not 'docname'.

    webhook_handler.py falls back to payload.get('name') when 'docname' is absent.
    A missing or empty 'name' would cause a silent empty-string lookup and a skipped
    or errored index — this test ensures the fallback path is exercised.
    """
    mock_erpnext.get_doc.side_effect = [CONTRACT_DOC, CONTRACT_SUPPLIER_DOC]

    body = json.dumps({"doctype": "Contract", "name": "CON-001"}).encode()
    response = http_client.post(
        "/webhook/erpnext",
        content=body,
        headers={
            "Content-Type": "application/json",
            "X-Frappe-Webhook-Signature": _sign(body),
        },
    )

    assert response.status_code == 200
    assert response.json() == {"status": "indexed", "docname": "CON-001"}
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
    mock_erpnext.get_doc.side_effect = [CONTRACT_DOC, CONTRACT_SUPPLIER_DOC]
    mock_vector_store.upsert_chunks.side_effect = RuntimeError("Qdrant connection refused")

    response = _post(http_client, {"doctype": "Contract", "docname": "CON-001"})

    # delete ran — document is now absent from the index
    mock_vector_store.delete_by_docname.assert_called_once_with("CON-001")
    # error must propagate as 5xx, not be swallowed as a 200 "indexed"
    assert response.status_code == 500


# ---------------------------------------------------------------------------
# Langfuse tracing (issue #123)
# ---------------------------------------------------------------------------


def test_no_trace_created_without_langfuse(
    http_client: TestClient, mock_erpnext: AsyncMock
) -> None:
    """The default wiring passes no Langfuse client — indexing still works and
    nothing tries to trace."""
    mock_erpnext.get_doc.side_effect = [CONTRACT_DOC, CONTRACT_SUPPLIER_DOC]
    response = _post(http_client, {"doctype": "Contract", "docname": "CON-001"})
    assert response.json()["status"] == "indexed"


def test_indexed_webhook_creates_trace_with_step_spans(
    traced_http_client: TestClient,
    mock_erpnext: AsyncMock,
    mock_langfuse: MagicMock,
) -> None:
    mock_erpnext.get_doc.side_effect = [CONTRACT_DOC, CONTRACT_SUPPLIER_DOC]

    _post(traced_http_client, {"doctype": "Contract", "docname": "CON-001"})

    mock_langfuse.trace.assert_called_once_with(
        name="webhook_reindex", input={"doctype": "Contract", "docname": "CON-001"}
    )
    trace = mock_langfuse.trace.return_value
    span_names = {c.kwargs["name"] for c in trace.span.call_args_list}
    assert span_names == {"fetch", "parse", "chunk", "upsert"}
    trace.update.assert_called_with(output={"status": "indexed", "chunk_count": 1})


def test_embed_step_is_traced_as_generation_with_usage(
    traced_http_client: TestClient,
    mock_erpnext: AsyncMock,
    mock_langfuse: MagicMock,
) -> None:
    """The embed step must be a Langfuse `generation` (model + token usage), so
    ingestion embedding cost is captured the way query-time generation cost is."""
    mock_erpnext.get_doc.side_effect = [CONTRACT_DOC, CONTRACT_SUPPLIER_DOC]

    _post(traced_http_client, {"doctype": "Contract", "docname": "CON-001"})

    trace = mock_langfuse.trace.return_value
    trace.generation.assert_called_once_with(name="embed", model="text-embedding-3-small")
    gen = trace.generation.return_value
    assert gen.end.call_args.kwargs["usage"] == {"input": 7, "total": 7}


def test_skipped_webhook_updates_trace_output(
    traced_http_client: TestClient,
    mock_erpnext: AsyncMock,
    mock_langfuse: MagicMock,
) -> None:
    empty_contract = {**CONTRACT_DOC, "contract_terms": ""}
    mock_erpnext.get_doc.side_effect = [empty_contract, CONTRACT_SUPPLIER_DOC]

    _post(traced_http_client, {"doctype": "Contract", "docname": "CON-001"})

    trace = mock_langfuse.trace.return_value
    trace.update.assert_called_with(output={"status": "skipped", "reason": "empty text"})
    trace.generation.assert_not_called()


def test_ignored_doctype_creates_no_trace(
    traced_http_client: TestClient, mock_langfuse: MagicMock
) -> None:
    _post(traced_http_client, {"doctype": "Sales Order", "docname": "SO-001"})
    mock_langfuse.trace.assert_not_called()


def test_failed_webhook_records_error_on_trace_and_failing_span(
    traced_http_client: TestClient,
    mock_erpnext: AsyncMock,
    mock_vector_store: MagicMock,
    mock_langfuse: MagicMock,
) -> None:
    mock_erpnext.get_doc.side_effect = [CONTRACT_DOC, CONTRACT_SUPPLIER_DOC]
    mock_vector_store.upsert_chunks.side_effect = RuntimeError("Qdrant down")

    response = _post(traced_http_client, {"doctype": "Contract", "docname": "CON-001"})

    assert response.status_code == 500
    trace = mock_langfuse.trace.return_value
    # traces have no `level`; the root records an error status, the span carries ERROR
    trace.update.assert_called_with(output={"status": "error", "error": "RuntimeError"})
    trace.span.return_value.end.assert_any_call(level="ERROR")
