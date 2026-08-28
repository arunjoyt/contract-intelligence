"""Unit tests for api/main.py — all external services are mocked.

The lifespan startup is triggered by the TestClient context manager.
Constructors for every heavy singleton are monkeypatched before the client
is created so no real network or model calls are made.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import jwt
import pytest
from fastapi.testclient import TestClient

from api.main import app
from pipeline.query_pipeline import SourceDoc

ADMIN_SECRET = "test-admin-secret"
WEBHOOK_SECRET = "test-webhook-secret"
JWT_SECRET = "test-jwt-secret-at-least-32-chars-long"
ALLOWED_ROLES = "Purchase Manager,System Manager"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_jwt(roles: list[str] | None = None) -> str:
    payload = {
        "sub": "test-user",
        "roles": roles if roles is not None else ["Purchase Manager"],
        "exp": datetime.now(tz=UTC) + timedelta(hours=8),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


def _auth(roles: list[str] | None = None) -> dict[str, str]:
    return {"Authorization": f"Bearer {_make_jwt(roles)}"}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_pipeline() -> MagicMock:
    p = MagicMock()
    p.run.return_value = {
        "answer": "Acme Corp issued PO-001. [PO-001]",
        "sources": [
            SourceDoc(
                docname="PO-001",
                source_doctype="Purchase Order",
                supplier="Acme Corp",
                chunk_index=0,
            )
        ],
    }
    return p


@pytest.fixture
def client(monkeypatch, mock_pipeline):
    mock_vs = MagicMock()
    mock_vs.get_all_texts.return_value = []

    mock_hs = MagicMock()
    mock_ec = AsyncMock()
    mock_ec.aclose = AsyncMock()

    monkeypatch.setenv("ADMIN_SECRET", ADMIN_SECRET)
    monkeypatch.setenv("WEBHOOK_SECRET", WEBHOOK_SECRET)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
    monkeypatch.setenv("QDRANT_COLLECTION", "test_col")
    monkeypatch.setenv("ERPNEXT_URL", "http://localhost:8005")
    monkeypatch.setenv("ERPNEXT_API_KEY", "key")
    monkeypatch.setenv("ERPNEXT_API_SECRET", "secret")
    monkeypatch.setenv("JWT_SECRET", JWT_SECRET)
    monkeypatch.setenv("JWT_EXPIRY_HOURS", "8")
    monkeypatch.setenv("ALLOWED_ROLES", ALLOWED_ROLES)
    monkeypatch.setenv("ERPNEXT_OAUTH_CLIENT_ID", "test-client-id")
    monkeypatch.setenv("ERPNEXT_OAUTH_CLIENT_SECRET", "test-client-secret")
    monkeypatch.setenv("OAUTH_REDIRECT_URI", "http://localhost:8000/auth/callback")
    monkeypatch.setenv("FRONTEND_URL", "http://localhost:8501")
    # api.main calls load_dotenv() at import, so a developer's real .env leaks
    # LANGFUSE_* into os.environ. Without this, the lifespan builds a real
    # Langfuse client and tests emit real traces to a local Langfuse instance
    # (CI has no .env so it never saw this). Force tracing off.
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_HOST", raising=False)

    monkeypatch.setattr("api.main.Embedder", MagicMock)
    monkeypatch.setattr("api.main.VectorStore", lambda: mock_vs)
    monkeypatch.setattr("api.main.HybridSearch", lambda embedder, vector_store: mock_hs)
    monkeypatch.setattr("api.main.Reranker", MagicMock)
    monkeypatch.setattr("api.main.QueryRewriter", lambda embedder: MagicMock())
    monkeypatch.setattr("api.main.QueryPipeline", lambda **kwargs: mock_pipeline)
    monkeypatch.setattr("api.main.ERPNextClient", lambda: mock_ec)

    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


def test_health_returns_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# /query — auth
# ---------------------------------------------------------------------------


def test_query_without_token_returns_401(client):
    resp = client.post("/query", json={"question": "What POs exist?"})
    assert resp.status_code == 401


def test_query_with_invalid_token_returns_401(client):
    resp = client.post(
        "/query",
        json={"question": "What POs exist?"},
        headers={"Authorization": "Bearer not-a-valid-jwt"},
    )
    assert resp.status_code == 401


def test_query_with_disallowed_role_returns_403(client):
    resp = client.post(
        "/query",
        json={"question": "What POs exist?"},
        headers=_auth(roles=["Sales User"]),
    )
    assert resp.status_code == 403


# ---------------------------------------------------------------------------
# /query — pipeline
# ---------------------------------------------------------------------------


def test_query_returns_answer_and_sources(client, mock_pipeline):
    resp = client.post("/query", json={"question": "What POs exist?"}, headers=_auth())
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert data["answer"] == "Acme Corp issued PO-001. [PO-001]"
    assert len(data["sources"]) == 1
    assert data["sources"][0]["docname"] == "PO-001"
    assert data["sources"][0]["source_doctype"] == "Purchase Order"
    assert data["sources"][0]["supplier"] == "Acme Corp"
    assert data["sources"][0]["erpnext_url"] == "http://localhost:8005/app/purchase-order/PO-001"


def test_query_erpnext_url_encodes_docname_with_spaces(client, mock_pipeline):
    mock_pipeline.run.return_value = {
        "answer": "See terms. [Standard Purchase Terms - Net 30]",
        "sources": [
            SourceDoc(
                docname="Standard Purchase Terms - Net 30",
                source_doctype="Terms and Conditions",
                supplier=None,
                chunk_index=0,
            )
        ],
    }
    resp = client.post("/query", json={"question": "What are the terms?"}, headers=_auth())
    assert resp.status_code == 200
    data = resp.json()
    assert data["sources"][0]["erpnext_url"] == (
        "http://localhost:8005/app/terms-and-conditions/"
        "Standard%20Purchase%20Terms%20-%20Net%2030"
    )


def test_query_passes_question_to_pipeline(client, mock_pipeline):
    client.post("/query", json={"question": "Show me contracts"}, headers=_auth())
    mock_pipeline.run.assert_called_once()
    args, kwargs = mock_pipeline.run.call_args
    assert args[0] == "Show me contracts"


def test_query_passes_filters_to_pipeline(client, mock_pipeline):
    client.post(
        "/query",
        json={"question": "POs from Acme", "filters": {"supplier": "Acme Corp"}},
        headers=_auth(),
    )
    _, kwargs = mock_pipeline.run.call_args
    assert kwargs.get("filters") == {"supplier": "Acme Corp"}


def test_query_with_null_filters(client, mock_pipeline):
    resp = client.post("/query", json={"question": "anything", "filters": None}, headers=_auth())
    assert resp.status_code == 200
    _, kwargs = mock_pipeline.run.call_args
    assert kwargs.get("filters") is None


def test_query_missing_question_returns_422(client):
    resp = client.post("/query", json={}, headers=_auth())
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# /auth/login
# ---------------------------------------------------------------------------


def test_auth_login_redirects_to_erpnext(client):
    resp = client.get("/auth/login", follow_redirects=False)
    assert resp.status_code in (302, 307)
    assert "frappe.integrations.oauth2.authorize" in resp.headers["location"]


# ---------------------------------------------------------------------------
# /auth/callback
# ---------------------------------------------------------------------------


def test_auth_callback_invalid_state_returns_400(client):
    resp = client.get("/auth/callback?code=somecode&state=badstate")
    assert resp.status_code == 400


def test_auth_callback_duplicate_request_replays_same_redirect(client, monkeypatch):
    monkeypatch.setattr(
        "api.routers.auth.exchange_code_for_token",
        AsyncMock(return_value={"access_token": "erp-token"}),
    )
    monkeypatch.setattr(
        "api.routers.auth.fetch_user_roles",
        AsyncMock(return_value=("user@example.com", ["System Manager"])),
    )

    login_resp = client.get("/auth/login", follow_redirects=False)
    state = login_resp.headers["location"].split("state=")[1].split("&")[0]

    first = client.get(f"/auth/callback?code=somecode&state={state}", follow_redirects=False)
    assert first.status_code == 307

    second = client.get(f"/auth/callback?code=somecode&state={state}", follow_redirects=False)
    assert second.status_code == 307
    assert second.headers["location"] == first.headers["location"]


def test_auth_callback_replay_with_mismatched_code_returns_400(client, monkeypatch):
    """A `state` hit in the completed-login cache must not be served if `code`
    doesn't match the original request -- otherwise anyone who obtains just the
    `state` value (e.g. from access logs) within the TTL window could replay it
    with an arbitrary `code` and get the same valid JWT. See issue #64."""
    monkeypatch.setattr(
        "api.routers.auth.exchange_code_for_token",
        AsyncMock(return_value={"access_token": "erp-token"}),
    )
    monkeypatch.setattr(
        "api.routers.auth.fetch_user_roles",
        AsyncMock(return_value=("user@example.com", ["System Manager"])),
    )

    login_resp = client.get("/auth/login", follow_redirects=False)
    state = login_resp.headers["location"].split("state=")[1].split("&")[0]

    first = client.get(f"/auth/callback?code=realcode&state={state}", follow_redirects=False)
    assert first.status_code == 307

    replay = client.get(
        f"/auth/callback?code=attacker-guess&state={state}", follow_redirects=False
    )
    assert replay.status_code == 400


# ---------------------------------------------------------------------------
# /ingest/full
# ---------------------------------------------------------------------------


def test_ingest_full_without_secret_returns_403(client):
    resp = client.post("/ingest/full")
    assert resp.status_code == 403


def test_ingest_full_with_wrong_secret_returns_403(client):
    resp = client.post("/ingest/full", headers={"X-Admin-Secret": "wrong"})
    assert resp.status_code == 403


def test_ingest_full_with_correct_secret_returns_202(client):
    resp = client.post("/ingest/full", headers={"X-Admin-Secret": ADMIN_SECRET})
    assert resp.status_code == 202
    assert resp.json() == {"status": "accepted"}


def test_ingest_full_returns_409_when_a_run_is_in_flight(client):
    """A second /ingest/full while one holds the lock is rejected, not queued
    (issue #125)."""

    class _Locked:
        def locked(self) -> bool:
            return True

    client.app.state.ingest_lock = _Locked()
    resp = client.post("/ingest/full", headers={"X-Admin-Secret": ADMIN_SECRET})
    assert resp.status_code == 409


# ---------------------------------------------------------------------------
# Full-ingest Langfuse tracing (issue #123)
# ---------------------------------------------------------------------------


async def test_run_full_ingest_traces_per_doc_span_and_embed_generation(monkeypatch):
    from api.main import _run_full_ingest

    contract = {
        "name": "CON-001",
        "party_name": "Acme",
        "contract_terms": "<p>Net 30 payment terms.</p>",
        "status": "Active",
    }

    ec = AsyncMock()
    ec.__aenter__.return_value = ec
    ec.__aexit__.return_value = None
    ec.get_list.side_effect = lambda doctype, limit=0: (
        [{"name": "CON-001"}] if doctype == "Contract" else []
    )
    ec.get_doc.side_effect = [contract, {"supplier_group": "Trade"}]
    ec.get_attached_files.return_value = []
    monkeypatch.setattr("api.main.ERPNextClient", lambda: ec)

    embedder = MagicMock()
    embedder.embed_texts_with_usage.return_value = ([[0.1] * 1536], {"input": 12, "total": 12})
    vector_store = MagicMock()
    hybrid_search = MagicMock()

    lf = MagicMock()
    trace = lf.trace.return_value

    await _run_full_ingest(embedder, vector_store, hybrid_search, lf)

    lf.trace.assert_called_once_with(name="full_ingest")
    span_names = [c.kwargs["name"] for c in trace.span.call_args_list]
    assert "list:Contract" in span_names
    assert "Contract:CON-001" in span_names
    trace.generation.assert_called_once_with(name="embed", model="text-embedding-3-small")
    trace.generation.return_value.end.assert_called_once_with(usage={"input": 12, "total": 12})
    trace.update.assert_called_once_with(
        output={"documents_indexed": 1, "chunks_indexed": 1}
    )


async def test_run_full_ingest_traces_listing_failure(monkeypatch):
    from api.main import _run_full_ingest

    ec = AsyncMock()
    ec.__aenter__.return_value = ec
    ec.__aexit__.return_value = None
    ec.get_list.side_effect = RuntimeError("ERPNext 503")
    monkeypatch.setattr("api.main.ERPNextClient", lambda: ec)

    lf = MagicMock()
    trace = lf.trace.return_value

    await _run_full_ingest(MagicMock(), MagicMock(), MagicMock(), lf)

    # every listing failed -> a per-doctype ERROR span, and an error-flagged 0/0 output
    trace.span.return_value.end.assert_any_call(level="ERROR")
    trace.update.assert_called_once_with(
        output={
            "documents_indexed": 0,
            "chunks_indexed": 0,
            "status": "error",
            "failed_listings": ["Contract", "Terms and Conditions"],
        }
    )


async def test_run_full_ingest_flags_zero_indexed_as_error(monkeypatch):
    """The #125 bug shape: get_list returns [] (no exception) for every doctype.
    The run must not report a clean 'complete' -- output carries status=error."""
    from api.main import _run_full_ingest

    ec = AsyncMock()
    ec.__aenter__.return_value = ec
    ec.__aexit__.return_value = None
    ec.get_list.return_value = []
    monkeypatch.setattr("api.main.ERPNextClient", lambda: ec)

    lf = MagicMock()
    trace = lf.trace.return_value

    await _run_full_ingest(MagicMock(), MagicMock(), MagicMock(), lf)

    trace.update.assert_called_once_with(
        output={"documents_indexed": 0, "chunks_indexed": 0, "status": "error"}
    )


async def test_run_full_ingest_skips_when_lock_already_held():
    import asyncio

    from api.main import _run_full_ingest

    lock = asyncio.Lock()
    await lock.acquire()
    lf = MagicMock()

    await _run_full_ingest(MagicMock(), MagicMock(), MagicMock(), lf, lock)

    lf.trace.assert_not_called()  # duplicate run skipped, nothing traced


async def test_run_full_ingest_holds_lock_for_the_duration(monkeypatch):
    import asyncio

    from api.main import _run_full_ingest

    lock = asyncio.Lock()
    seen = {}

    ec = AsyncMock()
    ec.__aenter__.return_value = ec
    ec.__aexit__.return_value = None

    def _list(*_a, **_k):
        seen["locked_during_run"] = lock.locked()
        return []

    ec.get_list.side_effect = _list
    monkeypatch.setattr("api.main.ERPNextClient", lambda: ec)

    await _run_full_ingest(MagicMock(), MagicMock(), MagicMock(), None, lock)

    assert seen["locked_during_run"] is True
    assert lock.locked() is False  # released on exit


async def test_run_full_ingest_without_langfuse_still_indexes(monkeypatch):
    from api.main import _run_full_ingest

    ec = AsyncMock()
    ec.__aenter__.return_value = ec
    ec.__aexit__.return_value = None
    ec.get_list.side_effect = lambda doctype, limit=0: (
        [{"name": "CON-001"}] if doctype == "Contract" else []
    )
    ec.get_doc.side_effect = [
        {"name": "CON-001", "party_name": "Acme", "contract_terms": "<p>Terms.</p>"},
        {"supplier_group": "Trade"},
    ]
    ec.get_attached_files.return_value = []
    monkeypatch.setattr("api.main.ERPNextClient", lambda: ec)

    embedder = MagicMock()
    embedder.embed_texts.return_value = [[0.1] * 1536]
    vector_store = MagicMock()
    hybrid_search = MagicMock()

    await _run_full_ingest(embedder, vector_store, hybrid_search)

    vector_store.upsert_chunks.assert_called_once()
    embedder.embed_texts.assert_called_once()


# ---------------------------------------------------------------------------
# /webhook/erpnext
# ---------------------------------------------------------------------------


def _sign(body: bytes, secret: str = WEBHOOK_SECRET) -> str:
    return base64.b64encode(
        hmac.new(secret.encode(), body, hashlib.sha256).digest()
    ).decode()


def test_webhook_invalid_signature_returns_401(client):
    body = json.dumps({"doctype": "Purchase Order", "docname": "PO-001"}).encode()
    resp = client.post(
        "/webhook/erpnext",
        content=body,
        headers={
            "Content-Type": "application/json",
            "X-Frappe-Webhook-Signature": "bad",
        },
    )
    assert resp.status_code == 401


def test_webhook_unsupported_doctype_returns_ignored(client):
    body = json.dumps({"doctype": "Sales Order", "docname": "SO-001"}).encode()
    resp = client.post(
        "/webhook/erpnext",
        content=body,
        headers={
            "Content-Type": "application/json",
            "X-Frappe-Webhook-Signature": _sign(body),
        },
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "ignored"
