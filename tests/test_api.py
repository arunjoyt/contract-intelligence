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
