"""Streamlit frontend tests via streamlit.testing.v1.AppTest.

In-process, no browser and no running server -- the script itself is executed
against a simulated session. `httpx.post` is monkeypatched globally (frontend/app.py
does `import httpx; httpx.post(...)`, so patching the shared module object reaches it
regardless of which module imported it).

See docs/IMPLEMENTATION_PLAN.md Step 17 for why this is a separate, lighter-weight
layer from the Playwright-driven tests/e2e/ suite (Desk UI, real browser).
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from streamlit.testing.v1 import AppTest

_FRONTEND_DIR = str(Path(__file__).resolve().parent.parent / "frontend")
if _FRONTEND_DIR not in sys.path:
    sys.path.insert(0, _FRONTEND_DIR)

_APP_PATH = str(Path(__file__).resolve().parent.parent / "frontend" / "app.py")


def _make_response(answer: str, sources: list[dict] | None = None) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"answer": answer, "sources": sources or []}
    return resp


@pytest.fixture
def logged_in_app(monkeypatch: pytest.MonkeyPatch) -> AppTest:
    monkeypatch.setenv("BACKEND_URL", "http://localhost:8000")
    at = AppTest.from_file(_APP_PATH)
    at.session_state["jwt"] = "test-jwt"
    at.run()
    return at


def test_basic_query_renders_answer(
    monkeypatch: pytest.MonkeyPatch, logged_in_app: AppTest
) -> None:
    monkeypatch.setattr(
        "httpx.post",
        lambda *a, **kw: _make_response(
            "Net 30. [PO-001]",
            [{"docname": "PO-001", "source_doctype": "Purchase Order", "supplier": "Acme Corp"}],
        ),
    )

    at = logged_in_app.chat_input[0].set_value("What are the payment terms?").run()

    assert at.exception.len == 0
    markdown_values = [m.value for m in at.markdown]
    assert "What are the payment terms?" in markdown_values
    assert "Net 30. [PO-001]" in markdown_values


def test_sidebar_supplier_filter_passed_to_api(
    monkeypatch: pytest.MonkeyPatch, logged_in_app: AppTest
) -> None:
    captured: dict = {}

    def fake_post(url, json, **kwargs):
        captured["json"] = json
        return _make_response("Net 30. [PO-001]")

    monkeypatch.setattr("httpx.post", fake_post)

    logged_in_app.sidebar.text_input[0].set_value("Acme Corp")
    at = logged_in_app.chat_input[0].set_value("What are the payment terms?").run()

    assert at.exception.len == 0
    assert captured["json"]["filters"] == {"supplier": "Acme Corp"}


def test_api_error_shows_message_without_crashing(
    monkeypatch: pytest.MonkeyPatch, logged_in_app: AppTest
) -> None:
    def fake_post(*a, **kw):
        raise ConnectionError("connection refused")

    monkeypatch.setattr("httpx.post", fake_post)

    at = logged_in_app.chat_input[0].set_value("What are the payment terms?").run()

    assert at.exception.len == 0
    markdown_values = [m.value for m in at.markdown]
    assert any("Error connecting to API" in v for v in markdown_values)
