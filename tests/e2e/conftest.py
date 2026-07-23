"""Shared fixtures for the Playwright-driven ERPNext Desk E2E suite.

See docs/IMPLEMENTATION_PLAN.md Step 17 and docs/ARCHITECTURE.md § Testing Strategy
(Layer 3) for why this suite exists: tests/test_integration.py drives ERPNext via
REST (`frappe.client.submit`/`save`), which bypasses the Frappe background worker
queue that fires webhooks on the real Desk UI path. This suite drives the actual
Desk UI in a real Chromium browser instead, to catch webhook-firing bugs REST calls
structurally cannot see.

Requires (not installed/run by default -- see README's Integration Tests section):
    pip install pytest-playwright && playwright install chromium
    RUN_E2E=1 in the environment
    ERPNEXT_ADMIN_USERNAME / ERPNEXT_ADMIN_PASSWORD in .env -- Desk UI login,
    distinct from ERPNEXT_API_KEY/SECRET (REST-only, used by ERPNextClient elsewhere)
    A running app (uvicorn or docker) reachable at API_URL (default localhost:8000),
    since ERPNext's webhook fires against whatever URL its Webhook records point to --
    this suite does not start the app itself.

NOTE ON SELECTORS: the Desk UI locators below (#login_email, [data-fieldname=...],
role=button names) follow Frappe's documented, version-stable conventions, but have
not been exercised against a live browser as part of building this scaffold --
verify against the target Frappe version on first real run (see docs/IMPLEMENTATION_PLAN.md
Step 17 for context on why this suite was added un-executed).
"""

from __future__ import annotations

import contextlib
import json
import os
import time
import uuid
from datetime import UTC, datetime, timedelta
from io import BytesIO

import httpx
import pytest
from dotenv import load_dotenv

load_dotenv()

_run = os.getenv("RUN_E2E")
live = pytest.mark.skipif(
    not _run, reason="set RUN_E2E=1 to run (requires: playwright install chromium)"
)

ERPNEXT_URL = os.environ.get("ERPNEXT_URL", "http://127.0.0.1:8005")
APP_URL = os.environ.get("API_URL", "http://localhost:8000")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
# Whatever collection the *live app process* is configured with -- webhook-triggered
# re-indexing lands there, not in any dedicated test collection we could create here.
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "contract")


# ---------------------------------------------------------------------------
# ERPNext REST helpers -- used only for test-data setup, never for the action
# under test (submit/save/cancel), which must go through the Desk UI.
# ---------------------------------------------------------------------------


def _erp_credentials() -> tuple[str, str, str]:
    key = os.environ.get("ERPNEXT_API_KEY", "")
    secret = os.environ.get("ERPNEXT_API_SECRET", "")
    if not key or not secret:
        pytest.skip("ERPNEXT_API_KEY / ERPNEXT_API_SECRET not set in .env")
    return ERPNEXT_URL, key, secret


def _erp_headers() -> dict:
    _, key, secret = _erp_credentials()
    return {"Authorization": f"token {key}:{secret}"}


def create_draft_contract() -> tuple[str, str]:
    """Create a minimal draft Contract via REST. Returns (docname, supplier).

    REST is fine for setup -- the thing under test is the Desk UI *action*
    (submit/save/cancel), not document creation.
    """
    headers = _erp_headers()
    supplier = (
        httpx.get(
            f"{ERPNEXT_URL}/api/resource/Supplier",
            params={"limit_page_length": 1, "fields": '["name"]'},
            headers=headers,
            timeout=10,
        )
        .json()
        .get("data", [{}])[0]
        .get("name")
    )
    if not supplier:
        pytest.skip("No Supplier found on ERPNext site")

    start_date = datetime.now(tz=UTC).strftime("%Y-%m-%d")
    end_date = (datetime.now(tz=UTC) + timedelta(days=365)).strftime("%Y-%m-%d")
    resp = httpx.post(
        f"{ERPNEXT_URL}/api/resource/Contract",
        json={
            "doctype": "Contract",
            "party_type": "Supplier",
            "party_name": supplier,
            "start_date": start_date,
            "end_date": end_date,
            "contract_terms": "<p>E2E test contract terms.</p>",
        },
        headers=headers,
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()["data"]["name"], supplier


def submit_contract_via_rest(docname: str) -> None:
    """Submit via REST -- used by tests that only need an already-submitted
    Contract as a starting point (e.g. the update-after-submit / cancel tests),
    where the Desk-driven action under test is something other than submit itself.
    """
    headers = _erp_headers()
    doc = httpx.get(
        f"{ERPNEXT_URL}/api/resource/Contract/{docname}", headers=headers, timeout=10
    ).json()["data"]
    httpx.post(
        f"{ERPNEXT_URL}/api/method/frappe.client.submit",
        data={"doc": json.dumps(doc)},
        headers=headers,
        timeout=15,
    ).raise_for_status()


def cancel_contract_via_rest(docname: str) -> None:
    """Best-effort cleanup cancel; safe to call from a finally block."""
    with contextlib.suppress(Exception):
        httpx.post(
            f"{ERPNEXT_URL}/api/method/frappe.client.cancel",
            data={"doctype": "Contract", "name": docname},
            headers=_erp_headers(),
            timeout=10,
        )


def attach_test_pdf(docname: str, paragraphs: int = 1) -> str:
    """Generate a minimal PDF via reportlab and attach it to a Contract via REST.
    Returns the generated filename. `paragraphs` controls text volume -- pass
    enough to push the doc's combined text over chunk_size=512 when a test needs
    to force multiple chunks (a single short paragraph plus the default
    `contract_terms` text normally chunks to just one).
    """
    from reportlab.pdfgen import canvas

    buffer = BytesIO()
    c = canvas.Canvas(buffer)
    y = 750
    line = "E2E test PDF attachment content, repeated to pad chunk length. "
    for i in range(paragraphs):
        c.drawString(72, y, f"{i}: {line}")
        y -= 20
        if y < 50:
            c.showPage()
            y = 750
    c.save()

    filename = f"e2e-test-{uuid.uuid4().hex[:8]}.pdf"
    resp = httpx.post(
        f"{ERPNEXT_URL}/api/method/upload_file",
        files={"file": (filename, buffer.getvalue(), "application/pdf")},
        data={"doctype": "Contract", "docname": docname, "is_private": 1},
        headers=_erp_headers(),
        timeout=15,
    )
    resp.raise_for_status()
    return filename


def delete_contract_via_rest(docname: str) -> None:
    """Best-effort cleanup delete (only works on a cancelled/draft doc)."""
    with contextlib.suppress(Exception):
        httpx.delete(
            f"{ERPNEXT_URL}/api/resource/Contract/{docname}", headers=_erp_headers(), timeout=10
        )


# ---------------------------------------------------------------------------
# Qdrant polling helpers -- read the *production* collection the live app
# writes to (mirrors tests/test_integration.py Group 9's approach).
# ---------------------------------------------------------------------------


def qdrant_points_for_docname(docname: str) -> list[dict]:
    try:
        r = httpx.post(
            f"{QDRANT_URL}/collections/{QDRANT_COLLECTION}/points/scroll",
            json={
                "filter": {"must": [{"key": "docname", "match": {"value": docname}}]},
                "limit": 50,
                "with_payload": True,
            },
            timeout=5,
        )
        if r.status_code == 200:
            return r.json()["result"]["points"]
    except Exception:
        pass
    return []


def poll_qdrant(docname: str, *, predicate, timeout: int = 60) -> list[dict]:
    """Poll until predicate(points) is True or timeout expires (webhook delivery +
    re-indexing is async relative to the Desk UI action that triggered it)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        points = qdrant_points_for_docname(docname)
        if predicate(points):
            return points
        time.sleep(3)
    return qdrant_points_for_docname(docname)


# ---------------------------------------------------------------------------
# Playwright / Desk UI fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def app_reachable() -> None:
    """Skip the whole suite if the app ERPNext's webhooks target isn't running."""
    try:
        httpx.get(f"{APP_URL}/health", timeout=3).raise_for_status()
    except Exception:
        pytest.skip(f"App not reachable at {APP_URL} -- start it before running RUN_E2E=1")


@pytest.fixture(scope="session")
def desk_login_state(browser, app_reachable):
    """Log into the ERPNext Desk once per session; reuse cookies (storage_state)
    across tests instead of re-authenticating in every test's own context."""
    username = os.environ.get("ERPNEXT_ADMIN_USERNAME")
    password = os.environ.get("ERPNEXT_ADMIN_PASSWORD")
    if not username or not password:
        pytest.skip("ERPNEXT_ADMIN_USERNAME / ERPNEXT_ADMIN_PASSWORD not set in .env")

    context = browser.new_context()
    page = context.new_page()
    page.goto(f"{ERPNEXT_URL}/login")
    page.fill("#login_email", username)
    page.fill("#login_password", password)
    page.click(".btn-login")
    page.wait_for_url(f"{ERPNEXT_URL}/app*", timeout=15000)
    state = context.storage_state()
    context.close()
    return state


@pytest.fixture
def desk_page(browser, desk_login_state):
    """A Playwright Page already authenticated against the ERPNext Desk."""
    context = browser.new_context(storage_state=desk_login_state)
    page = context.new_page()
    yield page
    context.close()


def click_primary_action(page, label: str) -> None:
    """Click a Desk form's primary action button (e.g. Submit, Cancel) and
    confirm the resulting Frappe confirmation dialog."""
    page.get_by_role("button", name=label, exact=True).click()
    with contextlib.suppress(Exception):
        page.get_by_role("button", name="Yes", exact=True).click(timeout=5000)


def wait_for_indicator(page, text: str, timeout: int = 15000) -> None:
    """Wait for the Desk form's status indicator pill to show `text`
    (e.g. "Submitted", "Cancelled")."""
    page.locator(".indicator-pill", has_text=text).first.wait_for(timeout=timeout)
