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

NOTE ON SELECTORS: the Desk UI locators below have been verified against a live
Frappe/ERPNext instance (see docs/IMPLEMENTATION_PLAN.md Step 17 for the live-run
results and the bugs that live run found and fixed) -- re-verify if the target
Frappe version differs meaningfully from the one this was built against.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
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
FRONTEND_URL = os.environ.get("FRONTEND_URL", "http://localhost:8501")
# Whatever collection the *live app process* is configured with -- webhook-triggered
# re-indexing lands there, not in any dedicated test collection we could create here.
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "contract")


def mint_test_jwt(username: str = "e2e-test") -> str:
    """Mint a JWT the same way `/auth/callback` does after a real OAuth exchange
    (`api.auth.jwt_handler.mint_token`), so browser tests can log into the
    Streamlit frontend via `{FRONTEND_URL}/?token=<jwt>` without driving
    ERPNext's OAuth consent screen through the browser."""
    from api.auth.jwt_handler import mint_token

    return mint_token(username=username, roles=["System Manager"])


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


def create_draft_contract(
    contract_terms: str = "<p>E2E test contract terms.</p>",
) -> tuple[str, str]:
    """Create a minimal draft Contract via REST. Returns (docname, supplier).

    REST is fine for setup -- the thing under test is usually the Desk UI
    *action* (submit/save/cancel), not document creation. `contract_terms` is
    overridable so a test can plant distinctive, uniquely-searchable content
    (see test_erpnext_to_streamlit_loop.py).
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
            "contract_terms": contract_terms,
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


def attach_test_pdf(docname: str, paragraphs: int = 1, line: str | None = None) -> str:
    """Generate a minimal PDF via reportlab and attach it to a Contract via REST.
    Returns the generated filename. `paragraphs` controls text volume -- pass
    enough to push the doc's combined text over chunk_size=512 when a test needs
    to force multiple chunks (a single short paragraph plus the default
    `contract_terms` text normally chunks to just one). `line` overrides the
    repeated filler text -- pass distinctive content a test needs to search for.
    """
    from reportlab.pdfgen import canvas

    buffer = BytesIO()
    c = canvas.Canvas(buffer)
    y = 750
    line = line or "E2E test PDF attachment content, repeated to pad chunk length. "
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


def delete_qdrant_points_for_docname(docname: str) -> None:
    """Best-effort delete of a docname's points from the *production* Qdrant
    collection. `on_cancel` re-indexes with status="Cancelled" rather than
    removing points (see docs/DEPLOYMENT.md), so cleanup must do this
    explicitly -- otherwise every test run leaves its points permanently
    discoverable by the real /query pipeline (see #95)."""
    with contextlib.suppress(Exception):
        httpx.post(
            f"{QDRANT_URL}/collections/{QDRANT_COLLECTION}/points/delete",
            json={"filter": {"must": [{"key": "docname", "match": {"value": docname}}]}},
            timeout=10,
        )


def cleanup_contract(docname: str) -> None:
    """Full best-effort teardown for a test Contract, safe to call from a
    `finally` block: cancel it (no-op if already cancelled/still draft),
    delete the ERPNext document, and delete its Qdrant points. Each step
    independently suppresses errors so one failing step (e.g. already
    cancelled) doesn't block the rest.

    Replaces bare `cancel_contract_via_rest` as the standard finally-block
    cleanup -- cancelling alone left every test Contract permanently in
    ERPNext *and* permanently indexed in the production Qdrant collection
    real /query requests hit (see #95).
    """
    cancel_contract_via_rest(docname)
    delete_contract_via_rest(docname)
    delete_qdrant_points_for_docname(docname)


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
    # Playwright glob patterns don't match `/` with a bare `*` (only `**` does),
    # so f"{ERPNEXT_URL}/app*" never matches ERPNext's actual post-login
    # redirect to /app/home -- use a regex instead. Found by running this live.
    page.wait_for_url(re.compile(rf"{re.escape(ERPNEXT_URL)}/app(/.*)?$"), timeout=15000)
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
    (e.g. "Cancelled"). NOTE: found via a live run that Contract shows its own
    business-status indicator ("Unsigned"/"Active") after Submit rather than a
    generic "Submitted" label -- use `wait_for_docstatus_submitted` for that
    case instead of this helper."""
    page.locator(".indicator-pill", has_text=text).first.wait_for(timeout=timeout)


def wait_for_docstatus_submitted(page, timeout: int = 15000) -> None:
    """Wait for a Desk form to reflect docstatus=1 after Submit. Contract's
    indicator pill shows its own computed business status (e.g. "Unsigned"),
    not a generic "Submitted" label, so the reliable signal is the Cancel
    primary action appearing -- Frappe only shows Cancel on a submitted,
    not-yet-cancelled document. Found via a live run against this doctype."""
    page.get_by_role("button", name="Cancel", exact=True).wait_for(timeout=timeout)


def set_is_signed_and_update(page, checked: bool) -> None:
    """Toggle Contract's `is_signed` checkbox (an allow_on_submit field) and
    save via the "Update" primary action -- the label Frappe uses for saving
    edits to an already-submitted document, distinct from "Save" (drafts only).
    The field wrapper contains a second, disabled, display-only checkbox input
    (hidden but present in the DOM), so the locator must scope to `.input-area`
    to avoid a strict-mode ambiguous-match error. Found via a live run.
    """
    checkbox = page.locator('[data-fieldname="is_signed"] .input-area input[type="checkbox"]')
    if checked:
        checkbox.check()
    else:
        checkbox.uncheck()
    page.get_by_role("button", name="Update", exact=True).click()
