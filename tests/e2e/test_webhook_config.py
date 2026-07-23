"""REST-only assertions that ERPNext's Webhook records match docs/DEPLOYMENT.md's
"Required webhook records" table exactly. No browser needed -- catches the class
of bug where indexing quietly stops working because a Webhook record is missing,
disabled, or misconfigured, before any Desk-UI-driven test even runs.
"""

from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor

import pytest
from dotenv import load_dotenv

from ingestion.erpnext_client import ERPNextClient

load_dotenv()

_run = os.getenv("RUN_E2E")
live = pytest.mark.skipif(not _run, reason="set RUN_E2E=1 to run")

# Mirrors docs/DEPLOYMENT.md's "Required webhook records" table exactly.
REQUIRED_WEBHOOKS = {
    "contract-on-submit": ("Contract", "on_submit"),
    "contract-on-update": ("Contract", "on_update"),
    "contract-on-cancel": ("Contract", "on_cancel"),
    "terms-on-update": ("Terms and Conditions", "on_update"),
}

EXPECTED_WEBHOOK_URL = os.environ.get(
    "E2E_WEBHOOK_URL", "http://127.0.0.1:8000/webhook/erpnext"
)


@pytest.fixture(scope="module")
def webhooks() -> dict[str, dict]:
    async def _fetch() -> dict[str, dict]:
        async with ERPNextClient() as client:
            records = await client.get_list(
                "Webhook",
                fields=[
                    "name",
                    "webhook_doctype",
                    "webhook_docevent",
                    "enabled",
                    "enable_security",
                    "request_url",
                    "webhook_secret",
                ],
                limit=0,
            )
        return {r["name"]: r for r in records}

    # Playwright's sync API (used by the Desk-UI test files collected alongside
    # this one) keeps its own event loop registered on the main thread once its
    # `browser` fixture has been instantiated -- calling asyncio.run() directly
    # here then raises "asyncio.run() cannot be called from a running event
    # loop", even though this fixture has nothing to do with Playwright. Found
    # by running the full tests/e2e/ suite together, not this file alone.
    # Running it in a fresh thread sidesteps whatever loop state the main
    # thread has.
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, _fetch()).result()


@live
def test_all_required_webhooks_exist(webhooks: dict[str, dict]) -> None:
    missing = [name for name in REQUIRED_WEBHOOKS if name not in webhooks]
    assert not missing, f"Missing Webhook records in ERPNext: {missing}"


@live
def test_required_webhooks_target_correct_doctype_and_event(webhooks: dict[str, dict]) -> None:
    mismatched = []
    for name, (doctype, event) in REQUIRED_WEBHOOKS.items():
        record = webhooks.get(name)
        if record is None:
            continue  # reported by test_all_required_webhooks_exist
        if record["webhook_doctype"] != doctype or record["webhook_docevent"] != event:
            mismatched.append(
                f"{name}: expected ({doctype}, {event}), "
                f"got ({record['webhook_doctype']}, {record['webhook_docevent']})"
            )
    assert not mismatched, "\n".join(mismatched)


@live
def test_required_webhooks_are_enabled(webhooks: dict[str, dict]) -> None:
    disabled = [
        name for name in REQUIRED_WEBHOOKS if name in webhooks and not webhooks[name]["enabled"]
    ]
    assert not disabled, f"Webhook records exist but are disabled: {disabled}"


@live
def test_required_webhooks_have_security_enabled(webhooks: dict[str, dict]) -> None:
    """Without enable_security, Frappe never sends X-Frappe-Webhook-Signature and
    every request gets rejected with 401 by ingestion/webhook_handler.py -- see
    docs/DEPLOYMENT.md's "Enable Security" note."""
    insecure = [
        name
        for name in REQUIRED_WEBHOOKS
        if name in webhooks and not webhooks[name]["enable_security"]
    ]
    assert not insecure, f"Webhook records without enable_security: {insecure}"


@live
def test_required_webhooks_point_to_same_configured_url(webhooks: dict[str, dict]) -> None:
    """All required webhooks should point at the same backend URL. Compared against
    E2E_WEBHOOK_URL if set (default assumes local dev, see .env.example)."""
    urls = {
        name: webhooks[name]["request_url"]
        for name in REQUIRED_WEBHOOKS
        if name in webhooks
    }
    wrong = {name: url for name, url in urls.items() if url != EXPECTED_WEBHOOK_URL}
    assert not wrong, f"Webhooks not pointing at {EXPECTED_WEBHOOK_URL}: {wrong}"
