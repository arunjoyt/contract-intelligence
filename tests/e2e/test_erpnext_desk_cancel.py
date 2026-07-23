"""Desk-driven Contract cancel -> webhook -> Qdrant re-index with status=Cancelled.

ingestion/webhook_handler.py is event-agnostic: on_cancel does NOT remove the
document's points from Qdrant, it re-indexes the full document with `status`
overridden to "Cancelled" (docstatus == 2), matching docs/DEPLOYMENT.md's "How
modifications are handled" section. This test asserts that behavior specifically
through the Desk UI cancel action, not REST.
"""

from __future__ import annotations

from tests.e2e.conftest import (
    ERPNEXT_URL,
    click_primary_action,
    create_draft_contract,
    live,
    poll_qdrant,
    submit_contract_via_rest,
    wait_for_indicator,
)
from tests.e2e.conftest import cancel_contract_via_rest as _cancel


@live
def test_contract_desk_cancel_reindexes_with_cancelled_status(desk_page) -> None:
    docname, _ = create_draft_contract()
    try:
        submit_contract_via_rest(docname)
        poll_qdrant(docname, predicate=lambda pts: len(pts) > 0)  # baseline: indexed

        desk_page.goto(f"{ERPNEXT_URL}/app/contract/{docname}")
        click_primary_action(desk_page, "Cancel")
        wait_for_indicator(desk_page, "Cancelled")

        points = poll_qdrant(
            docname,
            predicate=lambda pts: any(p["payload"].get("status") == "Cancelled" for p in pts),
        )
        assert points, f"No Qdrant points found for {docname} after Desk cancel"
        assert any(p["payload"].get("status") == "Cancelled" for p in points), (
            f"Qdrant payload for {docname} was not updated to status=Cancelled after Desk cancel"
        )
    finally:
        _cancel(docname)  # no-op if already cancelled; safe cleanup
