"""Desk-driven save on an already-submitted Contract -> webhook -> Qdrant update.

This is the open question flagged in docs/DEPLOYMENT.md's Future Enhancements:
`on_update_after_submit` was found not to fire reliably on Frappe 15 (verified
against Purchase Order, no longer in scope), but the registered webhook here is
plain `on_update` (`contract-on-update`) -- and Contract's DocType metadata lists
`status` as an `allow_on_submit` field, so a Desk save that only changes `status`
should be a legal, saveable edit on a submitted doc. This suite verifies that
empirically against Contract specifically, rather than assuming the Purchase
Order finding carries over.
"""

from __future__ import annotations

from tests.e2e.conftest import (
    ERPNEXT_URL,
    create_draft_contract,
    live,
    poll_qdrant,
    submit_contract_via_rest,
)
from tests.e2e.conftest import cancel_contract_via_rest as _cancel


@live
def test_contract_desk_status_change_after_submit_triggers_reindex(desk_page) -> None:
    docname, _ = create_draft_contract()
    try:
        submit_contract_via_rest(docname)
        poll_qdrant(docname, predicate=lambda pts: len(pts) > 0)  # baseline: indexed

        desk_page.goto(f"{ERPNEXT_URL}/app/contract/{docname}")
        desk_page.locator('[data-fieldname="status"] select').select_option("Active")
        desk_page.get_by_role("button", name="Save", exact=True).click()
        desk_page.wait_for_timeout(2000)  # let the save round-trip complete

        points = poll_qdrant(
            docname,
            predicate=lambda pts: any(p["payload"].get("status") == "Active" for p in pts),
        )
        assert any(p["payload"].get("status") == "Active" for p in points), (
            f"Qdrant payload for {docname} was not updated to status=Active after Desk save "
            "-- on_update may not be firing for post-submit Desk saves on Contract"
        )
    finally:
        _cancel(docname)


@live
def test_contract_desk_reindex_after_submit_is_idempotent(desk_page) -> None:
    """A second Desk save with the same field value should upsert in place
    (deterministic point IDs), not create duplicate points."""
    docname, _ = create_draft_contract()
    try:
        submit_contract_via_rest(docname)

        desk_page.goto(f"{ERPNEXT_URL}/app/contract/{docname}")
        desk_page.locator('[data-fieldname="status"] select').select_option("Active")
        desk_page.get_by_role("button", name="Save", exact=True).click()
        desk_page.wait_for_timeout(2000)
        points_after_first_save = poll_qdrant(docname, predicate=lambda pts: len(pts) > 0)

        # Re-select the same value -- Frappe only shows a Save action when the
        # form is dirty, so toggle away and back to force a real second save.
        desk_page.locator('[data-fieldname="status"] select').select_option("Inactive")
        desk_page.get_by_role("button", name="Save", exact=True).click()
        desk_page.wait_for_timeout(2000)
        desk_page.locator('[data-fieldname="status"] select').select_option("Active")
        desk_page.get_by_role("button", name="Save", exact=True).click()
        desk_page.wait_for_timeout(2000)

        points_after_second_save = poll_qdrant(
            docname,
            predicate=lambda pts: any(p["payload"].get("status") == "Active" for p in pts),
        )
        assert len(points_after_second_save) == len(points_after_first_save), (
            f"Point count for {docname} changed across repeated Desk saves "
            f"({len(points_after_first_save)} -> {len(points_after_second_save)}) "
            "-- expected idempotent upsert, not duplicate points"
        )
    finally:
        _cancel(docname)
