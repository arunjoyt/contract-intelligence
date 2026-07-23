"""Desk-driven Contract submit -> webhook -> Qdrant indexing.

Unlike tests/test_integration.py's REST-based submit (`frappe.client.submit`),
clicking Submit in the Desk UI goes through Frappe's real background worker queue
-- the path that can silently fail to fire a webhook even when test_webhook_config.py's
REST checks show the Webhook record is present and enabled.
"""

from __future__ import annotations

from tests.e2e.conftest import (
    ERPNEXT_URL,
    attach_test_pdf,
    cancel_contract_via_rest,
    click_primary_action,
    create_draft_contract,
    live,
    poll_qdrant,
    wait_for_docstatus_submitted,
)


@live
def test_contract_desk_submit_fires_webhook_and_indexes(desk_page) -> None:
    docname, _ = create_draft_contract()
    try:
        desk_page.goto(f"{ERPNEXT_URL}/app/contract/{docname}")
        click_primary_action(desk_page, "Submit")
        wait_for_docstatus_submitted(desk_page)

        points = poll_qdrant(docname, predicate=lambda pts: len(pts) > 0)
        assert points, f"No Qdrant points found for {docname} after Desk submit"
        assert points[0]["payload"]["docname"] == docname
    finally:
        cancel_contract_via_rest(docname)


@live
def test_contract_desk_submit_with_pdf_attachment_indexes_extra_chunks(desk_page) -> None:
    docname, _ = create_draft_contract()
    try:
        # Enough repeated lines to push combined text past chunk_size=512, so a
        # plain contract_terms-only submit (see the other test) chunks to 1 and
        # this one chunks to >=2 -- proving the attached PDF's text was actually
        # extracted and indexed, not just the Contract's own HTML fields.
        attach_test_pdf(docname, paragraphs=20)

        desk_page.goto(f"{ERPNEXT_URL}/app/contract/{docname}")
        click_primary_action(desk_page, "Submit")
        wait_for_docstatus_submitted(desk_page)

        points = poll_qdrant(docname, predicate=lambda pts: len(pts) >= 2)
        assert len(points) >= 2, (
            f"Expected >=2 chunks for {docname} with a PDF attachment, got {len(points)}"
        )
    finally:
        cancel_contract_via_rest(docname)
