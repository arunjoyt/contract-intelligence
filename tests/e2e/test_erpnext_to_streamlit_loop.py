"""Full-loop E2E: an ERPNext Desk UI action is reflected through the real
Streamlit UI, not just in Qdrant.

Consolidates what used to be three separate files (test_erpnext_desk_submit.py,
test_erpnext_desk_cancel.py, and a generic test_streamlit_frontend.py chat
test) into full-loop tests that verify the *user-observable* outcome, not an
internal implementation detail (a raw Qdrant `points/scroll` call). Each test
still drives the real ERPNext **Desk UI** (not REST) for the action under
test -- `frappe.client.submit`/`cancel` via REST bypasses the Frappe
background worker queue that fires webhooks on the real Desk path, which is
exactly the class of bug this suite exists to catch (see
docs/IMPLEMENTATION_PLAN.md Step 17) -- then verifies the result by asking
about it through the real Streamlit browser UI, the same path an actual user
takes.

Not consolidated here, on purpose:
- test_webhook_config.py: a static config check (do these Webhook records
  exist/enabled) with no ERPNext data and nothing to "loop" through the UI.
- test_erpnext_desk_update_after_submit.py: relies on precise checks (Frappe's
  Webhook Request Log, exact Qdrant payload values) to conclusively prove a
  confirmed platform gap. Routing that through an LLM-composed UI answer would
  trade the precision that caught the gap for flakiness, on the one test
  where precision is the entire point.

Each test plants a distinctive, made-up fact (a random tracking code that
can't already exist anywhere in the corpus) so retrieval and the assertion
are unambiguous -- not dependent on demo-data specifics that could change.
"""

from __future__ import annotations

import uuid

from tests.e2e.conftest import (
    ERPNEXT_URL,
    FRONTEND_URL,
    attach_test_pdf,
    cleanup_contract,
    click_primary_action,
    create_draft_contract,
    live,
    mint_test_jwt,
    poll_qdrant,
    submit_contract_via_rest,
    wait_for_docstatus_submitted,
    wait_for_indicator,
)


def _ask_and_wait_for_answer(page, question: str) -> str:
    """Navigate to the Streamlit frontend, ask `question`, and return the
    rendered assistant answer text once the real backend response lands.

    Polls for the *final* answer, not just non-empty text -- the assistant
    container renders immediately with st.spinner("Searching contract
    documents...")'s own (non-empty) text while waiting, which was found, via
    a live run, to trip a naive "wait for non-empty" check before the real
    answer arrived.
    """
    page.goto(f"{FRONTEND_URL}/?token={mint_test_jwt('e2e-loop-test')}")
    chat_input = page.get_by_placeholder("Ask a contract question...")
    chat_input.wait_for(timeout=15000)
    chat_input.fill(question)
    chat_input.press("Enter")

    messages = page.locator('[data-testid="stChatMessage"]')
    answer_text = ""
    for _ in range(30):
        answer_text = messages.nth(1).inner_text()
        if answer_text.strip() and "Searching contract documents" not in answer_text:
            break
        page.wait_for_timeout(2000)
    return answer_text


@live
def test_contract_desk_submitted_content_is_answerable_via_streamlit_ui(desk_page, page) -> None:
    tracking_code = f"TRACKING-CODE-{uuid.uuid4().hex[:12].upper()}"
    docname, _ = create_draft_contract(
        contract_terms=(
            f"<p>This agreement includes a unique tracking clause: {tracking_code}. "
            "This code must be referenced in all correspondence regarding this "
            "contract.</p>"
        )
    )
    try:
        desk_page.goto(f"{ERPNEXT_URL}/app/contract/{docname}")
        click_primary_action(desk_page, "Submit")
        wait_for_docstatus_submitted(desk_page)

        # Confirms the content landed in Qdrant; the webhook handler rebuilds
        # the BM25 index synchronously before returning, so by the time
        # points are visible here BM25 is already current too.
        poll_qdrant(docname, predicate=lambda pts: len(pts) > 0)

        answer_text = _ask_and_wait_for_answer(
            page,
            f"What is the tracking code mentioned in contract {docname}? "
            "It starts with TRACKING-CODE.",
        )
        assert tracking_code in answer_text, (
            f"Streamlit UI answer did not surface the freshly-indexed tracking "
            f"code {tracking_code!r} for {docname} after a Desk submit.\n"
            f"Answer was: {answer_text!r}"
        )
    finally:
        cleanup_contract(docname)


@live
def test_contract_desk_submitted_pdf_content_is_answerable_via_streamlit_ui(
    desk_page, page
) -> None:
    tracking_code = f"PDF-CODE-{uuid.uuid4().hex[:12].upper()}"
    docname, _ = create_draft_contract()
    try:
        attach_test_pdf(
            docname,
            paragraphs=20,
            line=f"This attachment's unique reference is {tracking_code}. ",
        )

        desk_page.goto(f"{ERPNEXT_URL}/app/contract/{docname}")
        click_primary_action(desk_page, "Submit")
        wait_for_docstatus_submitted(desk_page)

        # >=2 chunks proves the attached PDF's text was actually extracted and
        # indexed, not just the Contract's own (short) HTML fields.
        poll_qdrant(docname, predicate=lambda pts: len(pts) >= 2)

        answer_text = _ask_and_wait_for_answer(
            page,
            f"What is the unique reference code in the PDF attached to contract "
            f"{docname}? It starts with PDF-CODE.",
        )
        assert tracking_code in answer_text, (
            f"Streamlit UI answer did not surface the freshly-indexed PDF "
            f"attachment code {tracking_code!r} for {docname} after a Desk submit.\n"
            f"Answer was: {answer_text!r}"
        )
    finally:
        cleanup_contract(docname)


@live
def test_contract_desk_cancel_status_is_reflected_in_streamlit_ui(desk_page, page) -> None:
    # A bare docname isn't enough to retrieve this reliably: BM25 only indexes
    # chunk *text* (see retrieval/hybrid_search.py's _build_index), not
    # docname, so a docname-anchored question has no lexical anchor into a
    # corpus of dozens of other, more topically rich contracts (found live --
    # the first version of this test, which asked about the bare docname,
    # retrieved nothing and got a "could not find relevant information"
    # refusal). Embedding a distinctive phrase in the text itself, the same
    # technique the submit/PDF tests above use, gives retrieval a real anchor.
    tracking_code = f"CANCEL-CODE-{uuid.uuid4().hex[:12].upper()}"
    docname, _ = create_draft_contract(
        contract_terms=f"<p>Unique reference for this agreement: {tracking_code}.</p>"
    )
    try:
        submit_contract_via_rest(docname)  # get to submitted state; Cancel is under test
        poll_qdrant(docname, predicate=lambda pts: len(pts) > 0)

        desk_page.goto(f"{ERPNEXT_URL}/app/contract/{docname}")
        click_primary_action(desk_page, "Cancel")
        wait_for_indicator(desk_page, "Cancelled")

        # webhook_handler.py is event-agnostic: on_cancel re-indexes with
        # status overridden to "Cancelled" rather than deleting the points
        # (see docs/DEPLOYMENT.md "How modifications are handled").
        poll_qdrant(
            docname,
            predicate=lambda pts: any(p["payload"].get("status") == "Cancelled" for p in pts),
        )

        answer_text = _ask_and_wait_for_answer(
            page,
            f"What is the current status of the contract with reference code "
            f"{tracking_code}?",
        )
        assert "cancel" in answer_text.lower(), (
            f"Streamlit UI answer did not reflect the Cancelled status for {docname} "
            f"after a Desk cancel.\nAnswer was: {answer_text!r}"
        )
    finally:
        cleanup_contract(docname)  # no-op cancel if already cancelled
