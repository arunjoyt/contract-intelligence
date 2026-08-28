"""Real browser E2E test for Streamlit sidebar filter widgets.

Unlike tests/test_streamlit.py's streamlit.testing.v1.AppTest (in-process, no
browser, no running server, httpx.post mocked), this drives an actual running
Streamlit server through a real Chromium browser and hits the real FastAPI
backend. Basic chat rendering (message structure, Sources expander, a real
answer) is covered more rigorously by test_erpnext_to_streamlit_loop.py's
full-loop tests, which use fresh, distinctive content instead of whatever
happens to already be in the demo data -- this file only covers what those
don't: interacting with the sidebar filter widgets themselves.

Login is bypassed by minting a JWT directly (see tests/e2e/conftest.py's
`mint_test_jwt`) and navigating to `{FRONTEND_URL}/?token=<jwt>`, rather than
driving ERPNext's OAuth consent flow through the browser -- that handshake is
already covered by conftest.py's `desk_login_state` fixture and by
tests/test_integration.py's auth group.

Note on filter assertions: docs/ARCHITECTURE.md's "Filter behaviour" section
documents that metadata filters only apply to the Qdrant vector-search leg of
hybrid search -- BM25 is corpus-wide, so a supplier filter does not guarantee
every returned source matches that supplier (confirmed empirically while
writing this test: filtering by a real demo supplier still returned sources
from other suppliers for a generic question). The test below therefore only
asserts the filtered request completes and renders normally, not that results
are strictly scoped to the filter -- asserting the latter would be testing
already-documented, accepted behavior, not a regression.
"""

from __future__ import annotations

from tests.e2e.conftest import FRONTEND_URL, live, mint_test_jwt


@live
def test_streamlit_sidebar_filters_do_not_break_query(page) -> None:
    """Sidebar filter widgets (supplier + doctype) should reach the backend
    without erroring -- see module docstring for why this doesn't assert on
    exact result scoping."""
    page.goto(f"{FRONTEND_URL}/?token={mint_test_jwt('e2e-frontend-test')}")

    sidebar = page.locator('[data-testid="stSidebar"]')
    sidebar.get_by_placeholder("e.g. Acme Corp").fill("Zuckerman Security Ltd.")
    sidebar.get_by_role("combobox").first.click()
    page.get_by_text("Contract", exact=True).click()
    page.keyboard.press("Escape")  # close the multiselect dropdown

    chat_input = page.get_by_placeholder("Ask a contract question...")
    chat_input.fill("What are the payment terms?")
    chat_input.press("Enter")

    # Poll for the *final* answer, not just non-empty text -- the assistant
    # container renders immediately with st.spinner("Searching contract
    # documents...")'s own text while waiting, which is non-empty and was
    # found (live) to trip a naive "wait for non-empty" check before the real
    # answer arrived. Can't rely on the Sources expander appearing either -- a
    # filter combination narrow enough to return zero sources is a legitimate,
    # non-error outcome here.
    messages = page.locator('[data-testid="stChatMessage"]')
    answer_text = ""
    for _ in range(30):
        answer_text = messages.nth(1).inner_text()
        if answer_text.strip() and "Searching contract documents" not in answer_text:
            break
        page.wait_for_timeout(2000)

    assert answer_text.strip(), "Assistant message rendered empty with filters applied"
    assert "Error connecting to API" not in answer_text
