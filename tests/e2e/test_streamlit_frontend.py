"""Real browser E2E tests for the Streamlit frontend.

Unlike tests/test_streamlit.py's streamlit.testing.v1.AppTest (in-process, no
browser, no running server, httpx.post mocked), these tests drive an actual
running Streamlit server through a real Chromium browser and hit the real
FastAPI backend -- the full browser -> Streamlit -> FastAPI -> pipeline ->
Qdrant -> OpenAI -> browser round trip, with nothing mocked.

Login is bypassed by minting a JWT directly (same `mint_token` the real
/auth/callback uses after a successful ERPNext OAuth exchange) and navigating
to `{FRONTEND_URL}/?token=<jwt>` -- `handle_token_from_url()` in
frontend/auth_ui.py picks it up exactly as it would from a real OAuth redirect.
Driving the actual ERPNext OAuth consent flow through the browser is out of
scope here; that handshake is already covered by tests/e2e/conftest.py's
`desk_login_state` fixture logging into the Desk UI, and by
tests/test_integration.py's auth group hitting /auth/callback directly.

Note on filter assertions: docs/ARCHITECTURE.md's "Filter behaviour" section
documents that metadata filters only apply to the Qdrant vector-search leg of
hybrid search -- BM25 is corpus-wide, so a supplier filter does not guarantee
every returned source matches that supplier (confirmed empirically while
writing this test: filtering by a real demo supplier still returned sources
from other suppliers for a generic question). The filter test below therefore
only asserts the filtered request completes and renders normally, not that
results are strictly scoped to the filter -- asserting the latter would be
testing already-documented, accepted behavior, not a regression.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

from api.auth.jwt_handler import mint_token
from tests.e2e.conftest import live

load_dotenv()

FRONTEND_URL = os.environ.get("FRONTEND_URL", "http://localhost:8501")


def _mint_test_jwt() -> str:
    return mint_token(username="e2e-frontend-test", roles=["System Manager"])


@live
def test_streamlit_login_via_token_and_chat_returns_answer(page) -> None:
    page.goto(f"{FRONTEND_URL}/?token={_mint_test_jwt()}")

    chat_input = page.get_by_placeholder("Ask a contract question...")
    chat_input.wait_for(timeout=15000)
    chat_input.fill("What are the payment terms for our active contracts?")
    chat_input.press("Enter")

    expander = page.locator('[data-testid="stExpander"]')
    expander.wait_for(timeout=60000)

    messages = page.locator('[data-testid="stChatMessage"]')
    assert messages.count() == 2, (
        f"Expected 2 chat messages (user+assistant), got {messages.count()}"
    )
    assert "payment terms" in messages.nth(0).inner_text().lower()

    answer_text = messages.nth(1).inner_text()
    assert answer_text.strip(), "Assistant message rendered empty"
    assert "Error connecting to API" not in answer_text

    expander.click()
    sources_text = expander.inner_text()
    assert "Sources (" in sources_text
    assert sources_text.strip() != "Sources (0)"


@live
def test_streamlit_sidebar_filters_do_not_break_query(page) -> None:
    """Sidebar filter widgets (supplier + doctype) should reach the backend
    without erroring -- see module docstring for why this doesn't assert on
    exact result scoping."""
    page.goto(f"{FRONTEND_URL}/?token={_mint_test_jwt()}")

    sidebar = page.locator('[data-testid="stSidebar"]')
    sidebar.get_by_placeholder("e.g. Acme Corp").fill("Zuckerman Security Ltd.")
    sidebar.get_by_role("combobox").first.click()
    page.get_by_text("Contract", exact=True).click()
    page.keyboard.press("Escape")  # close the multiselect dropdown

    chat_input = page.get_by_placeholder("Ask a contract question...")
    chat_input.fill("What are the payment terms?")
    chat_input.press("Enter")

    # Poll for actual completion, not just the assistant container's presence --
    # Streamlit renders the st.chat_message("assistant") container immediately
    # and fills in content only once the backend responds (found via a live
    # run: waiting on the message locator's mere presence raced ahead of the
    # real answer and produced a false-empty assertion failure). Can't rely on
    # the Sources expander appearing either -- a filter combination narrow
    # enough to return zero sources is a legitimate, non-error outcome here.
    messages = page.locator('[data-testid="stChatMessage"]')
    answer_text = ""
    for _ in range(30):
        answer_text = messages.nth(1).inner_text()
        if answer_text.strip():
            break
        page.wait_for_timeout(2000)

    assert answer_text.strip(), "Assistant message rendered empty with filters applied"
    assert "Error connecting to API" not in answer_text
