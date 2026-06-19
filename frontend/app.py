"""Streamlit procurement intelligence chat UI.

Calls POST {BACKEND_URL}/query and renders the answer with a collapsible
Sources expander.  Sidebar filters are merged into the request payload.
"""

from __future__ import annotations

import os

import httpx
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
_MAX_HISTORY = 10
_DOCTYPES = ["Purchase Order", "Purchase Invoice", "Contract", "Supplier Scorecard"]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _render_sources(sources: list[dict]) -> None:
    with st.expander(f"Sources ({len(sources)})"):
        for s in sources:
            line = f"**{s['docname']}** &nbsp;·&nbsp; {s['source_doctype']}"
            if s.get("supplier"):
                line += f" &nbsp;·&nbsp; {s['supplier']}"
            st.markdown(line)


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Procurement Intelligence", layout="wide")
st.title("Procurement Intelligence")

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Filters")
    supplier = st.text_input("Supplier", placeholder="e.g. Acme Corp")
    doctypes = st.multiselect("Document type", _DOCTYPES)
    st.markdown("**Date range**")
    col_from, col_to = st.columns(2)
    with col_from:
        start_date = st.date_input("From", value=None, label_visibility="collapsed")
    with col_to:
        end_date = st.date_input("To", value=None, label_visibility="collapsed")
    status = st.selectbox(
        "Status",
        ["", "Draft", "Submitted", "Cancelled", "Active"],
        format_func=lambda x: x or "Any",
    )

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages: list[dict] = []

for msg in st.session_state.messages[-_MAX_HISTORY:]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            _render_sources(msg["sources"])

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------

if question := st.chat_input("Ask a procurement question..."):
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    filters: dict = {}
    if supplier:
        filters["supplier"] = supplier
    if doctypes:
        filters["source_doctype"] = doctypes[0] if len(doctypes) == 1 else doctypes
    if start_date:
        filters["start_date"] = start_date.isoformat()
    if end_date:
        filters["end_date"] = end_date.isoformat()
    if status:
        filters["status"] = status

    with st.chat_message("assistant"):
        with st.spinner("Searching procurement documents…"):
            try:
                resp = httpx.post(
                    f"{BACKEND_URL}/query",
                    json={"question": question, "filters": filters or None},
                    timeout=90.0,
                )
                resp.raise_for_status()
                data = resp.json()
                answer: str = data["answer"]
                sources: list[dict] = data.get("sources", [])
            except httpx.HTTPStatusError as exc:
                answer = f"API error {exc.response.status_code}: {exc.response.text}"
                sources = []
            except Exception as exc:
                answer = f"Error connecting to API: {exc}"
                sources = []

        st.markdown(answer)
        if sources:
            _render_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
