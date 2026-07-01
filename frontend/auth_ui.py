"""OAuth2 login/logout UI helpers for the Streamlit frontend."""

from __future__ import annotations

import os

import streamlit as st

_BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
# PUBLIC_API_URL is the externally reachable FastAPI URL used for browser redirects.
# Defaults to BACKEND_URL so local dev (where both are localhost:8000) works without
# extra config; must be set to the public domain in production (behind nginx).
_PUBLIC_API_URL = os.environ.get("PUBLIC_API_URL", _BACKEND_URL)


def handle_token_from_url() -> None:
    """Store JWT from ?token= query param (set by /auth/callback) in session state."""
    token = st.query_params.get("token")
    if token:
        st.session_state.jwt = token
        st.query_params.clear()


def show_login_page() -> None:
    st.title("Procurement Intelligence")
    st.markdown("Sign in with your ERPNext account to access procurement data.")
    login_url = f"{_PUBLIC_API_URL}/auth/login"
    st.markdown(
        f'<a href="{login_url}" target="_self" style="text-decoration:none;">'
        '<button style="padding:0.5rem 1.5rem;font-size:1rem;cursor:pointer;'
        'border:1px solid #ccc;border-radius:4px;background:#fff;color:#000;">'
        "Login with ERPNext"
        "</button></a>",
        unsafe_allow_html=True,
    )


def show_logout_button() -> None:
    if st.sidebar.button("Logout"):
        del st.session_state.jwt
        st.rerun()
