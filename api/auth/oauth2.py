from __future__ import annotations

import os
from urllib.parse import urlencode

import httpx

from .pkce import generate_code_challenge


def _erpnext_url() -> str:
    return os.environ["ERPNEXT_URL"].rstrip("/")


def _client_id() -> str:
    return os.environ["ERPNEXT_OAUTH_CLIENT_ID"]


def _client_secret() -> str:
    return os.environ["ERPNEXT_OAUTH_CLIENT_SECRET"]


def _redirect_uri() -> str:
    return os.environ["OAUTH_REDIRECT_URI"]


def build_authorize_url(code_verifier: str, state: str) -> str:
    params = {
        "client_id": _client_id(),
        "redirect_uri": _redirect_uri(),
        "response_type": "code",
        "scope": "openid all",
        "state": state,
        "code_challenge": generate_code_challenge(code_verifier),
        "code_challenge_method": "S256",
    }
    return f"{_erpnext_url()}/api/method/frappe.integrations.oauth2.authorize?{urlencode(params)}"


async def exchange_code_for_token(code: str, code_verifier: str) -> dict:
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{_erpnext_url()}/api/method/frappe.integrations.oauth2.get_token",
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": _redirect_uri(),
                "client_id": _client_id(),
                "client_secret": _client_secret(),
                "code_verifier": code_verifier,
            },
        )
        resp.raise_for_status()
        return resp.json()


async def fetch_user_roles(access_token: str) -> tuple[str, list[str]]:
    """Return (username, roles) using the ERPNext API with the OAuth access token."""
    headers = {"Authorization": f"Bearer {access_token}"}
    async with httpx.AsyncClient() as http:
        profile_resp = await http.get(
            f"{_erpnext_url()}/api/method/frappe.integrations.oauth2.openid_profile",
            headers=headers,
        )
        profile_resp.raise_for_status()
        username: str = profile_resp.json().get("sub", "")
        if not username:
            raise ValueError("No subject in openid profile response")

        user_resp = await http.get(
            f"{_erpnext_url()}/api/resource/User/{username}",
            params={"fields": '["name","roles"]'},
            headers=headers,
        )
        user_resp.raise_for_status()
        role_rows: list[dict] = user_resp.json().get("data", {}).get("roles", [])
        roles = [r["role"] for r in role_rows if r.get("role")]
        return username, roles
