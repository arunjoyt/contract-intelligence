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
    """Return (username, roles) using the ERPNext API.

    Identity is verified via the OAuth access token (openid profile).
    Role lookup uses the server-side API key because the OAuth Bearer token
    does not have permission to read the User DocType via the resource API.
    """
    async with httpx.AsyncClient() as http:
        # Step 1: verify identity with the OAuth token
        profile_resp = await http.get(
            f"{_erpnext_url()}/api/method/frappe.integrations.oauth2.openid_profile",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        profile_resp.raise_for_status()
        # Frappe wraps /api/method/ responses in {"message": <return_value>}
        profile_data = profile_resp.json()
        if "message" in profile_data:
            profile_data = profile_data["message"]
        # "sub" is a pairwise OIDC identifier (hashed, from User Social Login) — not
        # usable for the User.email lookup below. "email" is the real value we need.
        username: str = profile_data.get("email", "") or profile_data.get("sub", "")
        if not username:
            raise ValueError(f"No subject in openid profile response: {profile_data}")

        # Step 2: resolve email → docname (sub is email; User.name may differ e.g. "Administrator")
        api_key = os.environ.get("ERPNEXT_API_KEY", "")
        api_secret = os.environ.get("ERPNEXT_API_SECRET", "")
        admin_headers = {"Authorization": f"token {api_key}:{api_secret}"}

        list_resp = await http.get(
            f"{_erpnext_url()}/api/resource/User",
            params={
                "filters": f'[["email","=","{username}"]]',
                "fields": '["name"]',
                "limit": "1",
            },
            headers=admin_headers,
        )
        list_resp.raise_for_status()
        users = list_resp.json().get("data", [])
        if not users:
            raise ValueError(f"No ERPNext user found with email {username!r}")
        docname = users[0]["name"]

        # Step 3: fetch full user document to get roles
        user_resp = await http.get(
            f"{_erpnext_url()}/api/resource/User/{docname}",
            params={"fields": '["name","roles"]'},
            headers=admin_headers,
        )
        user_resp.raise_for_status()
        role_rows: list[dict] = user_resp.json().get("data", {}).get("roles", [])
        roles = [r["role"] for r in role_rows if r.get("role")]
        return docname, roles
