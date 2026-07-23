from __future__ import annotations

import hashlib
import logging
import os
import secrets
import time

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import RedirectResponse

from api.auth.jwt_handler import mint_token
from api.auth.oauth2 import build_authorize_url, exchange_code_for_token, fetch_user_roles
from api.auth.pkce import generate_code_verifier

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])

# ERPNext's own OAuth confirmation page (oauth_confirmation.html) binds "Allow" with a
# plain click handler that calls window.location.replace(success_url) — a double-click
# fires it twice, sending two requests for the same code/state before the first
# navigation completes. Cache completed logins briefly so the replay gets the same
# redirect instead of a confusing "invalid state" error after a successful login.
#
# The cache entry is keyed by `state` but also binds the original `code` (hashed) --
# a hit only replays the cached redirect if the request's `code` matches too. Without
# this, anyone who obtains just the `state` value (e.g. from proxy access logs) within
# the TTL window gets served the same valid JWT with an arbitrary `code`, skipping PKCE
# validation entirely. See issue #64.
_OAUTH_COMPLETED_TTL_SECONDS = 60


def _hash_code(code: str) -> str:
    return hashlib.sha256(code.encode()).hexdigest()


def _allowed_roles() -> list[str]:
    raw = os.environ.get(
        "ALLOWED_ROLES",
        "Purchase Manager,Purchase User,Accounts User,System Manager",
    )
    return [r.strip() for r in raw.split(",") if r.strip()]


@router.get("/login")
async def login(request: Request) -> RedirectResponse:
    code_verifier = generate_code_verifier()
    state = secrets.token_urlsafe(16)
    request.app.state.oauth_state[state] = code_verifier
    return RedirectResponse(build_authorize_url(code_verifier=code_verifier, state=state))


@router.get("/callback")
async def callback(request: Request, code: str, state: str) -> RedirectResponse:
    completed: dict[str, tuple[float, str, str]] = request.app.state.oauth_completed
    now = time.monotonic()
    for cached_state, (completed_at, _, _) in list(completed.items()):
        if now - completed_at > _OAUTH_COMPLETED_TTL_SECONDS:
            del completed[cached_state]

    if state in completed:
        _, cached_code_hash, redirect_url = completed[state]
        if secrets.compare_digest(cached_code_hash, _hash_code(code)):
            return RedirectResponse(redirect_url)
        raise HTTPException(status_code=400, detail="Invalid or expired OAuth state")

    code_verifier: str | None = request.app.state.oauth_state.pop(state, None)
    if not code_verifier:
        raise HTTPException(status_code=400, detail="Invalid or expired OAuth state")

    try:
        token_data = await exchange_code_for_token(code=code, code_verifier=code_verifier)
        access_token: str = token_data["access_token"]
        username, roles = await fetch_user_roles(access_token)
    except Exception as exc:
        logger.exception("OAuth token exchange failed")
        raise HTTPException(status_code=502, detail="OAuth exchange failed") from exc

    if not any(r in _allowed_roles() for r in roles):
        raise HTTPException(status_code=403, detail="Access denied — insufficient ERPNext roles")

    jwt_token = mint_token(username=username, roles=roles)
    frontend_url = os.environ.get("FRONTEND_URL", "http://localhost:8501")
    redirect_url = f"{frontend_url}?token={jwt_token}"
    completed[state] = (now, _hash_code(code), redirect_url)
    return RedirectResponse(redirect_url)
