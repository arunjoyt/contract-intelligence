from __future__ import annotations

import logging
import os
import secrets

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import RedirectResponse

from api.auth.jwt_handler import mint_token
from api.auth.oauth2 import build_authorize_url, exchange_code_for_token, fetch_user_roles
from api.auth.pkce import generate_code_verifier

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])


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
    return RedirectResponse(f"{frontend_url}?token={jwt_token}")
