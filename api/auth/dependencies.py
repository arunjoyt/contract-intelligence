from __future__ import annotations

import os

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .jwt_handler import decode_token

_bearer = HTTPBearer(auto_error=False)


def _allowed_roles() -> list[str]:
    raw = os.environ.get(
        "ALLOWED_ROLES",
        "Purchase Manager,Purchase User,Accounts User,System Manager",
    )
    return [r.strip() for r in raw.split(",") if r.strip()]


def get_current_user(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),  # noqa: B008
) -> dict:
    if creds is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        return decode_token(creds.credentials)
    except jwt.ExpiredSignatureError as exc:
        raise HTTPException(status_code=401, detail="Token expired") from exc
    except jwt.InvalidTokenError as exc:
        raise HTTPException(status_code=401, detail="Invalid token") from exc


def require_allowed_role(user: dict = Depends(get_current_user)) -> dict:  # noqa: B008
    if not any(r in _allowed_roles() for r in user.get("roles", [])):
        raise HTTPException(status_code=403, detail="Access denied")
    return user
