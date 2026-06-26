from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta

import jwt


def _secret() -> str:
    s = os.environ.get("JWT_SECRET", "")
    if not s:
        raise RuntimeError("JWT_SECRET is not set")
    return s


def _expiry_hours() -> int:
    return int(os.environ.get("JWT_EXPIRY_HOURS", "8"))


def mint_token(username: str, roles: list[str]) -> str:
    payload = {
        "sub": username,
        "roles": roles,
        "exp": datetime.now(tz=UTC) + timedelta(hours=_expiry_hours()),
    }
    return jwt.encode(payload, _secret(), algorithm="HS256")


def decode_token(token: str) -> dict:
    return jwt.decode(token, _secret(), algorithms=["HS256"])
