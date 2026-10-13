"""FastAPI dependencies for the Chat Safety route.

Authentication note: this is a **service-to-service** credential. It is
NOT a user JWT. The guardrail token must never be captured into the
user-JWT contextvar used by ``SwabiClient``. See integration plan §2.
"""

from __future__ import annotations

import secrets

from fastapi import Header, HTTPException, status

from app.core.config import settings


def verify_token(authorization: str | None = Header(default=None)) -> None:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = authorization.removeprefix("Bearer ").strip()
    if not secrets.compare_digest(token, settings.guardrail_api_token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )
