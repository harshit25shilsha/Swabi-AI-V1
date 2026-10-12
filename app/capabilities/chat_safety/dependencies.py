"""FastAPI dependencies for the Chat Safety route.

Lives next to the capability so the route file stays thin.

Authentication note: this is a **service-to-service** credential. It is
NOT a user JWT. The guardrail token must never be captured into the
user-JWT contextvar used by ``SwabiClient``. See integration plan §2.
"""
from __future__ import annotations

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
    if token != settings.GUARDRAIL_API_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )
