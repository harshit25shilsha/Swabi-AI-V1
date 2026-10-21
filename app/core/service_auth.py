"""Shared service-to-service authentication dependency.

Any route called by another backend service (not by a logged-in user)
uses ``verify_service_token``.

The token lives in ``Settings.guardrail_api_token``. The name is
historical — the token was introduced when the Guardrail Service was
the only service-to-service route. It is now the general inbound
credential for the AI service. Renaming it to ``ai_service_token`` is
planned for a later phase; do not change it today without coordinating
with the NestJS team, which sends this value.
"""

from __future__ import annotations

import secrets

from fastapi import Header, HTTPException, status

from app.core.config import settings


def verify_service_token(
    authorization: str | None = Header(default=None),
) -> None:
    """FastAPI dependency: check the inbound service token."""
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
