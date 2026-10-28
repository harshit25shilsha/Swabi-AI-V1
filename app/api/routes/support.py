"""RAG support route. Called by the frontend; user JWT is the only credential.

Frontend sends:
    Authorization: Bearer <user-jwt>

The AI service does NOT accept a service token on this route. The user
JWT is forwarded unchanged to the Swabi NestJS backend on any live-data
call inside the service.
"""
from __future__ import annotations

from fastapi import APIRouter, Header, HTTPException, Request, status

from app.capabilities.support.schemas import (
    SupportAskRequest,
    SupportAskResponse,
)
from app.capabilities.support.service import ask
from app.core.config import settings

router = APIRouter(prefix="/api/v1/support", tags=["support"])


@router.post("/ask", response_model=SupportAskResponse)
def support_ask(
    payload: SupportAskRequest,
    request: Request,
    authorization: str | None = Header(default=None),
) -> SupportAskResponse:
    if not settings.enable_rag:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG support is not enabled",
        )

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_jwt = authorization.removeprefix("Bearer ").strip()
    if not user_jwt:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Empty bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    result = ask(payload, user_jwt=user_jwt)
    result.request_id = getattr(request.state, "request_id", "")
    return result
