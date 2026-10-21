"""RAG support route. Thin: authenticate, invoke service, serialize."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.capabilities.support.schemas import (
    SupportAskRequest,
    SupportAskResponse,
)
from app.capabilities.support.service import ask
from app.core.config import settings
from app.core.service_auth import verify_service_token

router = APIRouter(prefix="/api/v1/support", tags=["support"])


@router.post(
    "/ask",
    response_model=SupportAskResponse,
    dependencies=[Depends(verify_service_token)],
)
def support_ask(
    payload: SupportAskRequest,
    request: Request,
) -> SupportAskResponse:
    if not settings.enable_rag:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG support is not enabled",
        )

    result = ask(payload)
    result.request_id = getattr(request.state, "request_id", "")
    return result
