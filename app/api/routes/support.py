"""RAG support route. Called by the frontend; user JWT is the only credential.

Frontend sends:
    Authorization: Bearer <user-jwt>

The AI service does NOT accept a service token on this route. The user
JWT is forwarded unchanged to the Swabi NestJS backend on any live-data
call inside the service.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.capabilities.support.schemas import (
    SupportAskRequest,
    SupportAskResponse,
)
from app.capabilities.support.service import ask
from app.core.config import settings
from app.core.route_auth import require_user_capability

router = APIRouter(prefix="/api/v1/support", tags=["support"])

# Register a Bearer security scheme rather than a plain header parameter so
# Swagger UI can use its Authorize dialog and include the token automatically.
user_bearer = HTTPBearer(
    auto_error=False,
    scheme_name="UserBearerAuth",
    description="User JWT forwarded to the Swabi backend for live-data requests.",
)


@router.post("/ask", response_model=SupportAskResponse)
def support_ask(
    payload: SupportAskRequest,
    request: Request,
    credentials: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(user_bearer),
    ],
    _principal: Annotated[
        object | None,
        require_user_capability("ai:support:ask", legacy_service_token=False),
    ] = None,
) -> SupportAskResponse:
    if not settings.enable_rag:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG support is not enabled",
        )

    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_jwt = credentials.credentials.strip()
    if not user_jwt:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Empty bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # In gateway mode the authoritative audience is in the verified principal
    # assertion.  The request body remains temporarily compatible but cannot
    # select a different knowledge scope.
    if (
        _principal is not None
        and getattr(_principal, "audience", payload.audience) != payload.audience
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Requested audience is not authorized",
        )

    result = ask(payload, user_jwt=user_jwt)
    result.request_id = getattr(request.state, "request_id", "")
    return result
