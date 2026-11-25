"""Route-level authentication during the single-JWT migration."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Header, HTTPException, status

from app.core.config import settings
from app.core.gateway_auth import require_gateway_capability
from app.core.service_auth import verify_service_token


def _require_opaque_bearer(
    authorization: str | None = Header(default=None),
) -> None:
    """Legacy Support behavior: require a syntactically valid Bearer value."""
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )


def require_user_capability(capability: str, *, legacy_service_token: bool = True):
    """Create a dependency that follows the configured migration mode.

    Settings are read at request time so staging can switch modes without
    rebuilding the FastAPI route graph.
    """

    gateway_dependency = require_gateway_capability(capability)

    def dependency(
        authorization: Annotated[str | None, Header()] = None,
        principal_context: Annotated[str | None, Header(alias="X-Swabi-Principal")] = None,
        principal_signature: Annotated[
            str | None, Header(alias="X-Swabi-Principal-Signature")
        ] = None,
    ) -> object | None:
        if settings.ai_auth_mode == "gateway":
            return gateway_dependency(
                authorization=authorization,
                principal_context=principal_context,
                principal_signature=principal_signature,
            )
        if legacy_service_token:
            verify_service_token(authorization=authorization)
        else:
            _require_opaque_bearer(authorization=authorization)
        return None

    return Depends(dependency)
