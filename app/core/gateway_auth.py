"""Trusted NestJS gateway assertion verification.

The public caller's JWT remains opaque to this service.  NestJS validates it,
then sends the original Bearer token plus a signed, short-lived assertion over
the private gateway-to-AI connection.  The assertion authorizes the AI route
and supplies the trusted audience used to scope knowledge retrieval.

The headers below are an internal contract.  They must be stripped from all
public ingress traffic and sent only over a protected channel (mTLS and
private networking).  The HMAC provides an additional integrity check.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from typing import Annotated

from fastapi import Header, HTTPException, status

from app.core.auth_context import set_user_id, set_user_jwt
from app.core.config import settings

_CONTEXT_HEADER = "X-Swabi-Principal"
_SIGNATURE_HEADER = "X-Swabi-Principal-Signature"


@dataclass(frozen=True)
class GatewayPrincipal:
    """Minimal, verified user context supplied by the NestJS gateway."""

    subject: str
    audience: str
    scopes: frozenset[str]
    expires_at: int


def _unauthorized(detail: str = "Invalid authentication context") -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def _decode_context(value: str, signature: str) -> GatewayPrincipal:
    expected = hmac.new(
        settings.ai_gateway_auth_secret.encode("utf-8"),
        value.encode("ascii"),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(signature, expected):
        raise _unauthorized()

    try:
        padding = "=" * (-len(value) % 4)
        raw = base64.urlsafe_b64decode((value + padding).encode("ascii"))
        payload = json.loads(raw)
    except (UnicodeEncodeError, ValueError, json.JSONDecodeError) as exc:
        raise _unauthorized() from exc

    if not isinstance(payload, dict):
        raise _unauthorized()

    subject = payload.get("sub")
    audience = payload.get("audience")
    scopes = payload.get("scopes")
    expires_at = payload.get("exp")
    if (
        not isinstance(subject, str)
        or not subject
        or audience not in {"customer", "vendor"}
        or not isinstance(scopes, list)
        or not all(isinstance(scope, str) and scope for scope in scopes)
        or not isinstance(expires_at, int)
        or expires_at <= time.time()
    ):
        raise _unauthorized()

    return GatewayPrincipal(
        subject=subject,
        audience=audience,
        scopes=frozenset(scopes),
        expires_at=expires_at,
    )


def require_gateway_capability(capability: str):
    """Return a FastAPI dependency requiring an authorized user capability."""

    def dependency(
        authorization: Annotated[str | None, Header()] = None,
        principal_context: Annotated[str | None, Header(alias=_CONTEXT_HEADER)] = None,
        principal_signature: Annotated[str | None, Header(alias=_SIGNATURE_HEADER)] = None,
    ) -> GatewayPrincipal:
        if not authorization:
            raise _unauthorized("Missing Authorization header")
        scheme, _, token = authorization.partition(" ")
        token = token.strip()
        if scheme.lower() != "bearer" or not token:
            raise _unauthorized("Invalid Authorization header")
        if not principal_context or not principal_signature:
            raise _unauthorized("Missing gateway authentication context")

        principal = _decode_context(principal_context, principal_signature)
        if capability not in principal.scopes:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient capability",
            )

        # The JWT is kept opaque and is only used for delegated live API calls.
        set_user_jwt(token)
        set_user_id(principal.subject)
        return principal

    return dependency
