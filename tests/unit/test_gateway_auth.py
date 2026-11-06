"""Tests for the NestJS-to-AI trusted-principal contract."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time

import pytest
from fastapi import HTTPException

from app.core.config import settings
from app.core.gateway_auth import require_gateway_capability
from app.core.route_auth import require_user_capability


def _assertion(*, scopes: list[str], audience: str = "customer", exp: int | None = None):
    payload = {
        "sub": "user-123",
        "audience": audience,
        "scopes": scopes,
        "exp": exp if exp is not None else int(time.time()) + 60,
    }
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    signature = hmac.new(
        settings.ai_gateway_auth_secret.encode(), encoded.encode(), hashlib.sha256
    ).hexdigest()
    return encoded, signature


@pytest.fixture(autouse=True)
def gateway_secret(monkeypatch):
    monkeypatch.setattr(settings, "ai_gateway_auth_secret", "test-gateway-secret")


def test_valid_assertion_accepts_authorized_capability():
    context, signature = _assertion(scopes=["ai:support:ask"])
    principal = require_gateway_capability("ai:support:ask")(
        authorization="Bearer opaque-user-jwt",
        principal_context=context,
        principal_signature=signature,
    )
    assert principal.subject == "user-123"
    assert principal.audience == "customer"


def test_expired_assertion_is_rejected():
    context, signature = _assertion(scopes=["ai:support:ask"], exp=int(time.time()) - 1)
    with pytest.raises(HTTPException) as exc:
        require_gateway_capability("ai:support:ask")(
            authorization="Bearer opaque-user-jwt",
            principal_context=context,
            principal_signature=signature,
        )
    assert exc.value.status_code == 401


def test_missing_capability_is_forbidden():
    context, signature = _assertion(scopes=["ai:chat-safety"])
    with pytest.raises(HTTPException) as exc:
        require_gateway_capability("ai:support:ask")(
            authorization="Bearer opaque-user-jwt",
            principal_context=context,
            principal_signature=signature,
        )
    assert exc.value.status_code == 403


def test_tampered_assertion_is_rejected():
    context, signature = _assertion(scopes=["ai:support:ask"])
    with pytest.raises(HTTPException) as exc:
        require_gateway_capability("ai:support:ask")(
            authorization="Bearer opaque-user-jwt",
            principal_context=context + "x",
            principal_signature=signature,
        )
    assert exc.value.status_code == 401


def test_route_dependency_enforces_gateway_mode(monkeypatch):
    monkeypatch.setattr(settings, "ai_auth_mode", "gateway")
    context, signature = _assertion(scopes=["ai:support:ask"])
    dependency = require_user_capability("ai:support:ask").dependency
    principal = dependency(
        authorization="Bearer opaque-user-jwt",
        principal_context=context,
        principal_signature=signature,
    )
    assert principal is not None
