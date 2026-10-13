"""Auth-boundary regression tests (integration plan §2)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.capabilities.chat_safety.schemas import Action, ModerationResponse
from app.core.config import settings
from app.main import USER_JWT_ROUTE_PREFIXES, app
from app.observability import counters

client = TestClient(app)
AUTH = {"Authorization": f"Bearer {settings.guardrail_api_token}"}


def test_guardrail_route_not_in_user_jwt_allow_list():
    """Sanity check: the guardrail route must be excluded from JWT capture."""
    for prefix in USER_JWT_ROUTE_PREFIXES:
        assert not "/api/v1/validate-message".startswith(prefix)


def test_guardrail_route_does_not_capture_user_jwt(monkeypatch):
    """A Bearer token on the guardrail route must not populate user_jwt_var."""
    captured: dict[str, str | None] = {}

    def fake_moderate(message: str) -> ModerationResponse:
        from app.core.auth_context import get_user_jwt

        captured["jwt"] = get_user_jwt()
        return ModerationResponse(
            allowed=True,
            action=Action.ALLOW,
            category=None,
            confidence=0.5,
            source="test",
        )

    monkeypatch.setattr("app.api.routes.chat_safety.moderate", fake_moderate)
    counters.reset()

    client.post(
        "/api/v1/validate-message",
        headers={"Authorization": "Bearer guardrail-token-value"},
        json={"message": "hello"},
    )

    assert captured.get("jwt") is None


def test_non_bearer_scheme_is_rejected():
    r = client.post(
        "/api/v1/validate-message",
        headers={"Authorization": "Basic dXNlcjpwYXNz"},
        json={"message": "hi"},
    )
    assert r.status_code == 401
