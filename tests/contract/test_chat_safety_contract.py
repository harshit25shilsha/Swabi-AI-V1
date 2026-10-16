"""Public contract test for POST /api/v1/validate-message.

Asserts the exact endpoint path, request fields, response fields, and
ALLOW/BLOCK semantics the NestJS backend relies on. If this fails, the
guardrail's public contract has drifted.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app

client = TestClient(app)
AUTH = {"Authorization": f"Bearer {settings.guardrail_api_token}"}


def test_endpoint_exists_and_accepts_message_only():
    r = client.post(
        "/api/v1/validate-message",
        headers=AUTH,
        json={"message": "Call me at 9876543210"},
    )
    assert r.status_code == 200


def test_optional_metadata_accepted():
    r = client.post(
        "/api/v1/validate-message",
        headers=AUTH,
        json={
            "message": "Call me at 9876543210",
            "message_id": "m1",
            "conversation_id": "c1",
            "sender_id": "s1",
            "sender_role": "user",
        },
    )
    assert r.status_code == 200


def test_response_has_exact_required_fields():
    r = client.post(
        "/api/v1/validate-message",
        headers=AUTH,
        json={"message": "Call me at 9876543210"},
    )
    body = r.json()
    expected = {
        "allowed",
        "action",
        "category",
        "confidence",
        "source",
        "request_id",
        "prompt_version",
        "provider",
    }
    assert set(body.keys()) == expected, (
        f"Response fields drifted. Extra: {set(body) - expected}. Missing: {expected - set(body)}."
    )


def test_block_semantics():
    r = client.post(
        "/api/v1/validate-message",
        headers=AUTH,
        json={"message": "Call me at 9876543210"},
    )
    body = r.json()
    assert body["allowed"] is False
    assert body["action"] == "BLOCK"
    assert body["category"] == "PHONE_NUMBER"
    assert isinstance(body["confidence"], float)


def test_missing_auth_is_401():
    r = client.post(
        "/api/v1/validate-message",
        json={"message": "hello"},
    )
    assert r.status_code == 401


def test_wrong_auth_is_401():
    r = client.post(
        "/api/v1/validate-message",
        headers={"Authorization": "Bearer not-the-right-token"},
        json={"message": "hello"},
    )
    assert r.status_code == 401


def test_request_id_header_present():
    r = client.post(
        "/api/v1/validate-message",
        headers=AUTH,
        json={"message": "Call me at 9876543210"},
    )
    assert "x-request-id" in r.headers
