"""Response shape tests for /api/v1/validate-message."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app

client = TestClient(app)
AUTH = {"Authorization": f"Bearer {settings.guardrail_api_token}"}


def test_response_has_source_request_id_prompt_version_provider():
    r = client.post(
        "/api/v1/validate-message",
        headers=AUTH,
        json={"message": "Call me at 9876543210"},
    )
    body = r.json()
    assert body["source"] == "deterministic"
    assert body["request_id"].startswith("gr_")
    assert body["prompt_version"].startswith("marketplace_")
    assert body["provider"] == ""  # deterministic path leaves provider empty


def test_response_header_has_request_id():
    r = client.post(
        "/api/v1/validate-message",
        headers=AUTH,
        json={"message": "Call me at 9876543210"},
    )
    assert r.headers["x-request-id"].startswith("gr_")


def test_request_id_propagated_from_client_header():
    r = client.post(
        "/api/v1/validate-message",
        headers={**AUTH, "X-Request-ID": "my-trace-123"},
        json={"message": "Call me at 9876543210"},
    )
    assert r.json()["request_id"] == "my-trace-123"
    assert r.headers["x-request-id"] == "my-trace-123"
