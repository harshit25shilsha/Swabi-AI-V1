"""Integration tests for POST /api/v1/validate-message."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.capabilities.chat_safety.schemas import LLMResult
from app.core.config import settings
from app.main import app
from app.observability import counters

client = TestClient(app)
AUTH = {"Authorization": f"Bearer {settings.guardrail_api_token}"}


@pytest.fixture(autouse=True)
def _reset_counters():
    counters.reset()
    yield
    counters.reset()


def test_missing_token_returns_401():
    r = client.post("/api/v1/validate-message", json={"message": "hi"})
    assert r.status_code == 401


def test_invalid_token_returns_401():
    r = client.post(
        "/api/v1/validate-message",
        headers={"Authorization": "Bearer wrong"},
        json={"message": "hi"},
    )
    assert r.status_code == 401


def test_missing_message_field_returns_422():
    r = client.post("/api/v1/validate-message", headers=AUTH, json={})
    assert r.status_code == 422


def test_empty_message_returns_422():
    r = client.post(
        "/api/v1/validate-message",
        headers=AUTH,
        json={"message": ""},
    )
    assert r.status_code == 422


def test_phone_blocked_via_api():
    r = client.post(
        "/api/v1/validate-message",
        headers=AUTH,
        json={"message": "Call me at 9876543210"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["allowed"] is False
    assert body["action"] == "BLOCK"
    assert body["category"] == "PHONE_NUMBER"
    assert body["source"] == "deterministic"


def test_allow_via_api_uses_llm():
    with patch("app.capabilities.chat_safety.service.moderate_with_llm") as mock_llm:
        mock_llm.return_value = (
            LLMResult(decision="ALLOW", category=None, confidence=0.98),
            "groq",
        )
        r = client.post(
            "/api/v1/validate-message",
            headers=AUTH,
            json={"message": "Is the villa available?"},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["allowed"] is True
    assert body["category"] is None
    assert body["provider"] == "groq"


def test_optional_metadata_accepted():
    r = client.post(
        "/api/v1/validate-message",
        headers=AUTH,
        json={
            "message": "Call me at 9876543210",
            "message_id": "msg_abc123",
            "conversation_id": "conv_456",
            "sender_id": "user_789",
            "sender_role": "user",
        },
    )
    assert r.status_code == 200
