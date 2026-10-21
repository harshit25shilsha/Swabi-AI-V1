"""Route-level tests for POST /api/v1/support/ask."""
from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.capabilities.support.schemas import (
    AnswerType,
    SupportAskResponse,
)
from app.core.config import settings
from app.main import app

client = TestClient(app)
AUTH = {"Authorization": f"Bearer {settings.guardrail_api_token}"}


def _response() -> SupportAskResponse:
    return SupportAskResponse(
        answer="Policy answer.",
        answer_type=AnswerType.ANSWERED,
        knowledge_version="abc123",
    )


@pytest.fixture(autouse=True)
def enable_rag(monkeypatch):
    monkeypatch.setattr(settings, "enable_rag", True)
    yield


def test_missing_token_returns_401(monkeypatch):
    monkeypatch.setattr(settings, "enable_rag", True)
    r = client.post(
        "/api/v1/support/ask",
        json={"question": "hi", "audience": "customer"},
    )
    assert r.status_code == 401


def test_missing_audience_returns_422():
    r = client.post(
        "/api/v1/support/ask",
        headers=AUTH,
        json={"question": "hi"},
    )
    assert r.status_code == 422


def test_empty_question_returns_422():
    r = client.post(
        "/api/v1/support/ask",
        headers=AUTH,
        json={"question": "", "audience": "customer"},
    )
    assert r.status_code == 422


def test_rag_disabled_returns_503(monkeypatch):
    monkeypatch.setattr(settings, "enable_rag", False)
    r = client.post(
        "/api/v1/support/ask",
        headers=AUTH,
        json={"question": "hi", "audience": "customer"},
    )
    assert r.status_code == 503


def test_happy_path():
    with patch("app.api.routes.support.ask", return_value=_response()):
        r = client.post(
            "/api/v1/support/ask",
            headers={**AUTH, "X-Request-ID": "test-trace"},
            json={"question": "policy?", "audience": "customer"},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "Policy answer."
    assert body["answer_type"] == "answered"
    assert body["request_id"] == "test-trace"


def test_response_has_all_fields():
    with patch("app.api.routes.support.ask", return_value=_response()):
        r = client.post(
            "/api/v1/support/ask",
            headers=AUTH,
            json={"question": "policy?", "audience": "customer"},
        )
    body = r.json()
    expected = {
        "answer", "answer_type", "citations", "confidence",
        "retrieval_mode_used", "knowledge_version",
        "request_id", "provider", "prompt_version",
    }
    assert set(body.keys()) == expected
