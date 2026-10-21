"""Contract lock for POST /api/v1/support/ask."""
from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

from app.capabilities.support.schemas import (
    AnswerType,
    SupportAskResponse,
)
from app.core.config import settings
from app.main import app

client = TestClient(app)
AUTH = {"Authorization": f"Bearer {settings.guardrail_api_token}"}


def _empty() -> SupportAskResponse:
    return SupportAskResponse(
        answer="",
        answer_type=AnswerType.INSUFFICIENT_CONTEXT,
        knowledge_version="",
    )


def test_endpoint_at_exact_path(monkeypatch):
    monkeypatch.setattr(settings, "enable_rag", True)
    with patch("app.api.routes.support.ask", return_value=_empty()):
        r = client.post(
            "/api/v1/support/ask",
            headers=AUTH,
            json={"question": "hi", "audience": "customer"},
        )
    assert r.status_code == 200


def test_response_field_set_locked(monkeypatch):
    monkeypatch.setattr(settings, "enable_rag", True)
    with patch("app.api.routes.support.ask", return_value=_empty()):
        r = client.post(
            "/api/v1/support/ask",
            headers=AUTH,
            json={"question": "hi", "audience": "customer"},
        )
    expected = {
        "answer", "answer_type", "citations", "confidence",
        "retrieval_mode_used", "knowledge_version",
        "request_id", "provider", "prompt_version",
    }
    assert set(r.json().keys()) == expected


def test_answer_type_values(monkeypatch):
    monkeypatch.setattr(settings, "enable_rag", True)
    with patch("app.api.routes.support.ask", return_value=_empty()):
        r = client.post(
            "/api/v1/support/ask",
            headers=AUTH,
            json={"question": "hi", "audience": "customer"},
        )
    assert r.json()["answer_type"] in {
        "answered", "insufficient_context", "needs_live_data", "out_of_scope",
    }


def test_auth_required():
    r = client.post(
        "/api/v1/support/ask",
        json={"question": "hi", "audience": "customer"},
    )
    assert r.status_code == 401
