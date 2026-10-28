"""Public contract lock: 12 fields now (was 9)."""
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


def _empty() -> SupportAskResponse:
    return SupportAskResponse(
        answer="", answer_type=AnswerType.INSUFFICIENT_CONTEXT,
        knowledge_version="",
    )


def test_response_field_set_locked(monkeypatch):
    monkeypatch.setattr(settings, "enable_rag", True)
    with patch("app.api.routes.support.ask", return_value=_empty()):
        r = client.post(
            "/api/v1/support/ask",
            headers={"Authorization": "Bearer fake-jwt"},
            json={"question": "hi", "audience": "customer"},
        )
    expected = {
        "answer", "answer_type", "citations", "confidence",
        "retrieval_mode_used", "knowledge_version",
        "request_id", "provider", "prompt_version",
        "live_data_used", "live_data_sources", "live_data_warning",
    }
    assert set(r.json().keys()) == expected


def test_live_data_fields_default_off(monkeypatch):
    monkeypatch.setattr(settings, "enable_rag", True)
    with patch("app.api.routes.support.ask", return_value=_empty()):
        r = client.post(
            "/api/v1/support/ask",
            headers={"Authorization": "Bearer fake-jwt"},
            json={"question": "hi", "audience": "customer"},
        )
    body = r.json()
    assert body["live_data_used"] is False
    assert body["live_data_sources"] == []
    assert body["live_data_warning"] is None


def test_authorization_header_required():
    r = client.post(
        "/api/v1/support/ask",
        json={"question": "hi", "audience": "customer"},
    )
    assert r.status_code == 401
