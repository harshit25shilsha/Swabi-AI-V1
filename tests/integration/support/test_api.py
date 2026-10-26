"""Route tests: user JWT in Authorization is the only credential."""
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


@pytest.fixture(autouse=True)
def enable_rag(monkeypatch):
    monkeypatch.setattr(settings, "enable_rag", True)
    yield


def _response() -> SupportAskResponse:
    return SupportAskResponse(
        answer="Policy answer.",
        answer_type=AnswerType.ANSWERED,
        knowledge_version="abc123",
    )


def test_missing_authorization_returns_401():
    r = client.post(
        "/api/v1/support/ask",
        json={"question": "hi", "audience": "customer"},
    )
    assert r.status_code == 401


def test_empty_bearer_returns_401():
    r = client.post(
        "/api/v1/support/ask",
        headers={"Authorization": "Bearer   "},
        json={"question": "hi", "audience": "customer"},
    )
    assert r.status_code == 401


def test_non_bearer_returns_401():
    r = client.post(
        "/api/v1/support/ask",
        headers={"Authorization": "Basic abc"},
        json={"question": "hi", "audience": "customer"},
    )
    assert r.status_code == 401


def test_missing_audience_returns_422():
    r = client.post(
        "/api/v1/support/ask",
        headers={"Authorization": "Bearer fake-jwt"},
        json={"question": "hi"},
    )
    assert r.status_code == 422


def test_empty_question_returns_422():
    r = client.post(
        "/api/v1/support/ask",
        headers={"Authorization": "Bearer fake-jwt"},
        json={"question": "", "audience": "customer"},
    )
    assert r.status_code == 422


def test_rag_disabled_returns_503(monkeypatch):
    monkeypatch.setattr(settings, "enable_rag", False)
    r = client.post(
        "/api/v1/support/ask",
        headers={"Authorization": "Bearer fake-jwt"},
        json={"question": "hi", "audience": "customer"},
    )
    assert r.status_code == 503


def test_happy_path_forwards_user_jwt():
    """Route must call ask() with the user_jwt extracted from Authorization."""
    with patch(
        "app.api.routes.support.ask",
        return_value=_response(),
    ) as mock_ask:
        r = client.post(
            "/api/v1/support/ask",
            headers={
                "Authorization": "Bearer user-jwt-abc",
                "X-Request-ID": "test-trace",
            },
            json={"question": "policy?", "audience": "customer"},
        )
    assert r.status_code == 200
    assert mock_ask.call_args.kwargs["user_jwt"] == "user-jwt-abc"
    assert r.json()["request_id"] == "test-trace"
