"""Integration tests for POST /api/v1/enquiry/understand."""

from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

from app.capabilities.enquiry.schemas import (
    ClarificationQuestion,
    EnquiryExtraction,
    EnquiryRequirements,
    EnquiryUnderstandResponse,
)
from app.core.config import settings
from app.main import app

client = TestClient(app)
AUTH = {"Authorization": f"Bearer {settings.guardrail_api_token}"}


def _sample_response() -> EnquiryUnderstandResponse:
    return EnquiryUnderstandResponse(
        extraction=EnquiryExtraction(
            destination="Manali",
            duration_days=5,
            travelers=4,
            description="Family trip to Manali",
            requirements=EnquiryRequirements(accommodation="5-Star Hotel"),
        ),
        missing_fields=["transport", "activities"],
        clarifying_questions=[
            ClarificationQuestion(field="transport", question="What transport would you prefer?"),
        ],
        completeness=0.8,
        provider="groq",
        prompt_version="enquiry_test123",
        request_id="",
    )


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


def test_missing_token_returns_401():
    r = client.post("/api/v1/enquiry/understand", json={"message": "hi"})
    assert r.status_code == 401


def test_wrong_token_returns_401():
    r = client.post(
        "/api/v1/enquiry/understand",
        headers={"Authorization": "Bearer wrong"},
        json={"message": "hi"},
    )
    assert r.status_code == 401


def test_non_bearer_scheme_returns_401():
    r = client.post(
        "/api/v1/enquiry/understand",
        headers={"Authorization": "Basic dXNlcjpwYXNz"},
        json={"message": "hi"},
    )
    assert r.status_code == 401


# ---------------------------------------------------------------------------
# Body validation
# ---------------------------------------------------------------------------


def test_missing_message_returns_422():
    r = client.post("/api/v1/enquiry/understand", headers=AUTH, json={})
    assert r.status_code == 422


def test_empty_message_returns_422():
    r = client.post(
        "/api/v1/enquiry/understand",
        headers=AUTH,
        json={"message": ""},
    )
    assert r.status_code == 422


def test_oversized_message_returns_422():
    r = client.post(
        "/api/v1/enquiry/understand",
        headers=AUTH,
        json={"message": "x" * 4001},
    )
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# Happy path (service mocked)
# ---------------------------------------------------------------------------


def test_understand_returns_200():
    with patch(
        "app.api.routes.enquiry.understand",
        return_value=_sample_response(),
    ):
        r = client.post(
            "/api/v1/enquiry/understand",
            headers=AUTH,
            json={"message": "5-day trip to Manali"},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["extraction"]["destination"] == "Manali"
    assert body["completeness"] == 0.8
    assert body["missing_fields"] == ["transport", "activities"]


def test_request_id_is_populated():
    with patch(
        "app.api.routes.enquiry.understand",
        return_value=_sample_response(),
    ):
        r = client.post(
            "/api/v1/enquiry/understand",
            headers={**AUTH, "X-Request-ID": "test-trace-1"},
            json={"message": "5-day trip to Manali"},
        )
    assert r.json()["request_id"] == "test-trace-1"
    assert r.headers["x-request-id"] == "test-trace-1"


def test_response_has_all_top_level_fields():
    with patch(
        "app.api.routes.enquiry.understand",
        return_value=_sample_response(),
    ):
        r = client.post(
            "/api/v1/enquiry/understand",
            headers=AUTH,
            json={"message": "trip"},
        )
    body = r.json()
    expected = {
        "extraction",
        "missing_fields",
        "clarifying_questions",
        "completeness",
        "provider",
        "prompt_version",
        "request_id",
    }
    assert set(body.keys()) == expected


def test_optional_fields_accepted():
    with patch(
        "app.api.routes.enquiry.understand",
        return_value=_sample_response(),
    ):
        r = client.post(
            "/api/v1/enquiry/understand",
            headers=AUTH,
            json={
                "message": "trip",
                "user_id": "u_1",
                "currency_hint": "INR",
            },
        )
    assert r.status_code == 200


def test_service_error_returns_502():
    from app.core.exceptions import AIProviderError

    with patch(
        "app.api.routes.enquiry.understand",
        side_effect=AIProviderError("groq down"),
    ):
        r = client.post(
            "/api/v1/enquiry/understand",
            headers=AUTH,
            json={"message": "trip"},
        )
    assert r.status_code == 502
    assert r.json()["error"] == "ai_provider"
