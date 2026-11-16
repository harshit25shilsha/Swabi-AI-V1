"""Public contract guards for POST /api/v1/enquiry/understand.

If these fail, the response shape has drifted and NestJS may break.
"""

from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

from app.capabilities.enquiry.schemas import (
    EnquiryExtraction,
    EnquiryUnderstandResponse,
)
from app.core.config import settings
from app.main import app

client = TestClient(app)
AUTH = {"Authorization": f"Bearer {settings.guardrail_api_token}"}


def _empty_response() -> EnquiryUnderstandResponse:
    return EnquiryUnderstandResponse(
        extraction=EnquiryExtraction(),
        completeness=0.0,
    )


# ---------------------------------------------------------------------------
# Endpoint existence and method
# ---------------------------------------------------------------------------


def test_endpoint_exists_at_exact_path():
    with patch("app.api.routes.enquiry.understand", return_value=_empty_response()):
        r = client.post(
            "/api/v1/enquiry/understand",
            headers=AUTH,
            json={"message": "hi"},
        )
    assert r.status_code == 200


def test_wrong_method_returns_405():
    r = client.get("/api/v1/enquiry/understand", headers=AUTH)
    assert r.status_code == 405


# ---------------------------------------------------------------------------
# Request shape
# ---------------------------------------------------------------------------


def test_minimal_request_accepted():
    with patch("app.api.routes.enquiry.understand", return_value=_empty_response()):
        r = client.post(
            "/api/v1/enquiry/understand",
            headers=AUTH,
            json={"message": "hi"},
        )
    assert r.status_code == 200


def test_optional_fields_accepted():
    with patch("app.api.routes.enquiry.understand", return_value=_empty_response()):
        r = client.post(
            "/api/v1/enquiry/understand",
            headers=AUTH,
            json={
                "message": "hi",
                "user_id": "u_1",
                "currency_hint": "INR",
            },
        )
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# Response field set — locked
# ---------------------------------------------------------------------------


def test_top_level_response_fields_locked():
    with patch("app.api.routes.enquiry.understand", return_value=_empty_response()):
        r = client.post(
            "/api/v1/enquiry/understand",
            headers=AUTH,
            json={"message": "hi"},
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
    assert set(body.keys()) == expected, (
        f"Response top-level fields drifted. "
        f"Extra: {set(body) - expected}. "
        f"Missing: {expected - set(body)}."
    )


def test_extraction_fields_locked():
    with patch("app.api.routes.enquiry.understand", return_value=_empty_response()):
        r = client.post(
            "/api/v1/enquiry/understand",
            headers=AUTH,
            json={"message": "hi"},
        )
    extraction = r.json()["extraction"]
    expected = {
        "title",
        "description",
        "destination",
        "travel_date",
        "travel_date_end",
        "date_flexibility",
        "duration_days",
        "duration_nights",
        "travelers",
        "adults",
        "children",
        "budget",
        "budget_range_min",
        "budget_range_max",
        "requirements",
        "rooms",
        "property_type",
        "trip_type",
        "flags",
        "tags",
    }
    assert set(extraction.keys()) == expected, (
        f"Extraction fields drifted. "
        f"Extra: {set(extraction) - expected}. "
        f"Missing: {expected - set(extraction)}."
    )


def test_requirements_fields_locked():
    with patch("app.api.routes.enquiry.understand", return_value=_empty_response()):
        r = client.post(
            "/api/v1/enquiry/understand",
            headers=AUTH,
            json={"message": "hi"},
        )
    requirements = r.json()["extraction"]["requirements"]
    expected = {
        "accommodation",
        "meals",
        "transport",
        "activities",
        "tour_guide",
        "special_requests",
    }
    assert set(requirements.keys()) == expected, (
        f"Requirements fields drifted. "
        f"Extra: {set(requirements) - expected}. "
        f"Missing: {expected - set(requirements)}."
    )


# ---------------------------------------------------------------------------
# Error contract
# ---------------------------------------------------------------------------


def test_missing_auth_is_401():
    r = client.post("/api/v1/enquiry/understand", json={"message": "hi"})
    assert r.status_code == 401


def test_invalid_body_is_422():
    r = client.post("/api/v1/enquiry/understand", headers=AUTH, json={})
    assert r.status_code == 422


def test_ai_provider_error_is_502():
    from app.core.exceptions import AIProviderError

    with patch(
        "app.api.routes.enquiry.understand",
        side_effect=AIProviderError("boom"),
    ):
        r = client.post(
            "/api/v1/enquiry/understand",
            headers=AUTH,
            json={"message": "hi"},
        )
    assert r.status_code == 502


def test_ai_timeout_is_504():
    from app.core.exceptions import AITimeoutError

    with patch(
        "app.api.routes.enquiry.understand",
        side_effect=AITimeoutError("slow"),
    ):
        r = client.post(
            "/api/v1/enquiry/understand",
            headers=AUTH,
            json={"message": "hi"},
        )
    assert r.status_code == 504
