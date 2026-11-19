"""Response-cache behavior for the enquiry API route."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.capabilities.enquiry.schemas import EnquiryExtraction, EnquiryUnderstandResponse
from app.core.config import settings
from app.main import app

client = TestClient(app)
AUTH = {"Authorization": f"Bearer {settings.guardrail_api_token}"}


def _response() -> EnquiryUnderstandResponse:
    return EnquiryUnderstandResponse(
        extraction=EnquiryExtraction(destination="Manali"),
        completeness=0.0,
        provider="groq",
        prompt_version="test-version",
    )


def test_enquiry_cache_hit_skips_llm_and_assigns_current_request_id(monkeypatch):
    monkeypatch.setattr(settings, "enable_enquiry_response_cache", True)
    get_json = AsyncMock(return_value=_response().model_dump(mode="json"))
    monkeypatch.setattr("app.api.routes.enquiry.get_json", get_json)

    with patch("app.api.routes.enquiry.understand") as understand:
        response = client.post(
            "/api/v1/enquiry/understand",
            headers={**AUTH, "X-Request-ID": "cache-hit-request"},
            json={"message": " trip   to Manali "},
        )

    assert response.status_code == 200
    assert response.json()["request_id"] == "cache-hit-request"
    get_json.assert_awaited_once()
    understand.assert_not_called()


def test_enquiry_cache_miss_stores_success_without_request_id(monkeypatch):
    monkeypatch.setattr(settings, "enable_enquiry_response_cache", True)
    monkeypatch.setattr("app.api.routes.enquiry.get_json", AsyncMock(return_value=None))
    set_json = AsyncMock()
    monkeypatch.setattr("app.api.routes.enquiry.set_json", set_json)

    with patch("app.api.routes.enquiry.understand", return_value=_response()) as understand:
        response = client.post(
            "/api/v1/enquiry/understand",
            headers={**AUTH, "X-Request-ID": "cache-miss-request"},
            json={"message": "trip to Manali", "currency_hint": "inr"},
        )

    assert response.status_code == 200
    understand.assert_called_once_with("trip to Manali", currency_hint="inr")
    set_json.assert_awaited_once()
    cached_value = set_json.await_args.args[1]
    assert cached_value["request_id"] == ""
    assert response.json()["request_id"] == "cache-miss-request"


def test_enquiry_cache_waiter_uses_first_request_result(monkeypatch):
    monkeypatch.setattr(settings, "enable_enquiry_response_cache", True)
    get_json = AsyncMock(side_effect=[None, _response().model_dump(mode="json")])
    monkeypatch.setattr("app.api.routes.enquiry.get_json", get_json)
    monkeypatch.setattr(
        "app.api.routes.enquiry.acquire_lock",
        AsyncMock(return_value=(True, None)),
    )

    with patch("app.api.routes.enquiry.understand") as understand:
        response = client.post(
            "/api/v1/enquiry/understand",
            headers={**AUTH, "X-Request-ID": "waiting-request"},
            json={"message": "trip to Manali"},
        )

    assert response.status_code == 200
    assert response.json()["request_id"] == "waiting-request"
    assert get_json.await_count == 2
    understand.assert_not_called()
