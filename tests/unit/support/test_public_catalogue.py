"""Public catalogue intents — no JWT required, filters honoured."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.capabilities.support.live_data_classifier import (
    LiveDataIntent,
    classify,
)
from app.capabilities.support.schemas import (
    AnswerType,
    LLMAnswerOutput,
    SupportAskRequest,
)
from app.capabilities.support.service import ask
from app.rag.schemas import Audience


@pytest.fixture
def retriever():
    r = MagicMock()
    r.search.return_value = []
    return r


@pytest.fixture
def patch_version_ok():
    with patch(
        "app.capabilities.support.service.check_version_authority"
    ) as m:
        m.return_value = MagicMock(ok=True, local_version="abc123")
        yield m


@pytest.fixture
def patch_gateway():
    with patch(
        "app.capabilities.support.service._get_gateway"
    ) as m:
        g = MagicMock()
        g.generate_structured_sync.return_value = LLMAnswerOutput(
            answer_type=AnswerType.ANSWERED,
            answer="There are packages available.",
            citation_chunk_ids=[],
            confidence=0.9,
        )
        m.return_value = g
        yield g


# Classifier
def test_deterministic_package_search_patterns():
    # May not hit deterministic — LLM handles. Just ensure no crash.
    for q in ["show me packages in Goa", "list packages"]:
        c = classify(q, "customer")
        assert c.intent in {
            LiveDataIntent.PACKAGE_SEARCH,
            LiveDataIntent.NONE,
        }


# Service — public intents must work without JWT
def test_package_search_no_jwt(
    retriever, patch_version_ok, patch_gateway
):
    fake = {
        "packages": [{
            "packageCode": "PKG-922", "title": "Rishikesh Tours",
            "destination": "India", "duration": "2 Days",
            "price": 2000, "currency": "INR",
            "agent": {"companyName": "Karan Tours"},
        }],
        "total": 1,
    }
    with patch(
        "app.capabilities.support.service._get_resolver"
    ) as rmock, patch(
        "app.capabilities.support.service.fetch_for_intent",
        return_value=("GET /api/packages", fake),
    ):
        rmock.return_value.resolve.return_value = MagicMock(
            intent=LiveDataIntent.PACKAGE_SEARCH,
            entities={"destination": "India"},
            source="llm",
            reason="",
        )
        r = ask(
            SupportAskRequest(
                question="show me packages in India",
                audience=Audience.CUSTOMER,
            ),
            retriever=retriever,
            user_jwt=None,          # ← key assertion
        )
    assert r.live_data_used is True
    assert r.live_data_sources == ["GET /api/packages"]
    assert r.retrieval_mode_used == "not_applicable"


def test_package_detail_no_jwt(
    retriever, patch_version_ok, patch_gateway
):
    uuid = "17da203f-942d-4650-b9bb-b84a0d928aca"
    fake = {"package": {
        "id": uuid, "packageCode": "PKG-922",
        "title": "Rishikesh Tours", "destination": "India",
        "duration": "2 Days", "numberOfDays": 2,
        "price": 2000, "basePrice": 3000,
        "savingsPercent": 33, "currency": "INR",
        "inclusions": ["Laxman Jhula"],
        "activities": [], "agent": {"companyName": "Karan Tours"},
        "isEscrowSafe": True,
    }}
    with patch(
        "app.capabilities.support.service._get_resolver"
    ) as rmock, patch(
        "app.capabilities.support.service.fetch_for_intent",
        return_value=("GET /api/packages/{id}", fake),
    ):
        rmock.return_value.resolve.return_value = MagicMock(
            intent=LiveDataIntent.PACKAGE_DETAIL,
            entities={"package_id": uuid},
            source="llm",
            reason="",
        )
        r = ask(
            SupportAskRequest(
                question=f"tell me about package {uuid}",
                audience=Audience.CUSTOMER,
            ),
            retriever=retriever,
            user_jwt=None,
        )
    assert r.live_data_used is True
    assert r.live_data_sources == ["GET /api/packages/{id}"]

    # ← replaced: assert on the prompt the LLM actually received
    prompt = patch_gateway.generate_structured_sync.call_args.kwargs["prompt"]
    assert "LIVE DATA" in prompt
    assert "Rishikesh Tours" in prompt
    assert "PKG-922" in prompt


def test_activity_search_no_jwt(
    retriever, patch_version_ok, patch_gateway
):
    fake = {
        "activities": [{
            "name": "Bungee Jumping", "category": "adventure",
            "state": "Uttrakhand", "country": "India",
            "price": "1500.00", "currency": "INR",
            "durationHours": 6,
        }],
        "total": 1,
    }
    with patch(
        "app.capabilities.support.service._get_resolver"
    ) as rmock, patch(
        "app.capabilities.support.service.fetch_for_intent",
        return_value=("GET /api/activities/public/list", fake),
    ):
        rmock.return_value.resolve.return_value = MagicMock(
            intent=LiveDataIntent.ACTIVITY_SEARCH,
            entities={"category": "adventure"},
            source="llm",
            reason="",
        )
        r = ask(
            SupportAskRequest(
                question="show me adventure activities",
                audience=Audience.CUSTOMER,
            ),
            retriever=retriever,
            user_jwt=None,
        )
    assert r.live_data_used is True
    assert r.live_data_sources == ["GET /api/activities/public/list"]

def test_popular_destinations_no_jwt(
    retriever, patch_version_ok, patch_gateway
):
    fake = {"destinations": [{
        "destination": "Srinagar",
        "total_enquiries": 21,
        "avg_budget": "16380.95",
        "unique_users": 5,
        "completed_enquiries": 1,
    }]}
    with patch(
        "app.capabilities.support.service._get_resolver"
    ) as rmock, patch(
        "app.capabilities.support.service.fetch_for_intent",
        return_value=("GET /api/analytics/popular-destinations", fake),
    ):
        rmock.return_value.resolve.return_value = MagicMock(
            intent=LiveDataIntent.POPULAR_DESTINATIONS,
            entities={},
            source="llm",
            reason="",
        )
        r = ask(
            SupportAskRequest(
                question="trending destinations",
                audience=Audience.CUSTOMER,
            ),
            retriever=retriever,
            user_jwt=None,
        )
    assert r.live_data_used is True
    assert r.live_data_sources == ["GET /api/analytics/popular-destinations"]

    # ← replaced
    prompt = patch_gateway.generate_structured_sync.call_args.kwargs["prompt"]
    assert "LIVE DATA" in prompt
    assert "Srinagar" in prompt
    assert "21" in prompt

def test_package_detail_formatter_produces_expected_shape():
    from app.capabilities.support.live_data_format import format_for_intent

    uuid = "17da203f-942d-4650-b9bb-b84a0d928aca"
    payload = {"package": {
        "id": uuid, "packageCode": "PKG-922",
        "title": "Rishikesh Tours", "destination": "India",
        "duration": "2 Days", "numberOfDays": 2,
        "price": 2000, "basePrice": 3000,
        "savingsPercent": 33, "currency": "INR",
        "inclusions": ["Laxman Jhula", "Bungee Jumping"],
        "activities": [
            {"name": "Laxman Jhula"},
            {"name": "Bungee Jumping"},
        ],
        "agent": {"companyName": "Karan Tours", "isVerified": True},
        "isEscrowSafe": True,
    }}
    out = format_for_intent(LiveDataIntent.PACKAGE_DETAIL, payload)
    assert "PKG-922" in out
    assert "Rishikesh Tours" in out
    assert "Karan Tours" in out
    assert "INR 2,000.00" in out
    assert "Laxman Jhula" in out


def test_activity_formatter_omits_records_without_a_name():
    from app.capabilities.support.live_data_format import format_for_intent

    out = format_for_intent(
        LiveDataIntent.ACTIVITY_SEARCH,
        {
            "activities": [
                {
                    "name": "",
                    "category": "adventure",
                    "state": "Karnataka",
                    "country": "India",
                    "price": "1500.00",
                },
                {
                    "name": "Bungee Jumping",
                    "category": "adventure",
                    "state": "Uttrakhand",
                    "country": "India",
                    "price": "1500.00",
                },
            ],
            "total": 2,
        },
    )

    assert "Bungee Jumping" in out
    assert "Karnataka" not in out
    assert "India" in out


def test_activity_formatter_does_not_invent_missing_names():
    from app.capabilities.support.live_data_format import format_for_intent

    out = format_for_intent(
        LiveDataIntent.ACTIVITY_SEARCH,
        {"activities": [{"name": "", "category": "adventure"}], "total": 1},
    )

    assert "without usable names" in out
    assert "(No title)" not in out
