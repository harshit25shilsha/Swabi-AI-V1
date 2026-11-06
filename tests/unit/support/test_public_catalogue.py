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
    with patch("app.capabilities.support.service.check_version_authority") as m:
        m.return_value = MagicMock(ok=True, local_version="abc123")
        yield m


@pytest.fixture
def patch_gateway():
    with patch("app.capabilities.support.service._get_gateway") as m:
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
def test_package_search_no_jwt(retriever, patch_version_ok, patch_gateway):
    fake = {
        "packages": [
            {
                "packageCode": "PKG-922",
                "title": "Rishikesh Tours",
                "destination": "India",
                "duration": "2 Days",
                "price": 2000,
                "currency": "INR",
                "agent": {"companyName": "Karan Tours"},
            }
        ],
        "total": 1,
    }
    with (
        patch("app.capabilities.support.service._get_resolver") as rmock,
        patch(
            "app.capabilities.support.service.fetch_for_intent",
            return_value=("GET /api/packages", fake),
        ),
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
            user_jwt=None,  # ← key assertion
        )
    assert r.live_data_used is True
    assert r.live_data_sources == ["GET /api/packages"]
    assert r.retrieval_mode_used == "not_applicable"


def test_location_package_search_filters_api_results_and_uses_verified_output(
    retriever, patch_version_ok, patch_gateway
):
    patch_gateway.generate_structured_sync.return_value = LLMAnswerOutput(
        answer_type=AnswerType.ANSWERED,
        answer=(
            "Here are two packages:\n"
            "- Rishikesh Tours (PKG-922)\n"
            "- Uttar Pradesh Tour (PKG-921)"
        ),
        citation_chunk_ids=[],
        confidence=0.99,
    )
    with (
        patch("app.capabilities.support.service._get_resolver") as rmock,
        patch(
            "app.capabilities.support.service.fetch_for_intent",
            return_value=(
                "GET /api/packages",
                {
                    "packages": [
                        {
                            "packageCode": "PKG-922",
                            "title": "Rishikesh Tours",
                            "destination": "India",
                            "duration": "2 Days",
                            "numberOfDays": 2,
                            "price": 2000,
                            "currency": "INR",
                        },
                        {
                            "packageCode": "PKG-921",
                            "title": "Uttar Pradesh Tour",
                            "destination": "India",
                            "duration": "2 Days",
                            "numberOfDays": 2,
                            "price": 4000,
                            "currency": "INR",
                        },
                    ],
                    "total": 13,
                },
            ),
        ) as fetch,
    ):
        rmock.return_value.resolve.return_value = MagicMock(
            intent=LiveDataIntent.PACKAGE_SEARCH,
            entities={"query": "Uttarakhand"},
            requested_limit=2,
            source="llm",
            reason="",
            use_rag=True,
        )
        response = ask(
            SupportAskRequest(
                question="provide me 2 packages uttarakhand based",
                audience=Audience.CUSTOMER,
            ),
            retriever=retriever,
            user_jwt=None,
        )

    assert response.answer_type == AnswerType.ANSWERED
    assert "Rishikesh Tours" in response.answer
    assert "PKG-922" in response.answer
    assert "Uttar Pradesh Tour" not in response.answer
    assert "PKG-921" not in response.answer
    assert fetch.call_args.kwargs["entities"] == {
        "query": "Uttarakhand",
        "limit": 20,
    }


def test_location_search_retries_unfiltered_catalogue_when_backend_filter_returns_none(
    retriever, patch_version_ok, patch_gateway
):
    with (
        patch("app.capabilities.support.service._get_resolver") as rmock,
        patch(
            "app.capabilities.support.service.fetch_for_intent",
            side_effect=[
                ("GET /api/packages", {"packages": [], "total": 0}),
                (
                    "GET /api/packages",
                    {
                        "packages": [
                            {
                                "packageCode": "PKG-922",
                                "title": "Rishikesh Tours",
                                "destination": "India",
                                "numberOfDays": 2,
                                "price": 2000,
                                "currency": "INR",
                            },
                            {
                                "packageCode": "PKG-921",
                                "title": "Uttar Pradesh Tour",
                                "destination": "India",
                                "numberOfDays": 2,
                                "price": 4000,
                                "currency": "INR",
                            },
                        ],
                        "total": 13,
                    },
                ),
            ],
        ) as fetch,
    ):
        rmock.return_value.resolve.return_value = MagicMock(
            intent=LiveDataIntent.PACKAGE_SEARCH,
            entities={"query": "Uttarakhand"},
            requested_limit=2,
            source="llm",
            reason="",
        )
        response = ask(
            SupportAskRequest(
                question="provide me 2 packages uttarakhand based",
                audience=Audience.CUSTOMER,
            ),
            retriever=retriever,
            user_jwt=None,
        )

    assert "Rishikesh Tours" in response.answer
    assert "PKG-922" in response.answer
    assert "Uttar Pradesh Tour" not in response.answer
    assert "PKG-921" not in response.answer
    assert fetch.call_count == 2
    assert fetch.call_args_list[0].kwargs["entities"] == {
        "query": "Uttarakhand",
        "limit": 20,
    }
    assert fetch.call_args_list[1].kwargs["entities"] == {"limit": 20}


def test_uttar_pradesh_search_does_not_return_rishikesh_package(
    retriever, patch_version_ok, patch_gateway
):
    with (
        patch("app.capabilities.support.service._get_resolver") as rmock,
        patch(
            "app.capabilities.support.service.fetch_for_intent",
            return_value=(
                "GET /api/packages",
                {
                    "packages": [
                        {
                            "packageCode": "PKG-922",
                            "title": "Rishikesh Tours",
                            "destination": "India",
                        },
                        {
                            "packageCode": "PKG-921",
                            "title": "Uttar Pradesh Tour",
                            "destination": "India",
                        },
                    ],
                    "total": 13,
                },
            ),
        ) as fetch,
    ):
        rmock.return_value.resolve.return_value = MagicMock(
            intent=LiveDataIntent.PACKAGE_SEARCH,
            entities={"query": "uttar pradesh"},
            requested_limit=2,
            source="deterministic",
            reason="location_package_search",
        )
        response = ask(
            SupportAskRequest(
                question="provide me 2 uttar pradesh based packages",
                audience=Audience.CUSTOMER,
            ),
            retriever=retriever,
            user_jwt=None,
        )

    assert "Uttar Pradesh Tour" in response.answer
    assert "PKG-921" in response.answer
    assert "Rishikesh Tours" not in response.answer
    assert "PKG-922" not in response.answer
    assert fetch.call_args.kwargs["entities"] == {
        "query": "uttar pradesh",
        "limit": 20,
    }


def test_winter_package_search_returns_live_catalogue_and_discloses_limit(
    retriever, patch_version_ok, patch_gateway
):
    question = "provide me available packages in swabi platform for winter that i can book"
    patch_gateway.generate_structured_sync.return_value = LLMAnswerOutput(
        answer_type=AnswerType.ANSWERED,
        answer=(
            "Rishikesh Tours (PKG-922) is available in the catalogue. The catalogue "
            "cannot filter by season, so please confirm winter availability "
            "with the provider before booking."
        ),
        citation_chunk_ids=[],
        confidence=0.9,
    )
    with (
        patch("app.capabilities.support.service._get_resolver") as rmock,
        patch(
            "app.capabilities.support.service.fetch_for_intent",
            return_value=(
                "GET /api/packages",
                {
                    "packages": [
                        {
                            "packageCode": "PKG-922",
                            "title": "Rishikesh Tours",
                            "destination": "India",
                            "duration": "2 Days",
                            "price": 2000,
                            "currency": "INR",
                        }
                    ],
                    "total": 1,
                },
            ),
        ),
    ):
        rmock.return_value.resolve.return_value = MagicMock(
            intent=LiveDataIntent.PACKAGE_SEARCH,
            entities={},
            source="deterministic",
            reason="public_package_search",
        )
        response = ask(
            SupportAskRequest(
                question=question,
                audience=Audience.CUSTOMER,
            ),
            retriever=retriever,
            user_jwt=None,
        )

    assert response.live_data_used is True
    assert response.live_data_sources == ["GET /api/packages"]
    assert response.answer_type == AnswerType.ANSWERED
    assert "PKG-922" in response.answer
    assert "only lets me filter packages by destination" in response.answer
    assert "provider before booking" in response.answer
    assert response.provider == "groq"
    prompt = patch_gateway.generate_structured_sync.call_args.kwargs["prompt"]
    assert "only lets me filter packages by destination" in prompt
    patch_gateway.generate_structured_sync.assert_called_once()
    retriever.search.assert_not_called()


def test_package_detail_no_jwt(retriever, patch_version_ok, patch_gateway):
    uuid = "17da203f-942d-4650-b9bb-b84a0d928aca"
    fake = {
        "package": {
            "id": uuid,
            "packageCode": "PKG-922",
            "title": "Rishikesh Tours",
            "destination": "India",
            "duration": "2 Days",
            "numberOfDays": 2,
            "price": 2000,
            "basePrice": 3000,
            "savingsPercent": 33,
            "currency": "INR",
            "inclusions": ["Laxman Jhula"],
            "activities": [],
            "agent": {"companyName": "Karan Tours"},
            "isEscrowSafe": True,
        }
    }
    patch_gateway.generate_structured_sync.return_value = LLMAnswerOutput(
        answer_type=AnswerType.ANSWERED,
        answer="Rishikesh Tours (PKG-922) is a 2-day package in India.",
        citation_chunk_ids=[],
        confidence=0.9,
    )
    with (
        patch("app.capabilities.support.service._get_resolver") as rmock,
        patch(
            "app.capabilities.support.service.fetch_for_intent",
            return_value=("GET /api/packages/{id}", fake),
        ),
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

    assert r.answer_type == AnswerType.ANSWERED
    assert "Rishikesh Tours" in r.answer
    assert "PKG-922" in r.answer
    patch_gateway.generate_structured_sync.assert_called_once()


def test_activity_search_no_jwt(retriever, patch_version_ok, patch_gateway):
    fake = {
        "activities": [
            {
                "name": "Bungee Jumping",
                "category": "adventure",
                "state": "Uttrakhand",
                "country": "India",
                "price": "1500.00",
                "currency": "INR",
                "durationHours": 6,
            }
        ],
        "total": 1,
    }
    with (
        patch("app.capabilities.support.service._get_resolver") as rmock,
        patch(
            "app.capabilities.support.service.fetch_for_intent",
            return_value=("GET /api/activities/public/list", fake),
        ),
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


def test_popular_destinations_no_jwt(retriever, patch_version_ok, patch_gateway):
    fake = {
        "destinations": [
            {
                "destination": "Srinagar",
                "total_enquiries": 21,
                "avg_budget": "16380.95",
                "unique_users": 5,
                "completed_enquiries": 1,
            }
        ]
    }
    patch_gateway.generate_structured_sync.return_value = LLMAnswerOutput(
        answer_type=AnswerType.ANSWERED,
        answer="Srinagar is trending, with 21 enquiries and an average budget of INR 16,380.95.",
        citation_chunk_ids=[],
        confidence=0.9,
    )
    with (
        patch("app.capabilities.support.service._get_resolver") as rmock,
        patch(
            "app.capabilities.support.service.fetch_for_intent",
            return_value=("GET /api/analytics/popular-destinations", fake),
        ),
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

    assert r.answer_type == AnswerType.ANSWERED
    assert "Srinagar" in r.answer
    assert "21" in r.answer
    patch_gateway.generate_structured_sync.assert_called_once()


def test_package_detail_formatter_produces_expected_shape():
    from app.capabilities.support.live_data_format import format_for_intent

    uuid = "17da203f-942d-4650-b9bb-b84a0d928aca"
    payload = {
        "package": {
            "id": uuid,
            "packageCode": "PKG-922",
            "title": "Rishikesh Tours",
            "destination": "India",
            "duration": "2 Days",
            "numberOfDays": 2,
            "price": 2000,
            "basePrice": 3000,
            "savingsPercent": 33,
            "currency": "INR",
            "inclusions": ["Laxman Jhula", "Bungee Jumping"],
            "activities": [
                {"name": "Laxman Jhula"},
                {"name": "Bungee Jumping"},
            ],
            "agent": {"companyName": "Karan Tours", "isVerified": True},
            "isEscrowSafe": True,
        }
    }
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
