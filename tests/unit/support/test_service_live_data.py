"""Service wiring with live data, end-to-end (LLM and retriever mocked)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.capabilities.support.live_data_classifier import LiveDataIntent
from app.capabilities.support.schemas import (
    AnswerType,
    LLMAnswerOutput,
    SupportAskRequest,
)
from app.capabilities.support.service import ask
from app.rag.schemas import Audience, RetrievedChunk


@pytest.fixture
def retriever():
    r = MagicMock()
    r.search.return_value = []
    return r


@pytest.fixture
def gateway_answered():
    g = MagicMock()
    g.generate_structured_sync.return_value = LLMAnswerOutput(
        answer_type=AnswerType.ANSWERED,
        answer="You have 3 upcoming bookings.",
        citation_chunk_ids=[],
        confidence=0.9,
    )
    return g


@pytest.fixture
def patch_gateway(gateway_answered):
    with patch(
        "app.capabilities.support.service._get_gateway",
        return_value=gateway_answered,
    ):
        yield gateway_answered


@pytest.fixture
def patch_version_ok():
    with patch("app.capabilities.support.service.check_version_authority") as m:
        v = MagicMock(ok=True, local_version="abc123")
        m.return_value = v
        yield m


# ---------------------------------------------------------------------------
# Audience mismatch -> needs_live_data, no fetch
# ---------------------------------------------------------------------------


def test_customer_asking_vendor_scope(retriever, patch_version_ok):
    r = ask(
        SupportAskRequest(
            question="What is my vendor wallet balance?",
            audience=Audience.CUSTOMER,
        ),
        retriever=retriever,
        user_jwt="jwt-x",
    )
    assert r.answer_type == AnswerType.NEEDS_LIVE_DATA
    retriever.search.assert_not_called()


def test_vendor_asking_customer_scope(retriever, patch_version_ok):
    r = ask(
        SupportAskRequest(
            question="What are my payments?",
            audience=Audience.VENDOR,
        ),
        retriever=retriever,
        user_jwt="jwt-x",
    )
    assert r.answer_type == AnswerType.NEEDS_LIVE_DATA


def test_llm_classified_cross_role_request_is_blocked_before_fetch(retriever):
    with (
        patch("app.capabilities.support.service._get_resolver") as resolver,
        patch("app.capabilities.support.service.fetch_for_intent") as fetch,
    ):
        resolver.return_value.resolve.return_value = MagicMock(
            intent=LiveDataIntent.VENDOR_WALLET,
            entities={},
            source="llm",
            reason="llm_conf=0.95",
            use_rag=False,
        )
        response = ask(
            SupportAskRequest(
                question="Show the provider's wallet balance",
                audience=Audience.CUSTOMER,
            ),
            retriever=retriever,
            user_jwt="customer-jwt",
        )

    assert response.answer_type == AnswerType.NEEDS_LIVE_DATA
    fetch.assert_not_called()
    retriever.search.assert_not_called()


# ---------------------------------------------------------------------------
# No JWT -> needs_live_data, no fetch
# ---------------------------------------------------------------------------


def test_live_intent_without_jwt_returns_needs_live(retriever, patch_version_ok):
    r = ask(
        SupportAskRequest(
            question="What are my bookings?",
            audience=Audience.CUSTOMER,
        ),
        retriever=retriever,
        user_jwt=None,
    )
    assert r.answer_type == AnswerType.NEEDS_LIVE_DATA
    assert "sign in" in r.answer.lower()


# ---------------------------------------------------------------------------
# Successful fetch -> live data in answer, correct fields
# ---------------------------------------------------------------------------


def test_fetch_success_marks_live_data_used(retriever, patch_version_ok, patch_gateway):
    fake_payload = {
        "bookings": [
            {
                "bookingCode": "BK-1",
                "metadata": {"destination": "Goa"},
                "startDate": "2026-11-01T00:00:00Z",
                "endDate": "2026-11-03T00:00:00Z",
                "amount": "5000",
                "currency": "INR",
                "status": "upcoming",
                "paymentStatus": "paid",
            }
        ],
        "total": 1,
    }
    with patch(
        "app.capabilities.support.service.fetch_for_intent",
        return_value=("GET /api/bookings/my", fake_payload),
    ):
        r = ask(
            SupportAskRequest(
                question="What are my bookings?",
                audience=Audience.CUSTOMER,
            ),
            retriever=retriever,
            user_jwt="jwt-x",
        )
    assert r.live_data_used is True
    assert r.live_data_sources == ["GET /api/bookings/my"]
    assert r.retrieval_mode_used == "not_applicable"
    assert r.answer_type == AnswerType.ANSWERED
    patch_gateway.generate_structured_sync.assert_called_once()
    patch_version_ok.assert_not_called()
    retriever.search.assert_not_called()


def test_wallet_balance_uses_live_data_without_rag(retriever, patch_version_ok, patch_gateway):
    with patch(
        "app.capabilities.support.service.fetch_for_intent",
        return_value=(
            "GET /api/wallet/me/balance",
            {
                "coinBalance": 0,
                "totalCoinsBalance": 1021,
                "pendingCoins": 1021,
                "lifetimeSavings": 0,
                "creditBalance": 0,
            },
        ),
    ):
        response = ask(
            SupportAskRequest(
                question="check my wallet balance",
                audience=Audience.CUSTOMER,
            ),
            retriever=retriever,
            user_jwt="jwt-x",
        )

    assert response.live_data_used is True
    assert response.live_data_sources == ["GET /api/wallet/me/balance"]
    assert response.retrieval_mode_used == "not_applicable"
    retriever.search.assert_not_called()
    patch_gateway.generate_structured_sync.assert_called_once()
    patch_version_ok.assert_not_called()


# ---------------------------------------------------------------------------
# Fetch failure -> needs_live_data with clear message
# ---------------------------------------------------------------------------


def test_fetch_failure_returns_needs_live(retriever, patch_version_ok):
    from app.capabilities.support.live_data import LiveDataFetchError

    with patch(
        "app.capabilities.support.service.fetch_for_intent",
        side_effect=LiveDataFetchError("timeout"),
    ):
        r = ask(
            SupportAskRequest(
                question="What are my bookings?",
                audience=Audience.CUSTOMER,
            ),
            retriever=retriever,
            user_jwt="jwt-x",
        )
    assert r.answer_type == AnswerType.NEEDS_LIVE_DATA
    assert "couldn't retrieve" in r.answer.lower()
    assert "swabi support" in r.answer.lower()


# ---------------------------------------------------------------------------
# Non-live question -> live_data_used stays False
# ---------------------------------------------------------------------------


def test_policy_question_no_live_data(retriever, patch_version_ok, patch_gateway):
    retriever.search.return_value = [
        RetrievedChunk(
            chunk_id="c1",
            section_id="4.1",
            section_path="p",
            text="Cancel policy",
            score=0.8,
        ),
    ]
    # Answer must cite a retrieved chunk
    patch_gateway.generate_structured_sync.return_value = LLMAnswerOutput(
        answer_type=AnswerType.ANSWERED,
        answer="Policy: ...",
        citation_chunk_ids=["c1"],
        confidence=0.9,
    )
    r = ask(
        SupportAskRequest(
            question="What is the cancellation policy?",
            audience=Audience.CUSTOMER,
        ),
        retriever=retriever,
        user_jwt="jwt-x",
    )
    assert r.live_data_used is False
    assert r.live_data_sources == []


def test_cancel_and_refund_instructions_use_rag_without_live_api(
    retriever, patch_version_ok, patch_gateway
):
    retriever.search.return_value = [
        RetrievedChunk(
            chunk_id="c1",
            section_id="refund-policy",
            section_path="Policies/Refunds",
            text="Cancellation and refund policy details.",
            score=0.8,
        ),
    ]
    patch_gateway.generate_structured_sync.return_value = LLMAnswerOutput(
        answer_type=AnswerType.ANSWERED,
        answer="Follow the cancellation and refund policy.",
        citation_chunk_ids=["c1"],
        confidence=0.9,
    )
    with patch(
        "app.capabilities.support.service.fetch_for_intent"
    ) as fetch_for_intent:
        response = ask(
            SupportAskRequest(
                question="How can I cancel my booking and get a refund?",
                audience=Audience.CUSTOMER,
            ),
            retriever=retriever,
            user_jwt="jwt-x",
        )

    fetch_for_intent.assert_not_called()
    retriever.search.assert_called_once()
    assert response.live_data_used is False
    assert response.retrieval_mode_used == "hybrid"
    assert [citation.chunk_id for citation in response.citations] == ["c1"]


def test_policy_and_booking_status_uses_rag_and_live_api(
    retriever, patch_version_ok, patch_gateway
):
    retriever.search.return_value = [
        RetrievedChunk(
            chunk_id="c1",
            section_id="refund-policy",
            section_path="Policies/Refunds",
            text="Cancellation policy details.",
            score=0.8,
        ),
    ]
    patch_gateway.generate_structured_sync.return_value = LLMAnswerOutput(
        answer_type=AnswerType.ANSWERED,
        answer="Your booking is upcoming; see the cited policy for cancellation terms.",
        citation_chunk_ids=["c1"],
        confidence=0.9,
    )
    with patch(
        "app.capabilities.support.service.fetch_for_intent",
        return_value=(
            "GET /api/bookings/my",
            {"bookings": [{"bookingCode": "BK-1", "status": "upcoming"}], "total": 1},
        ),
    ) as fetch_for_intent:
        response = ask(
            SupportAskRequest(
                question=(
                    "What is the cancellation policy and what is the status "
                    "of my booking?"
                ),
                audience=Audience.CUSTOMER,
            ),
            retriever=retriever,
            user_jwt="jwt-x",
        )

    fetch_for_intent.assert_called_once()
    retriever.search.assert_called_once()
    assert response.live_data_used is True
    assert response.retrieval_mode_used == "hybrid_live"
    assert [citation.chunk_id for citation in response.citations] == ["c1"]


def test_mixed_request_without_jwt_still_answers_from_rag(
    retriever, patch_version_ok, patch_gateway
):
    retriever.search.return_value = [
        RetrievedChunk(
            chunk_id="c1",
            section_id="refund-policy",
            section_path="Policies/Refunds",
            text="Cancellation policy details.",
            score=0.8,
        ),
    ]
    patch_gateway.generate_structured_sync.return_value = LLMAnswerOutput(
        answer_type=AnswerType.ANSWERED,
        answer="See the cited cancellation policy.",
        citation_chunk_ids=["c1"],
        confidence=0.9,
    )
    with patch(
        "app.capabilities.support.service.fetch_for_intent"
    ) as fetch_for_intent:
        response = ask(
            SupportAskRequest(
                question=(
                    "What is the cancellation policy and what is the status "
                    "of my booking?"
                ),
                audience=Audience.CUSTOMER,
            ),
            retriever=retriever,
            user_jwt=None,
        )

    fetch_for_intent.assert_not_called()
    retriever.search.assert_called_once()
    assert response.live_data_used is False
    assert response.live_data_warning is not None
    assert response.citations[0].chunk_id == "c1"


def test_mixed_request_api_failure_still_answers_from_rag(
    retriever, patch_version_ok, patch_gateway
):
    from app.capabilities.support.live_data import LiveDataFetchError

    retriever.search.return_value = [
        RetrievedChunk(
            chunk_id="c1",
            section_id="refund-policy",
            section_path="Policies/Refunds",
            text="Cancellation policy details.",
            score=0.8,
        ),
    ]
    patch_gateway.generate_structured_sync.return_value = LLMAnswerOutput(
        answer_type=AnswerType.ANSWERED,
        answer="Follow the cited cancellation policy.",
        citation_chunk_ids=["c1"],
        confidence=0.9,
    )
    with patch(
        "app.capabilities.support.service.fetch_for_intent",
        side_effect=LiveDataFetchError("backend unavailable"),
    ):
        response = ask(
            SupportAskRequest(
                question=(
                    "What is the cancellation policy and what is the status "
                    "of my booking?"
                ),
                audience=Audience.CUSTOMER,
            ),
            retriever=retriever,
            user_jwt="jwt-x",
        )

    retriever.search.assert_called_once()
    assert response.answer_type == AnswerType.ANSWERED
    assert response.live_data_used is False
    assert response.live_data_warning is not None
    assert response.citations[0].chunk_id == "c1"


def test_booking_detail_uses_live_data_without_rag_chunks(
    retriever, patch_version_ok, patch_gateway
):
    booking_id = "46f46148-e88d-42b6-bbd4-d87f14009f65"
    fake_payload = {
        "booking": {
            "id": booking_id,
            "bookingCode": "BK-8941",
            "bookingType": "package",
            "package": {"name": "Kashmir Tour"},
            "startDate": "2026-11-01T00:00:00Z",
            "endDate": "2026-11-07T00:00:00Z",
            "status": "upcoming",
            "paymentStatus": "paid",
            "amount": "25000",
            "currency": "INR",
        }
    }
    patch_gateway.generate_structured_sync.return_value = LLMAnswerOutput(
        answer_type=AnswerType.ANSWERED,
        answer="Booking BK-8941 is upcoming from 1–7 November 2026.",
        citation_chunk_ids=[],
        confidence=0.93,
    )

    with patch(
        "app.capabilities.support.service.fetch_for_intent",
        return_value=("GET /api/bookings/my/{id}", fake_payload),
    ):
        result = ask(
            SupportAskRequest(
                question=f"What is the status of booking {booking_id}?",
                audience=Audience.CUSTOMER,
            ),
            retriever=retriever,
            user_jwt="test-user-jwt",
        )

    assert result.answer_type == AnswerType.ANSWERED
    assert result.live_data_used is True
    assert result.live_data_sources == ["GET /api/bookings/my/{id}"]
    assert "upcoming" in result.answer.lower()
    retriever.search.assert_not_called()
    patch_version_ok.assert_not_called()
    patch_gateway.generate_structured_sync.assert_called_once()
    prompt = patch_gateway.generate_structured_sync.call_args.kwargs["prompt"]
    assert "BK-8941" in prompt
    assert "Booking status: upcoming" in prompt


def test_live_data_generation_failure_uses_verified_live_facts(
    retriever, patch_version_ok, patch_gateway
):
    booking_id = "46f46148-e88d-42b6-bbd4-d87f14009f65"

    fake_payload = {
        "booking": {
            "id": booking_id,
            "bookingCode": "BK-8941",
            "bookingType": "package",
            "package": {"name": "Kashmir Tour"},
            "startDate": "2026-11-01T00:00:00Z",
            "endDate": "2026-11-07T00:00:00Z",
            "status": "upcoming",
            "paymentStatus": "paid",
            "amount": "25000",
            "currency": "INR",
        }
    }
    patch_gateway.generate_structured_sync.side_effect = RuntimeError("provider unavailable")

    with patch(
        "app.capabilities.support.service.fetch_for_intent",
        return_value=("GET /api/bookings/my/{id}", fake_payload),
    ):
        result = ask(
            SupportAskRequest(
                question=f"What is the status of booking {booking_id}?",
                audience=Audience.CUSTOMER,
            ),
            retriever=retriever,
            user_jwt="test-user-jwt",
        )

    assert result.answer_type == AnswerType.ANSWERED
    assert result.live_data_used is True
    assert result.live_data_sources == ["GET /api/bookings/my/{id}"]
    assert "BK-8941" in result.answer
    assert "upcoming" in result.answer.lower()
    assert result.live_data_warning is not None
    assert not result.answer.startswith("I retrieved your current Swabi information")
    retriever.search.assert_not_called()
    patch_version_ok.assert_not_called()
    patch_gateway.generate_structured_sync.assert_called_once()


def test_insufficient_model_answer_uses_conversational_live_catalogue_fallback(
    retriever, patch_version_ok, patch_gateway
):
    packages = [
        {
            "packageCode": f"PKG-{index}",
            "title": f"Tour {index}",
            "destination": "Goa",
            "price": index * 1000,
            "currency": "INR",
        }
        for index in range(1, 3)
    ]
    with (
        patch("app.capabilities.support.service._get_resolver") as resolver,
        patch(
            "app.capabilities.support.service.fetch_for_intent",
            return_value=(
                "GET /api/packages",
                {"packages": packages, "total": 13},
            ),
        ),
    ):
        resolver.return_value.resolve.return_value = MagicMock(
            intent=LiveDataIntent.PACKAGE_SEARCH,
            entities={},
            source="llm",
            reason="llm_conf=0.95",
            use_rag=False,
            requested_limit=2,
        )
        patch_gateway.generate_structured_sync.return_value = LLMAnswerOutput(
            answer_type=AnswerType.INSUFFICIENT_CONTEXT,
            answer="",
            citation_chunk_ids=[],
            confidence=0.2,
        )
        response = ask(
            SupportAskRequest(
                question="Show me only two packages",
                audience=Audience.CUSTOMER,
            ),
            retriever=retriever,
        )

    assert response.answer_type == AnswerType.ANSWERED
    assert response.answer.startswith(
        "I found 13 packages in the catalogue. Here are 2 to get you started:"
    )
    assert response.answer.count("\n- ") == 2
    assert "I found your live information" not in response.answer
    assert "Tour 3" not in response.answer
    assert response.live_data_used is True
    patch_version_ok.assert_not_called()
    retriever.search.assert_not_called()


def test_requested_limit_is_applied_to_packages_and_generation_context(
    retriever, patch_version_ok, patch_gateway
):
    packages = [
        {
            "packageCode": f"PKG-{index}",
            "title": f"Tour {index}",
            "destination": "Goa",
            "price": index * 1000,
            "currency": "INR",
        }
        for index in range(1, 4)
    ]
    with (
        patch("app.capabilities.support.service._get_resolver") as resolver,
        patch(
            "app.capabilities.support.service.fetch_for_intent",
            return_value=(
                "GET /api/packages",
                {"packages": packages, "total": 13},
            ),
        ) as fetch,
    ):
        resolver.return_value.resolve.return_value = MagicMock(
            intent=LiveDataIntent.PACKAGE_SEARCH,
            entities={"query": "Goa"},
            source="llm",
            reason="",
            use_rag=False,
            requested_limit=2,
        )
        patch_gateway.generate_structured_sync.return_value = LLMAnswerOutput(
            answer_type=AnswerType.ANSWERED,
            answer="Here are two Goa packages:\n- Tour 1\n- Tour 2",
            citation_chunk_ids=[],
            confidence=0.95,
        )
        response = ask(
            SupportAskRequest(
                question="provide me only 2 packages available in this platform",
                audience=Audience.CUSTOMER,
            ),
            retriever=retriever,
        )

    assert "Here are 2 to get you started" in response.answer
    assert "Tour 1" in response.answer and "Tour 2" in response.answer
    assert "Tour 3" not in response.answer
    assert fetch.call_args.kwargs["entities"] == {"query": "Goa", "limit": 20}
    prompt = patch_gateway.generate_structured_sync.call_args.kwargs["prompt"]
    assert "REQUESTED RESULT LIMIT: 2" in prompt
    assert "Tour 1" in prompt and "Tour 2" in prompt
    assert "Tour 3" not in prompt
    assert "3 packages" in prompt
    assert "Here are 2 to get you started" in prompt
    assert "Before a result list, add one short, natural lead-in" in (
        patch_gateway.generate_structured_sync.call_args.kwargs["system"]
    )
    patch_version_ok.assert_not_called()
    retriever.search.assert_not_called()


def test_requested_limit_is_applied_to_payment_history(
    retriever, patch_version_ok, patch_gateway
):
    payments = [
        {
            "createdAt": f"2026-10-{day:02d}T00:00:00Z",
            "amount": day * 100,
            "currency": "INR",
            "status": "completed",
        }
        for day in range(1, 4)
    ]
    with (
        patch("app.capabilities.support.service._get_resolver") as resolver,
        patch(
            "app.capabilities.support.service.fetch_for_intent",
            return_value=("GET /api/payments/my-payments", {"payments": payments, "total": 20}),
        ),
    ):
        resolver.return_value.resolve.return_value = MagicMock(
            intent=LiveDataIntent.MY_PAYMENTS,
            entities={},
            source="llm",
            reason="",
            use_rag=False,
            requested_limit=2,
        )
        patch_gateway.generate_structured_sync.return_value = LLMAnswerOutput(
            answer_type=AnswerType.ANSWERED,
            answer="Your two latest payments:\n- INR 100\n- INR 200",
            citation_chunk_ids=[],
            confidence=0.95,
        )
        response = ask(
            SupportAskRequest(
                question="Show me my last two payment history",
                audience=Audience.CUSTOMER,
            ),
            retriever=retriever,
            user_jwt="test-user-jwt",
        )

    assert response.answer.startswith("Your two latest payments")
    prompt = patch_gateway.generate_structured_sync.call_args.kwargs["prompt"]
    assert prompt.count("status=completed") == 2
    assert "03 Oct 2026" not in prompt
    assert "REQUESTED RESULT LIMIT: 2" in prompt
    patch_version_ok.assert_not_called()
    retriever.search.assert_not_called()


def test_unspecified_payment_history_is_summarized_from_available_records(
    retriever, patch_version_ok, patch_gateway
):
    payments = [
        {
            "createdAt": f"2026-10-{day:02d}T00:00:00Z",
            "amount": day * 100,
            "currency": "INR",
            "status": "completed" if day % 2 else "pending",
        }
        for day in range(1, 21)
    ]
    with (
        patch("app.capabilities.support.service._get_resolver") as resolver,
        patch(
            "app.capabilities.support.service.fetch_for_intent",
            return_value=(
                "GET /api/payments/my-payments",
                {"payments": payments, "total": len(payments)},
            ),
        ),
    ):
        resolver.return_value.resolve.return_value = MagicMock(
            intent=LiveDataIntent.MY_PAYMENTS,
            entities={},
            source="deterministic",
            reason="customer transaction history",
            use_rag=False,
            requested_limit=None,
        )
        patch_gateway.generate_structured_sync.return_value = LLMAnswerOutput(
            answer_type=AnswerType.ANSWERED,
            answer=(
                "I found 20 payment records, with completed and pending "
                "payments. Would you like the latest five, or a date or "
                "status filter?"
            ),
            citation_chunk_ids=[],
            confidence=0.95,
        )
        response = ask(
            SupportAskRequest(
                question="Show my transaction history",
                audience=Audience.CUSTOMER,
            ),
            retriever=retriever,
            user_jwt="test-user-jwt",
        )

    assert response.answer.startswith("I found 20 payment records")
    assert "latest five" in response.answer
    prompt = patch_gateway.generate_structured_sync.call_args.kwargs["prompt"]
    assert prompt.count("status=") == 20
    assert "PAYMENT HISTORY SUMMARY MODE" in prompt
    assert "20 of 20" in prompt
    patch_version_ok.assert_not_called()
    retriever.search.assert_not_called()


def test_generated_live_list_that_ignores_limit_falls_back_to_exact_facts(
    retriever, patch_version_ok, patch_gateway
):
    packages = [
        {
            "packageCode": f"PKG-{index}",
            "title": f"Tour {index}",
            "destination": "Goa",
            "price": index * 1000,
            "currency": "INR",
        }
        for index in range(1, 4)
    ]
    with (
        patch("app.capabilities.support.service._get_resolver") as resolver,
        patch(
            "app.capabilities.support.service.fetch_for_intent",
            return_value=(
                "GET /api/packages",
                {"packages": packages, "total": 13},
            ),
        ),
    ):
        resolver.return_value.resolve.return_value = MagicMock(
            intent=LiveDataIntent.PACKAGE_SEARCH,
            entities={},
            source="llm",
            reason="",
            use_rag=False,
            requested_limit=2,
        )
        patch_gateway.generate_structured_sync.return_value = LLMAnswerOutput(
            answer_type=AnswerType.ANSWERED,
            answer="Here are five:\n- Tour A\n- Tour B\n- Tour C\n- Tour D\n- Tour E",
            citation_chunk_ids=[],
            confidence=0.95,
        )
        response = ask(
            SupportAskRequest(
                question="Show me only two packages",
                audience=Audience.CUSTOMER,
            ),
            retriever=retriever,
        )

    assert response.answer.count("\n- ") == 2
    assert "Tour 1" in response.answer
    assert "Tour 2" in response.answer
    assert "Tour 3" not in response.answer


def test_empty_live_catalogue_cannot_be_filled_with_generated_records(
    retriever, patch_version_ok, patch_gateway
):
    with (
        patch("app.capabilities.support.service._get_resolver") as resolver,
        patch(
            "app.capabilities.support.service.fetch_for_intent",
            return_value=("GET /api/packages", {"packages": [], "total": 0}),
        ),
    ):
        resolver.return_value.resolve.return_value = MagicMock(
            intent=LiveDataIntent.PACKAGE_SEARCH,
            entities={},
            source="llm",
            reason="",
            use_rag=False,
            requested_limit=2,
        )
        patch_gateway.generate_structured_sync.return_value = LLMAnswerOutput(
            answer_type=AnswerType.ANSWERED,
            answer="Here is a package:\n- Surprise Goa Tour",
            citation_chunk_ids=[],
            confidence=0.95,
        )
        response = ask(
            SupportAskRequest(
                question="Show me two packages in Goa",
                audience=Audience.CUSTOMER,
            ),
            retriever=retriever,
        )

    assert "couldn't find any packages" in response.answer.lower()
    assert "Surprise Goa Tour" not in response.answer
