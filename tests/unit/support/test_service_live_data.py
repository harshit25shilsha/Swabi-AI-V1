"""Service wiring with live data, end-to-end (LLM and retriever mocked)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

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
    with patch(
        "app.capabilities.support.service.check_version_authority"
    ) as m:
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

def test_fetch_success_marks_live_data_used(
    retriever, patch_version_ok, patch_gateway
):
    fake_payload = {
        "bookings": [{
            "bookingCode": "BK-1", "metadata": {"destination": "Goa"},
            "startDate": "2026-11-01T00:00:00Z",
            "endDate": "2026-11-03T00:00:00Z",
            "amount": "5000", "currency": "INR",
            "status": "upcoming", "paymentStatus": "paid",
        }],
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


def test_wallet_balance_uses_live_data_without_rag(
    retriever, patch_version_ok, patch_gateway
):
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


# ---------------------------------------------------------------------------
# Non-live question -> live_data_used stays False
# ---------------------------------------------------------------------------

def test_policy_question_no_live_data(
    retriever, patch_version_ok, patch_gateway
):
    retriever.search.return_value = [
        RetrievedChunk(
            chunk_id="c1", section_id="4.1", section_path="p",
            text="Cancel policy", score=0.8,
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

    prompt_sent_to_llm = (
        patch_gateway.generate_structured_sync.call_args.kwargs["prompt"]
    )
    assert "LIVE DATA" in prompt_sent_to_llm
    assert "BK-8941" in prompt_sent_to_llm
    assert "- Booking status: upcoming" in prompt_sent_to_llm


def test_booking_detail_insufficient_context_falls_back_to_live_data(
    retriever, patch_version_ok, patch_gateway
):
    booking_id = "46f46148-e88d-42b6-bbd4-d87f14009f65"

    patch_gateway.generate_structured_sync.return_value = LLMAnswerOutput(
        answer_type=AnswerType.INSUFFICIENT_CONTEXT,
        answer="I don't have enough context.",
        citation_chunk_ids=[],
        confidence=0.0,
    )

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
    retriever.search.assert_not_called()
