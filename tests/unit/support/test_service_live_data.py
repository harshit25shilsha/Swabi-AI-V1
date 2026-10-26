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
    assert r.answer_type == AnswerType.ANSWERED


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
