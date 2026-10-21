"""Service orchestration tests. Retriever and gateway are mocked."""
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


def _chunk(i: int = 1) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"id{i}", section_id=f"s{i}",
        section_path=f"p{i}", text=f"text{i}", score=0.8,
    )


@pytest.fixture
def retriever():
    r = MagicMock()
    r.search.return_value = [_chunk()]
    return r


@pytest.fixture
def gateway():
    g = MagicMock()
    g.generate_structured_sync.return_value = LLMAnswerOutput(
        answer_type=AnswerType.ANSWERED,
        answer="Cancellation policy: 100% for >30 days.",
        citation_chunk_ids=["id1"],
        confidence=0.9,
    )
    return g


@pytest.fixture
def patch_gateway(gateway):
    with patch(
        "app.capabilities.support.service._get_gateway",
        return_value=gateway,
    ):
        yield gateway


@pytest.fixture
def patch_version_ok():
    with patch(
        "app.capabilities.support.service.check_version_authority"
    ) as m:
        v = MagicMock()
        v.ok = True
        v.local_version = "abc123"
        m.return_value = v
        yield m


# ---------------------------------------------------------------------------
# Live-data short-circuits before anything else
# ---------------------------------------------------------------------------

def test_live_data_returns_needs_live_data(retriever, patch_version_ok):
    r = ask(
        SupportAskRequest(question="What is my booking status?",
                          audience=Audience.CUSTOMER),
        retriever=retriever,
    )
    assert r.answer_type == AnswerType.NEEDS_LIVE_DATA
    retriever.search.assert_not_called()


# ---------------------------------------------------------------------------
# Version authority fails closed
# ---------------------------------------------------------------------------

def test_version_unavailable_returns_insufficient(retriever):
    with patch(
        "app.capabilities.support.service.check_version_authority"
    ) as m:
        v = MagicMock()
        v.ok = False
        v.local_version = "abc"
        v.reason = "alias mismatch"
        m.return_value = v
        r = ask(
            SupportAskRequest(question="What is the policy?",
                              audience=Audience.CUSTOMER),
            retriever=retriever,
        )
    assert r.answer_type == AnswerType.INSUFFICIENT_CONTEXT
    retriever.search.assert_not_called()


# ---------------------------------------------------------------------------
# Empty retrieval -> insufficient
# ---------------------------------------------------------------------------

def test_empty_retrieval_returns_insufficient(
    retriever, patch_version_ok, patch_gateway
):
    retriever.search.return_value = []
    r = ask(
        SupportAskRequest(question="What about x?",
                          audience=Audience.CUSTOMER),
        retriever=retriever,
    )
    assert r.answer_type == AnswerType.INSUFFICIENT_CONTEXT
    patch_gateway.generate_structured_sync.assert_not_called()


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_answered_with_citation(retriever, patch_version_ok, patch_gateway):
    r = ask(
        SupportAskRequest(question="What is the cancellation policy?",
                          audience=Audience.CUSTOMER),
        retriever=retriever,
    )
    assert r.answer_type == AnswerType.ANSWERED
    assert len(r.citations) == 1
    assert r.citations[0].chunk_id == "id1"
    assert r.knowledge_version == "abc123"
    assert r.provider == "groq"


def test_answered_but_bad_citation_downgrades(
    retriever, patch_version_ok, gateway
):
    gateway.generate_structured_sync.return_value = LLMAnswerOutput(
        answer_type=AnswerType.ANSWERED,
        answer="Some answer.",
        citation_chunk_ids=["not-a-real-id"],
        confidence=0.9,
    )
    with patch(
        "app.capabilities.support.service._get_gateway",
        return_value=gateway,
    ):
        r = ask(
            SupportAskRequest(question="What about y?",
                              audience=Audience.CUSTOMER),
            retriever=retriever,
        )
    assert r.answer_type == AnswerType.INSUFFICIENT_CONTEXT


def test_forbidden_path_downgrades(
    retriever, patch_version_ok, gateway
):
    gateway.generate_structured_sync.return_value = LLMAnswerOutput(
        answer_type=AnswerType.ANSWERED,
        answer="Visit /admin/escrows to check.",
        citation_chunk_ids=["id1"],
        confidence=0.9,
    )
    with patch(
        "app.capabilities.support.service._get_gateway",
        return_value=gateway,
    ):
        r = ask(
            SupportAskRequest(question="What about z?",
                              audience=Audience.CUSTOMER),
            retriever=retriever,
        )
    assert r.answer_type == AnswerType.INSUFFICIENT_CONTEXT


def test_financial_promise_downgrades(
    retriever, patch_version_ok, gateway
):
    gateway.generate_structured_sync.return_value = LLMAnswerOutput(
        answer_type=AnswerType.ANSWERED,
        answer="You will get a refund of 5000.",
        citation_chunk_ids=["id1"],
        confidence=0.9,
    )
    with patch(
        "app.capabilities.support.service._get_gateway",
        return_value=gateway,
    ):
        r = ask(
            SupportAskRequest(question="What about w?",
                              audience=Audience.CUSTOMER),
            retriever=retriever,
        )
    assert r.answer_type == AnswerType.INSUFFICIENT_CONTEXT


# ---------------------------------------------------------------------------
# LLM returns out_of_scope
# ---------------------------------------------------------------------------

def test_llm_out_of_scope_returned(retriever, patch_version_ok, gateway):
    gateway.generate_structured_sync.return_value = LLMAnswerOutput(
        answer_type=AnswerType.OUT_OF_SCOPE,
        answer="",
        citation_chunk_ids=[],
        confidence=0.0,
    )
    with patch(
        "app.capabilities.support.service._get_gateway",
        return_value=gateway,
    ):
        r = ask(
            SupportAskRequest(question="What's Mars weather like?",
                              audience=Audience.CUSTOMER),
            retriever=retriever,
        )
    assert r.answer_type == AnswerType.OUT_OF_SCOPE


# ---------------------------------------------------------------------------
# LLM error fails closed
# ---------------------------------------------------------------------------

def test_llm_error_returns_insufficient(retriever, patch_version_ok):
    with patch(
        "app.capabilities.support.service._get_gateway",
    ) as m:
        m.return_value.generate_structured_sync.side_effect = RuntimeError("boom")
        r = ask(
            SupportAskRequest(question="Policy?",
                              audience=Audience.CUSTOMER),
            retriever=retriever,
        )
    assert r.answer_type == AnswerType.INSUFFICIENT_CONTEXT


# ---------------------------------------------------------------------------
# Audience is forwarded
# ---------------------------------------------------------------------------

def test_audience_forwarded(retriever, patch_version_ok, patch_gateway):
    ask(
        SupportAskRequest(question="policy?",
                          audience=Audience.VENDOR),
        retriever=retriever,
    )
    assert retriever.search.call_args.kwargs["audience"] == "vendor"
