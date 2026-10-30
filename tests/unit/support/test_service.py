"""Service orchestration tests. Retriever and gateway are mocked."""
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
    assert r.retrieval_mode_used == "hybrid"
    patch_gateway.generate_structured_sync.assert_not_called()


def test_vendor_insufficient_answer_suggests_relevant_app_sections(
    retriever, patch_version_ok, patch_gateway
):
    retriever.search.return_value = []
    with patch(
        "app.capabilities.support.service._get_resolver"
    ) as resolver:
        resolver.return_value.resolve.return_value = MagicMock(
            intent=LiveDataIntent.NONE,
            entities={},
            source="none",
            reason="not_live_intent_candidate",
        )
        response = ask(
            SupportAskRequest(
                question="Does Swabi support importing vendor itineraries?",
                audience=Audience.VENDOR,
            ),
            retriever=retriever,
        )

    assert response.answer_type == AnswerType.INSUFFICIENT_CONTEXT
    assert "New Enquiries" in response.answer
    assert "bids" in response.answer
    assert "Wallet" in response.answer
    assert "Withdrawals" in response.answer
    assert "Swabi support" in response.answer
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
    assert r.retrieval_mode_used == "hybrid"
    assert retriever.search.call_args.kwargs["index_version"] == "abc123"


def test_booking_by_enquiry_retrieves_workflow_and_uses_llm(
    retriever, patch_version_ok, patch_gateway
):
    question = "How can I book a package in the Swabi platform?"

    response = ask(
        SupportAskRequest(question=question, audience=Audience.CUSTOMER),
        retriever=retriever,
    )

    assert response.answer_type == AnswerType.ANSWERED
    assert retriever.search.call_args.args[0] == (
        "Swabi customer booking process: post a trip enquiry with "
        "requirements, vendors place bids, customer reviews bids and "
        "chats or negotiates, customer accepts a bid, booking is created "
        "and payment starts."
    )
    prompt = patch_gateway.generate_structured_sync.call_args.kwargs["prompt"]
    assert question in prompt
    patch_gateway.generate_structured_sync.assert_called_once()


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


def test_answered_with_mixed_citations_keeps_only_retrieved_citations(
    retriever, patch_version_ok, gateway
):
    gateway.generate_structured_sync.return_value = LLMAnswerOutput(
        answer_type=AnswerType.ANSWERED,
        answer="Some grounded answer.",
        citation_chunk_ids=["id1", "not-a-real-id"],
        confidence=0.9,
    )
    with patch(
        "app.capabilities.support.service._get_gateway",
        return_value=gateway,
    ):
        response = ask(
            SupportAskRequest(
                question="What about y?",
                audience=Audience.CUSTOMER,
            ),
            retriever=retriever,
        )

    assert response.answer_type == AnswerType.ANSWERED
    assert [citation.chunk_id for citation in response.citations] == ["id1"]


def test_forbidden_path_is_redacted_without_discarding_grounded_answer(
    retriever, patch_version_ok, gateway
):
    gateway.generate_structured_sync.return_value = LLMAnswerOutput(
        answer_type=AnswerType.ANSWERED,
        answer="The escrow is released by an administrator. Visit /admin/escrows to check.",
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
    assert r.answer_type == AnswerType.ANSWERED
    assert "released by an administrator" in r.answer
    assert "/admin" not in r.answer
    assert "redacted" in r.answer
    assert r.citations[0].chunk_id == "id1"


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


def test_support_prompt_limits_retrieved_context_and_completion(
    retriever, patch_version_ok, patch_gateway, monkeypatch
):
    retriever.search.return_value = [_chunk(i) for i in range(1, 7)]
    monkeypatch.setattr(
        "app.capabilities.support.service.settings.support_max_chunk_chars",
        3,
    )
    monkeypatch.setattr(
        "app.capabilities.support.service.settings.support_max_context_chunks",
        5,
    )
    monkeypatch.setattr(
        "app.capabilities.support.service.settings.support_answer_max_tokens",
        1536,
    )

    ask(
        SupportAskRequest(
            question="What is the policy?",
            audience=Audience.CUSTOMER,
        ),
        retriever=retriever,
    )

    assert retriever.search.call_args.kwargs["top_k"] == 5
    _, kwargs = patch_gateway.generate_structured_sync.call_args
    assert kwargs["max_tokens"] == 1536
    assert "1-3 short sentences" in kwargs["system"]
    assert "--- chunk 5" in kwargs["prompt"]
    assert "--- chunk 6" not in kwargs["prompt"]
    assert "text1" not in kwargs["prompt"]
    assert "tex..." in kwargs["prompt"]


@pytest.mark.parametrize(
    ("question", "audience"),
    [
        ("I paid for a trip but need to cancel. How is my refund decided?", Audience.CUSTOMER),
        ("What happens after a vendor submits a withdrawal request?", Audience.VENDOR),
        ("What should I do if the vendor does not show up for my booking?", Audience.CUSTOMER),
    ],
)
def test_process_questions_use_rag(
    question, audience, retriever, patch_version_ok, patch_gateway
):
    response = ask(
        SupportAskRequest(question=question, audience=audience),
        retriever=retriever,
    )

    assert response.answer_type == AnswerType.ANSWERED
    retriever.search.assert_called_once()
    assert retriever.search.call_args.kwargs["audience"] == audience.value
