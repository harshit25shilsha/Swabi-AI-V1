"""Support schemas — validation boundaries."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.capabilities.support.schemas import (
    AnswerType,
    Citation,
    LLMAnswerOutput,
    SupportAskRequest,
    SupportAskResponse,
)
from app.rag.schemas import Audience


def test_request_requires_question():
    with pytest.raises(ValidationError):
        SupportAskRequest(audience="customer")


def test_request_requires_audience():
    with pytest.raises(ValidationError):
        SupportAskRequest(question="hi")


def test_request_rejects_empty_question():
    with pytest.raises(ValidationError):
        SupportAskRequest(question="", audience="customer")


def test_request_rejects_oversized_question():
    with pytest.raises(ValidationError):
        SupportAskRequest(question="x" * 2001, audience="customer")


def test_request_audience_enum():
    r = SupportAskRequest(question="hi", audience=Audience.CUSTOMER)
    assert r.audience == Audience.CUSTOMER


def test_request_audience_rejects_admin():
    with pytest.raises(ValidationError):
        SupportAskRequest(question="hi", audience="admin")


def test_response_confidence_bounds():
    SupportAskResponse(answer="a", answer_type=AnswerType.ANSWERED,
                       confidence=0.0)
    SupportAskResponse(answer="a", answer_type=AnswerType.ANSWERED,
                       confidence=1.0)
    with pytest.raises(ValidationError):
        SupportAskResponse(answer="a", answer_type=AnswerType.ANSWERED,
                           confidence=1.5)


def test_citation_snippet_length_capped():
    with pytest.raises(ValidationError):
        Citation(chunk_id="c", section_id="s", section_path="p",
                 snippet="x" * 301)


def test_llm_output_shape():
    out = LLMAnswerOutput(
        answer_type=AnswerType.ANSWERED,
        answer="text",
        citation_chunk_ids=["c1"],
        confidence=0.9,
    )
    assert out.answer_type == AnswerType.ANSWERED


def test_llm_output_limits_citations_to_three():
    with pytest.raises(ValidationError):
        LLMAnswerOutput(
            answer_type=AnswerType.ANSWERED,
            answer="text",
            citation_chunk_ids=["c1", "c2", "c3", "c4"],
            confidence=0.9,
        )
