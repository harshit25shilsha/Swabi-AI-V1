"""Support retrieval query expansions."""

from __future__ import annotations

import pytest

from app.capabilities.support.retrieval_query import prepare_retrieval_query


@pytest.mark.parametrize(
    "question",
    [
        "what is Swabi?",
        "what is this platform for?",
        "What does the platform do?",
        "Tell me about Swabi",
        "What is the purpose of this platform?",
    ],
)
def test_platform_definition_questions_use_overview_query(question):
    query = prepare_retrieval_query(question)

    assert "Swabi platform overview and purpose" in query
    assert "travel marketplace" in query
    assert "independent travel-service vendors" in query
    assert "escrow where applicable" in query


def test_unrelated_question_is_not_rewritten():
    question = "What is the cancellation policy?"

    assert prepare_retrieval_query(question) == question


@pytest.mark.parametrize(
    "question",
    [
        "What is the cancellation policy on this platform?",
        "What is Swabi's cancellation policy?",
        "What does this platform charge for?",
    ],
)
def test_platform_policy_questions_keep_their_original_retrieval_query(question):
    assert prepare_retrieval_query(question) == question
