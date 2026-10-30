"""Resolver: deterministic hit, scope block, LLM fallback."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.capabilities.support.intent_llm import LLMIntentResult
from app.capabilities.support.intent_resolver import IntentResolver
from app.capabilities.support.live_data_classifier import LiveDataIntent


def _resolver(llm_result=None):
    llm = MagicMock()
    llm.classify.return_value = llm_result
    return IntentResolver(llm_classifier=llm), llm


def test_deterministic_hit_skips_llm():
    r, llm = _resolver()
    out = r.resolve("What are my bookings?", "customer")
    assert out.intent == LiveDataIntent.MY_BOOKINGS
    assert out.source == "deterministic"
    llm.classify.assert_not_called()


def test_scope_block_skips_llm():
    r, llm = _resolver()
    out = r.resolve("What is my vendor wallet balance?", "customer")
    assert out.intent == LiveDataIntent.NONE
    assert "scope" in out.reason.lower()
    llm.classify.assert_not_called()


def test_no_match_consults_llm_and_uses_result():
    r, llm = _resolver(
        llm_result=LLMIntentResult(
            intent=LiveDataIntent.MY_BOOKINGS,
            confidence=0.9,
        )
    )
    # Phrasing that the deterministic classifier does not match.
    out = r.resolve("please surface the trips I recently confirmed", "customer")
    assert out.intent == LiveDataIntent.MY_BOOKINGS
    assert out.source == "llm"
    llm.classify.assert_called_once()


def test_knowledge_question_skips_llm():
    r, llm = _resolver()

    out = r.resolve("What is the cancellation policy?", "customer")

    assert out.intent == LiveDataIntent.NONE
    assert out.reason == "not_live_intent_candidate"
    llm.classify.assert_not_called()


def test_public_catalogue_request_consults_llm():
    r, llm = _resolver(
        llm_result=LLMIntentResult(
            intent=LiveDataIntent.PACKAGE_SEARCH,
            confidence=0.9,
        )
    )

    out = r.resolve("Can you show available packages in Goa?", "customer")

    assert out.intent == LiveDataIntent.PACKAGE_SEARCH
    llm.classify.assert_called_once()


def test_direct_package_search_uses_public_intent_without_entity_extraction():
    r, llm = _resolver()

    out = r.resolve(
        "provide me available packages in swabi platform for winter that i can book",
        "customer",
    )

    assert out.intent == LiveDataIntent.PACKAGE_SEARCH
    assert out.entities == {}
    assert out.source == "deterministic"
    llm.classify.assert_not_called()


def test_seasonal_package_search_keeps_extracted_destination():
    r, llm = _resolver(
        llm_result=LLMIntentResult(
            intent=LiveDataIntent.PACKAGE_SEARCH,
            entities={"query": "Goa"},
            confidence=0.9,
        )
    )

    out = r.resolve("Which packages are available for winter in Goa?", "customer")

    assert out.intent == LiveDataIntent.PACKAGE_SEARCH
    assert out.entities == {"query": "Goa"}
    assert out.source == "llm"
    llm.classify.assert_called_once()


def test_package_code_detail_uses_deterministic_public_intent():
    r, llm = _resolver()

    out = r.resolve("Tell me the details for package PKG-922", "customer")

    assert out.intent == LiveDataIntent.PACKAGE_DETAIL
    assert out.entities == {"package_code": "PKG-922"}
    assert out.source == "deterministic"
    llm.classify.assert_not_called()


def test_general_travel_question_skips_llm():
    r, llm = _resolver()

    out = r.resolve("what is the weather on mars", "customer")

    assert out.intent == LiveDataIntent.NONE
    assert out.source == "none"
    llm.classify.assert_not_called()


@pytest.mark.parametrize(
    ("question", "audience"),
    [
        ("I paid for a trip but need to cancel. How is my refund decided?", "customer"),
        ("What happens after a vendor submits a withdrawal request?", "vendor"),
        ("What should I do if the vendor does not show up for my booking?", "customer"),
        ("how can i book package in swabi platform using enquiries", "customer"),
        ("How can I book a package in the Swabi platform?", "customer"),
    ],
)
def test_process_questions_skip_live_intent_resolution(question, audience):
    r, llm = _resolver()

    out = r.resolve(question, audience)

    assert out.intent == LiveDataIntent.NONE
    assert out.reason == "general_process_question"
    llm.classify.assert_not_called()


def test_llm_unavailable_returns_none():
    llm = MagicMock()
    llm.classify.return_value = None
    r = IntentResolver(llm_classifier=llm)
    out = r.resolve("please surface the trips I recently confirmed", "customer")
    assert out.intent == LiveDataIntent.NONE
    assert out.source == "none"
