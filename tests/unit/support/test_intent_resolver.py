"""Resolver: deterministic hit, scope block, LLM fallback."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.capabilities.support.intent_llm import LLMIntentResult
from app.capabilities.support.intent_resolver import IntentResolver
from app.capabilities.support.live_data_classifier import LiveDataIntent
from app.capabilities.support.schemas import ConversationTurn


def _resolver(llm_result=None):
    llm = MagicMock()
    llm.classify.return_value = llm_result
    return IntentResolver(llm_classifier=llm), llm


def test_list_intent_uses_structured_source_plan():
    r, llm = _resolver(
        llm_result=LLMIntentResult(
            intent=LiveDataIntent.MY_BOOKINGS,
            requires_knowledge=False,
            confidence=0.9,
        )
    )
    out = r.resolve("What are my bookings?", "customer")
    assert out.intent == LiveDataIntent.MY_BOOKINGS
    assert out.source == "llm"
    llm.classify.assert_called_once()


def test_customer_transaction_history_routes_to_payment_history():
    r, llm = _resolver(
        llm_result=LLMIntentResult(
            intent=LiveDataIntent.MY_PAYMENTS,
            requires_knowledge=False,
            confidence=0.9,
        )
    )

    out = r.resolve("Show my transaction history", "customer")

    assert out.intent == LiveDataIntent.MY_PAYMENTS
    assert out.source == "llm"
    llm.classify.assert_called_once()


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
            requires_knowledge=False,
            confidence=0.9,
        )
    )
    # Phrasing that the deterministic classifier does not match.
    out = r.resolve("please surface the trips I recently confirmed", "customer")
    assert out.intent == LiveDataIntent.MY_BOOKINGS
    assert out.source == "llm"
    llm.classify.assert_called_once()


def test_knowledge_question_uses_structured_source_plan():
    r, llm = _resolver(
        llm_result=LLMIntentResult(
            intent=LiveDataIntent.NONE,
            requires_knowledge=True,
            knowledge_query="What is the cancellation policy?",
            confidence=0.95,
        )
    )

    out = r.resolve("What is the cancellation policy?", "customer")

    assert out.intent == LiveDataIntent.NONE
    assert out.reason == "llm_none"
    assert out.use_rag is True
    assert out.knowledge_query == "What is the cancellation policy?"
    llm.classify.assert_called_once()


def test_cancellation_instructions_are_not_routed_to_bookings_api():
    r, llm = _resolver(
        llm_result=LLMIntentResult(
            intent=LiveDataIntent.NONE,
            requires_knowledge=True,
            knowledge_query="How can I cancel my booking and get a refund?",
            confidence=0.95,
        )
    )

    out = r.resolve("How can I cancel my booking and get a refund?", "customer")

    assert out.intent == LiveDataIntent.NONE
    assert out.reason == "llm_none"
    assert out.use_rag is True
    llm.classify.assert_called_once()


def test_policy_and_booking_status_question_uses_both_sources():
    r, llm = _resolver(
        llm_result=LLMIntentResult(
            intent=LiveDataIntent.MY_BOOKINGS,
            requires_knowledge=True,
            knowledge_query="What is the cancellation policy?",
            confidence=0.95,
        )
    )

    out = r.resolve(
        "What is the cancellation policy and what is the status of my booking?",
        "customer",
    )

    assert out.intent == LiveDataIntent.MY_BOOKINGS
    assert out.use_rag is True
    assert out.source == "llm"
    llm.classify.assert_called_once()


def test_latest_bookings_and_cancellation_policy_uses_both_sources():
    r, llm = _resolver(
        llm_result=LLMIntentResult(
            intent=LiveDataIntent.MY_BOOKINGS,
            requires_knowledge=True,
            knowledge_query="Explain the cancellation policy.",
            confidence=0.95,
        )
    )

    out = r.resolve(
        "Show my latest bookings and explain the cancellation policy.",
        "customer",
    )

    assert out.intent == LiveDataIntent.MY_BOOKINGS
    assert out.use_rag is True
    assert out.source == "llm"
    llm.classify.assert_called_once()


def test_popular_destinations_and_booking_process_uses_both_sources():
    r, llm = _resolver(
        llm_result=LLMIntentResult(
            intent=LiveDataIntent.POPULAR_DESTINATIONS,
            requires_knowledge=True,
            knowledge_query="How can I book a package on this platform?",
            confidence=0.95,
        )
    )

    out = r.resolve(
        "provide popular destinations in this platform and how can i book package in this platform",
        "customer",
    )

    assert out.intent == LiveDataIntent.POPULAR_DESTINATIONS
    assert out.use_rag is True
    assert out.source == "llm"
    llm.classify.assert_called_once()


def test_coins_balance_and_bidding_workflow_plan_both_sources():
    r, llm = _resolver(
        llm_result=LLMIntentResult(
            intent=LiveDataIntent.MY_COINS,
            requires_knowledge=True,
            knowledge_query="How does bidding work on Swabi?",
            confidence=0.97,
        )
    )

    out = r.resolve(
        "provide me my swabi coins balance and how bidding works in this platform",
        "customer",
    )

    assert out.intent == LiveDataIntent.MY_COINS
    assert out.use_rag is True
    assert out.knowledge_query == "How does bidding work on Swabi?"
    llm.classify.assert_called_once()


def test_how_to_check_booking_status_does_not_add_rag():
    r, llm = _resolver(
        llm_result=LLMIntentResult(
            intent=LiveDataIntent.MY_BOOKINGS,
            requires_knowledge=False,
            confidence=0.95,
        )
    )

    out = r.resolve("How can I check my booking status?", "customer")

    assert out.intent == LiveDataIntent.MY_BOOKINGS
    assert out.use_rag is False
    llm.classify.assert_called_once()


def test_public_catalogue_request_consults_llm():
    r, llm = _resolver(
        llm_result=LLMIntentResult(
            intent=LiveDataIntent.PACKAGE_SEARCH,
            requires_knowledge=False,
            confidence=0.9,
        )
    )

    out = r.resolve("Can you show available packages in Goa?", "customer")

    assert out.intent == LiveDataIntent.PACKAGE_SEARCH
    llm.classify.assert_called_once()


def test_direct_unfiltered_package_search_uses_plan_and_extracts_count():
    r, llm = _resolver(
        llm_result=LLMIntentResult(
            intent=LiveDataIntent.PACKAGE_SEARCH,
            entities={"query": ""},
            requested_limit=2,
            requires_knowledge=False,
            confidence=0.95,
        )
    )

    out = r.resolve(
        "provide me only 2 packages available in this platform",
        "customer",
    )

    assert out.intent == LiveDataIntent.PACKAGE_SEARCH
    assert out.entities == {"query": ""}
    assert out.requested_limit == 2
    assert out.source == "llm"
    llm.classify.assert_called_once()


def test_location_based_package_search_keeps_destination_and_result_limit():
    r, llm = _resolver(
        llm_result=LLMIntentResult(
            intent=LiveDataIntent.PACKAGE_SEARCH,
            requested_limit=2,
            requires_knowledge=False,
            confidence=0.95,
        )
    )

    out = r.resolve("provide me 2 packages uttarakhand based", "customer")

    assert out.intent == LiveDataIntent.PACKAGE_SEARCH
    assert out.entities == {"query": "uttarakhand"}
    assert out.requested_limit == 2
    assert out.source == "llm"
    llm.classify.assert_called_once()


def test_destination_before_based_package_search_keeps_full_destination():
    r, llm = _resolver(
        llm_result=LLMIntentResult(
            intent=LiveDataIntent.PACKAGE_SEARCH,
            requested_limit=2,
            requires_knowledge=False,
            confidence=0.95,
        )
    )

    out = r.resolve("provide me 2 uttar pradesh based packages", "customer")

    assert out.intent == LiveDataIntent.PACKAGE_SEARCH
    assert out.entities == {"query": "uttar pradesh"}
    assert out.requested_limit == 2
    assert out.source == "llm"
    llm.classify.assert_called_once()


def test_filtered_package_request_still_uses_semantic_classification():
    r, llm = _resolver(
        llm_result=LLMIntentResult(
            intent=LiveDataIntent.PACKAGE_SEARCH,
            entities={"query": ""},
            requires_knowledge=False,
            confidence=0.9,
        )
    )

    out = r.resolve(
        "provide me available packages in swabi platform for winter that i can book",
        "customer",
    )

    assert out.intent == LiveDataIntent.PACKAGE_SEARCH
    assert out.source == "llm"
    llm.classify.assert_called_once()


def test_package_search_plan_preserves_location():
    r, llm = _resolver(
        llm_result=LLMIntentResult(
            intent=LiveDataIntent.PACKAGE_SEARCH,
            requires_knowledge=False,
            confidence=0.9,
        )
    )

    out = r.resolve("Show me packages in Goa", "customer")

    assert out.intent == LiveDataIntent.PACKAGE_SEARCH
    assert out.entities == {"query": "Goa"}
    assert out.source == "llm"
    llm.classify.assert_called_once()


def test_seasonal_package_search_keeps_extracted_destination():
    r, llm = _resolver(
        llm_result=LLMIntentResult(
            intent=LiveDataIntent.PACKAGE_SEARCH,
            entities={"query": "Goa"},
            requires_knowledge=False,
            confidence=0.9,
        )
    )

    out = r.resolve("Which packages are available for winter in Goa?", "customer")

    assert out.intent == LiveDataIntent.PACKAGE_SEARCH
    assert out.entities == {"query": "Goa"}
    assert out.source == "llm"
    llm.classify.assert_called_once()


def test_package_code_detail_uses_planned_public_intent():
    r, llm = _resolver(
        llm_result=LLMIntentResult(
            intent=LiveDataIntent.PACKAGE_DETAIL,
            entities={"package_code": "PKG-922"},
            requires_knowledge=False,
            confidence=0.95,
        )
    )

    out = r.resolve("Tell me the details for package PKG-922", "customer")

    assert out.intent == LiveDataIntent.PACKAGE_DETAIL
    assert out.entities == {"package_code": "PKG-922"}
    assert out.source == "llm"
    llm.classify.assert_called_once()


def test_general_travel_question_uses_knowledge_path():
    r, llm = _resolver(
        llm_result=LLMIntentResult(
            intent=LiveDataIntent.NONE,
            requires_knowledge=False,
            confidence=0.95,
        )
    )

    out = r.resolve("what is the weather on mars", "customer")

    assert out.intent == LiveDataIntent.NONE
    assert out.source == "llm"
    llm.classify.assert_called_once()


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
def test_process_questions_are_planned_for_knowledge(question, audience):
    r, llm = _resolver(
        llm_result=LLMIntentResult(
            intent=LiveDataIntent.NONE,
            requires_knowledge=True,
            knowledge_query=question,
            confidence=0.95,
        )
    )

    out = r.resolve(question, audience)

    assert out.intent == LiveDataIntent.NONE
    assert out.use_rag is True
    assert out.knowledge_query == question
    llm.classify.assert_called_once()


def test_llm_unavailable_returns_none():
    llm = MagicMock()
    llm.classify.return_value = None
    r = IntentResolver(llm_classifier=llm)
    out = r.resolve("please surface the trips I recently confirmed", "customer")
    assert out.intent == LiveDataIntent.NONE
    assert out.source == "none"
    assert out.use_rag is True


@pytest.mark.parametrize(
    ("question", "intent", "limit"),
    [
        ("Show me my last two payment history", LiveDataIntent.MY_PAYMENTS, 2),
        ("Could you list a couple of my bookings?", LiveDataIntent.MY_BOOKINGS, 2),
    ],
)
def test_llm_result_preserves_requested_list_limit(question, intent, limit):
    r, llm = _resolver(
        llm_result=LLMIntentResult(
            intent=intent,
            requested_limit=limit,
            requires_knowledge=False,
            confidence=0.95,
        )
    )

    out = r.resolve(question, "customer")

    assert out.intent == intent
    assert out.requested_limit == limit
    assert out.source == "llm"


def test_explicit_count_fallback_is_applied_when_llm_omits_limit():
    r, _ = _resolver(
        llm_result=LLMIntentResult(
            intent=LiveDataIntent.MY_PAYMENTS,
            requires_knowledge=False,
            confidence=0.95,
        )
    )

    out = r.resolve("Show me my last two payment history", "customer")

    assert out.requested_limit == 2


def test_common_count_phrase_is_preserved_when_llm_omits_limit():
    r, _ = _resolver(
        llm_result=LLMIntentResult(
            intent=LiveDataIntent.MY_BOOKINGS,
            requires_knowledge=False,
            confidence=0.95,
        )
    )

    out = r.resolve("List a couple of my bookings", "customer")

    assert out.requested_limit == 2


@pytest.mark.parametrize(
    "question",
    [
        "Show packages under 2,000",
        "Find packages for 2 travelers",
    ],
)
def test_fallback_does_not_confuse_filters_with_result_limits(question):
    r, _ = _resolver(
        llm_result=LLMIntentResult(
            intent=LiveDataIntent.PACKAGE_SEARCH,
            requires_knowledge=False,
            confidence=0.95,
        )
    )

    out = r.resolve(question, "customer")

    assert out.requested_limit is None


def test_short_followup_is_classified_with_recent_context():
    history = [
        ConversationTurn(role="user", content="Show me packages in Goa"),
        ConversationTurn(role="assistant", content="I found several packages."),
    ]
    r, llm = _resolver(
        llm_result=LLMIntentResult(
            intent=LiveDataIntent.PACKAGE_SEARCH,
            entities={"query": "Goa"},
            requested_limit=2,
            requires_knowledge=False,
            confidence=0.95,
        )
    )

    out = r.resolve("Only two, please", "customer", conversation_history=history)

    assert out.intent == LiveDataIntent.PACKAGE_SEARCH
    assert out.entities == {"query": "Goa"}
    assert out.requested_limit == 2
    assert llm.classify.call_args.kwargs["conversation_history"] == history
