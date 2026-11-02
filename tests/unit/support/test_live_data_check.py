"""Live-data pre-check tests."""

from __future__ import annotations

import pytest

from app.capabilities.support.live_data_check import is_live_data_question


@pytest.mark.parametrize(
    "q",
    [
        "What is the status of my booking?",
        "Show me my bids",
        "Where is my refund?",
        "What is my wallet balance?",
        "Track my trip please",
        "Current price for that package",
        "Is it available this weekend?",
        "How much did I pay for the Manali trip?",
    ],
)
def test_live_data_questions_flagged(q):
    assert is_live_data_question(q) is True


@pytest.mark.parametrize(
    "q",
    [
        "What is the cancellation policy?",
        "How does escrow work?",
        "I paid for a trip but need to cancel. How is my refund decided?",
        "What happens after a vendor submits a withdrawal request?",
        "What should I do if the vendor does not show up for my booking?",
        "How do I cancel my booking?",
        "How can I cancel my booking and get a refund?",
        "What activities are good for families in Kashmir?",
        "Tell me about Swabi Plus",
        "What is the vendor commission?",
    ],
)
def test_policy_questions_not_flagged(q):
    assert is_live_data_question(q) is False


def test_empty_string_not_flagged():
    assert is_live_data_question("") is False


def test_cancellation_instructions_are_a_knowledge_question():
    from app.capabilities.support.live_data_check import is_general_process_question

    question = "How can I cancel my booking and get a refund?"

    assert is_general_process_question(question) is True
    assert is_live_data_question(question) is False


def test_policy_and_account_status_question_is_mixed():
    from app.capabilities.support.live_data_check import is_general_process_question

    question = "What is the cancellation policy and what is the status of my booking?"

    assert is_general_process_question(question) is True
    assert is_live_data_question(question) is True


def test_how_to_inspect_booking_status_is_live_only():
    from app.capabilities.support.live_data_check import is_general_process_question

    question = "How can I check my booking status?"

    assert is_general_process_question(question) is False
    assert is_live_data_question(question) is True
