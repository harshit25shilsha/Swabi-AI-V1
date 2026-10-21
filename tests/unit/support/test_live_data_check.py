"""Live-data pre-check tests."""
from __future__ import annotations

import pytest

from app.capabilities.support.live_data_check import is_live_data_question


@pytest.mark.parametrize("q", [
    "What is the status of my booking?",
    "Show me my bids",
    "Where is my refund?",
    "What is my wallet balance?",
    "Track my trip please",
    "Current price for that package",
    "Is it available this weekend?",
    "How much did I pay for the Manali trip?",
])
def test_live_data_questions_flagged(q):
    assert is_live_data_question(q) is True


@pytest.mark.parametrize("q", [
    "What is the cancellation policy?",
    "How does escrow work?",
    "What activities are good for families in Kashmir?",
    "Tell me about Swabi Plus",
    "What is the vendor commission?",
])
def test_policy_questions_not_flagged(q):
    assert is_live_data_question(q) is False


def test_empty_string_not_flagged():
    assert is_live_data_question("") is False
