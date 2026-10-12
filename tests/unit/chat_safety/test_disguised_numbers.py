"""
Unit tests for the deterministic layer. LLM-only cases are covered by the
integration script tests/test_integration_llm.py (runs against the real
Groq API).
"""

import pytest

from app.capabilities.chat_safety.detectors.encoded_digits import detect_encoded_digits
from app.capabilities.chat_safety.detectors.number_words import detect_number_word_sequence


@pytest.mark.parametrize(
    "text",
    [
        "do din ka rent kitna hai?",
        "teen raat ke liye booking karni hai",
        "paanch sau rupaye discount milega?",
        # NOTE: a run of 4+ spelled digits ("eight seven three four") is
        # blocked by design (see test_number_words.py), so it is not listed
        # as normal chat.
        "room 2 has two more windows than room 1",
        "Is the villa available from 10th to 12th October?",
        "What payment methods does the platform support?",
    ],
)
def test_no_false_positive_on_normal_chat(text):
    assert detect_number_word_sequence(text) is False
    assert detect_encoded_digits(text) is False
