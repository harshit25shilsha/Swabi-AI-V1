"""Tests for the scattered-number detector."""

from __future__ import annotations

import pytest

from app.capabilities.chat_safety.detectors.scattered_numbers import (
    detect_scattered_number_sequence,
)
from app.capabilities.chat_safety.schemas import GuardrailDecision

# ---------------------------------------------------------------------------
# BLOCK — the target case and close variants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        # The exact user report
        (
            "ek room chahiye, do bed ke saath, teen raat ke liye, chaar log hain, "
            "paach bje phuchenge, das bje khayenge, gyarah bje niklenge, "
            "bara bje lunch karenge, pandrah minute late honge."
        ),
        # English variant
        (
            "one room please, two beds, three nights, four guests, "
            "five o'clock we arrive, ten o'clock we eat, eleven we leave, "
            "twelve for lunch, fifteen minutes late"
        ),
        # Mixed script
        (
            "ek room, 2 bed, teen raat, 4 log, paanch baje, 10 baje, "
            "gyarah baje, 12 baje, pandrah minute"
        ),
        # Devanagari
        ("एक कमरा, दो बेड, तीन रात, चार लोग, पांच बजे, दस बजे, ग्यारह बजे, बारह बजे, पंद्रह मिनट"),
        # Plain sequence
        "one two three four five six seven eight",
        "1 2 3 4 5 6 7 8 9",
    ],
)
def test_blocks_scattered_chain(text):
    assert detect_scattered_number_sequence(text) == GuardrailDecision.BLOCK


# ---------------------------------------------------------------------------
# REVIEW — suspicious but not decisive
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "I need 3 rooms, 4 beds, 5 towels, 6 pillows, 7 chairs, 8 lamps",
        "ek 2 teen 4 paanch 6 saat",  # only 6 numbers, mixed script
    ],
)
def test_reviews_moderate_chains(text):
    assert detect_scattered_number_sequence(text) == GuardrailDecision.REVIEW


# ---------------------------------------------------------------------------
# ALLOW — normal chat and quantity lists
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Is the villa available?",
        "do din ka rent kitna hai?",
        "teen raat ke liye booking karni hai",
        "I want a 5-day family trip to Manali in December for 2 adults "
        "and two children aged 7 and 12, around 60k",
        "We need 3 rooms, 2 beds, 5 pillows, 4 towels",
        "2 adults, 2 kids, 2 nights, 2 rooms",
        "",
        "hello",
    ],
)
def test_allows_normal_chat(text):
    assert detect_scattered_number_sequence(text) == GuardrailDecision.ALLOW


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_input_allows():
    assert detect_scattered_number_sequence("") == GuardrailDecision.ALLOW


def test_too_few_numbers_allows():
    assert detect_scattered_number_sequence("ek do teen") == GuardrailDecision.ALLOW


def test_very_large_values_ignored():
    # 500, 1000 are quantities, not phone digits
    text = "rooms 500 rupees, deposit 1000 rupees, per night 250, tax 50"
    assert detect_scattered_number_sequence(text) == GuardrailDecision.ALLOW


def test_repeated_values_reduce_score():
    # 8 numbers but only 2 distinct -> not a phone number
    text = "2 rooms 2 beds 2 nights 2 pillows 2 towels 2 lamps 2 fans 2 locks"
    assert detect_scattered_number_sequence(text) != GuardrailDecision.BLOCK
