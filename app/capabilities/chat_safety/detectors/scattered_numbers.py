"""Detect a phone number (or similar identifier) hidden as scattered numbers.

Unlike `number_words` (which requires adjacent digit-words) this detector
looks at the **aggregate** pattern: N numbers spread across otherwise
innocent content, each in its own clause, separated by ordinary words.

Returns a tri-state GuardrailDecision:
    BLOCK    — strong evidence of a hidden phone number
    REVIEW   — suspicious; escalate even if the LLM says ALLOW
    ALLOW    — no signal

Design:
    * Chain analysis: consecutive numeric tokens separated by at most
      MAX_GAP ordinary words are considered "chained".
    * Distinct-value count within the chain drives the decision.
    * Value filter: only values 0-99 count (phone-number-like). Larger
      values (e.g. 500, 2000) are more likely quantities than digits.
    * Repeats within the chain reduce the score: a list like
      "2 people, 2 nights, 2 rooms" is not a phone number.

The detector intentionally accepts some REVIEW false positives on dense
quantity lists — REVIEW means human look, not auto-block. It is designed
to never silently pass a distributed phone number.
"""

from __future__ import annotations

import re

from app.capabilities.chat_safety.detectors.normalize import normalize
from app.capabilities.chat_safety.schemas import GuardrailDecision

# Value map — spelled numbers and raw digits.
# Covers English, Hinglish, Devanagari. Extend here as new variants appear.

_VALUE_MAP: dict[str, int] = {
    # Raw digits 0-99 handled separately (see _value_of) — this map holds
    # only word forms.
    # English 0-15
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    # Hinglish
    "shunya": 0,
    "sifar": 0,
    "ek": 1,
    "do": 2,
    "teen": 3,
    "tin": 3,
    "char": 4,
    "chaar": 4,
    "panch": 5,
    "paanch": 5,
    "paach": 5,
    "chhe": 6,
    "chhah": 6,
    "che": 6,
    "saat": 7,
    "sat": 7,
    "aath": 8,
    "ath": 8,
    "nau": 9,
    "das": 10,
    "dus": 10,
    "gyarah": 11,
    "gyara": 11,
    "gyareh": 11,
    "barah": 12,
    "barha": 12,
    "bara": 12,
    "terah": 13,
    "terha": 13,
    "tera": 13,
    "chaudah": 14,
    "chauda": 14,
    "pandrah": 15,
    "pandra": 15,
    "pandhra": 15,
    # Devanagari
    "शून्य": 0,
    "सिफ़र": 0,
    "सिफर": 0,
    "एक": 1,
    "दो": 2,
    "तीन": 3,
    "चार": 4,
    "पांच": 5,
    "पाँच": 5,
    "पाच": 5,
    "छह": 6,
    "छः": 6,
    "सात": 7,
    "आठ": 8,
    "नौ": 9,
    "दस": 10,
    "ग्यारह": 11,
    "बारह": 12,
    "तेरह": 13,
    "चौदह": 14,
    "पंद्रह": 15,
}

_TOKEN_RE = re.compile(r"[\w\u0900-\u097F]+", re.UNICODE)

# Two numeric tokens are "chained" if at most this many ordinary words
# sit between them. Four allows "bje phuchenge" (2), "ke saath" (2),
# "minute late honge" (3) etc., while still resetting on long sentences.
_MAX_GAP = 4

# Value range that counts as "phone-number-like".
_MAX_VALUE = 99

# Decision thresholds, based on the size of the largest chain.
# Tuned so that:
#   * User's leak (9 unique numbers) -> BLOCK
#   * Typical chat (0-3 numbers)    -> ALLOW
#   * Quantity list with repeats    -> ALLOW or REVIEW
_MIN_CHAIN_FOR_BLOCK = 8
_MIN_CHAIN_FOR_REVIEW = 6


def _value_of(token: str) -> int | None:
    """Return the integer value of a token, or None.

    Handles: raw digits ("42"), word forms ("paach"), Devanagari.
    Only values in 0.._MAX_VALUE count.
    """
    if token in _VALUE_MAP:
        v = _VALUE_MAP[token]
        return v if 0 <= v <= _MAX_VALUE else None
    # Raw digits
    if token.isdigit():
        try:
            v = int(token)
        except ValueError:
            return None
        return v if 0 <= v <= _MAX_VALUE else None
    return None


def _find_chains(numeric: list[tuple[int, int]]) -> list[list[int]]:
    """Group consecutive numeric positions into chains.

    ``numeric`` is a list of (token_index, value), sorted by index.
    Returns a list of chains, where each chain is the list of values.
    A chain ends when the gap to the next number exceeds _MAX_GAP.
    """
    if not numeric:
        return []
    chains: list[list[int]] = [[numeric[0][1]]]
    for prev, curr in zip(numeric, numeric[1:], strict=False):
        gap = curr[0] - prev[0] - 1
        if gap <= _MAX_GAP:
            chains[-1].append(curr[1])
        else:
            chains.append([curr[1]])
    return chains


def detect_scattered_number_sequence(text: str) -> GuardrailDecision:
    """Tri-state detection for scattered-number phone leakage."""
    if not text:
        return GuardrailDecision.ALLOW

    tokens = _TOKEN_RE.findall(normalize(text).lower())
    if len(tokens) < 4:
        return GuardrailDecision.ALLOW

    numeric: list[tuple[int, int]] = []
    for i, tok in enumerate(tokens):
        v = _value_of(tok)
        if v is not None:
            numeric.append((i, v))

    if len(numeric) < 4:
        return GuardrailDecision.ALLOW

    chains = _find_chains(numeric)

    # Score each chain by the number of *distinct* values.
    # Repeats reduce the score: a genuine quantity list often repeats values
    # ("2 rooms, 2 beds, 2 nights"), a phone number rarely does.
    best_score = 0
    for chain in chains:
        distinct = len(set(chain))
        # Penalize repeats: if only half the values are distinct, halve.
        if distinct < len(chain):
            distinct = max(1, distinct)
        best_score = max(best_score, distinct)

    if best_score >= _MIN_CHAIN_FOR_BLOCK:
        return GuardrailDecision.BLOCK
    if best_score >= _MIN_CHAIN_FOR_REVIEW:
        return GuardrailDecision.REVIEW
    return GuardrailDecision.ALLOW
