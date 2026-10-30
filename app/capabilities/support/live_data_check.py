"""Deterministic live-data pre-check.

Questions about the caller's own bookings, bids, payments, wallet,
escrow, refunds, disputes, tracking devices, or live availability are
not answerable from RAG. They must be answered by the Swabi backend
(Batch 2D). Until then they return needs_live_data with a clear message.

This is a pre-check: the LLM is never consulted for these.
"""
from __future__ import annotations

import re

_CURRENT_STATE_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Explicit status or history requests.
    re.compile(
        r"\b(status|history|transactions?)\s+(of|for)\s+(my\s+)?"
        r"(booking|bid|order|payment|refund|dispute|withdrawal|wallet)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(where|when)\s+(is|are|was|were|will)\s+(my\s+)?"
        r"(refund|payment|withdrawal|booking|order)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(has|have)\s+my\s+(refund|payment|withdrawal|booking|order)\s+"
        r"(been\s+)?(processed|approved|released|confirmed|cancelled|canceled)\b",
        re.IGNORECASE,
    ),
    # Direct requests to inspect personal account data.
    re.compile(
        r"\b(show|list|check|get|view)\s+(me\s+)?my\s+"
        r"(bookings?|trips?|bids?|payments?|orders?|wallet|escrow|"
        r"disputes?|withdrawals?|devices?|enquir(?:y|ies))\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bmy\s+wallet\s+balance\b", re.IGNORECASE),
    re.compile(r"\bhow\s+much\s+did\s+i\s+pay\b", re.IGNORECASE),
    re.compile(
        r"\b(track|tracking)\s+(my\s+)?(trip|booking|device)\b",
        re.IGNORECASE,
    ),
    # Live availability / pricing.
    re.compile(
        r"\b(current\s+price|current\s+availability|available\s+"
        r"(this|next|today|tomorrow|tonight))\b",
        re.IGNORECASE,
    ),
)

_PERSONAL_STATE_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Personal state
    re.compile(
        r"\bmy\s+(booking|bookings|bid|bids|payment|payments|order|orders"
        r"|wallet|escrow|dispute|disputes|withdrawal"
        r"|device|devices|account\s+balance)\b",
        re.IGNORECASE,
    ),
)

_GENERAL_PROCESS_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\bwhat\s+(happens|should\s+i\s+do|can\s+i\s+do)\b"
        r"|\bhow\s+do\s+i\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bhow\s+can\s+i\b.{0,100}\b"
        r"(book|create|post|submit|accept)\b.{0,100}\b"
        r"(packages?|enquir(?:y|ies)|bids?|bookings?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bhow\s+(does|do|is|are)\b.{0,100}\b"
        r"(work|decided|calculated|determined|handled|processed)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(steps?|process|eligibility|minimum\s+(?:amount|withdrawal))\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bwhat\s+if\b|\bif\b.{0,80}\b(does\s+not|doesn't|fails\s+to)\s+show\b",
        re.IGNORECASE,
    ),
)


def is_general_process_question(question: str) -> bool:
    """True for policy/how-to questions that can be answered from knowledge."""
    return bool(question) and any(
        pattern.search(question) for pattern in _GENERAL_PROCESS_PATTERNS
    )


def is_live_data_question(question: str) -> bool:
    """True when wording explicitly requests current, personal marketplace data."""
    if not question:
        return False
    if any(pattern.search(question) for pattern in _CURRENT_STATE_PATTERNS):
        return True
    if is_general_process_question(question):
        return False
    return any(pattern.search(question) for pattern in _PERSONAL_STATE_PATTERNS)
