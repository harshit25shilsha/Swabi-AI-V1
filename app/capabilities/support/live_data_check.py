"""Deterministic live-data pre-check.

Questions about the caller's own bookings, bids, payments, wallet,
escrow, refunds, disputes, tracking devices, or live availability are
not answerable from RAG. They must be answered by the Swabi backend
(Batch 2D). Until then they return needs_live_data with a clear message.

This is a pre-check: the LLM is never consulted for these.
"""
from __future__ import annotations

import re

_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Personal state
    re.compile(
        r"\bmy\s+(booking|bookings|bid|bids|payment|payments|order|orders"
        r"|wallet|escrow|refund|refunds|dispute|disputes|withdrawal"
        r"|device|devices|account\s+balance)\b",
        re.IGNORECASE,
    ),
    # Status of specific things
    re.compile(
        r"\bstatus\s+(of|for)\s+(my\s+)?"
        r"(booking|bid|order|payment|refund|dispute|withdrawal|device)\b",
        re.IGNORECASE,
    ),
    # Live availability / pricing
    re.compile(
        r"\b(current\s+price|current\s+availability|available\s+(this|next|today|tomorrow|tonight))",
        re.IGNORECASE,
    ),
    # Tracking
    re.compile(
        r"\b(track|tracking)\s+(my\s+)?(trip|booking|device)\b",
        re.IGNORECASE,
    ),
    # Money paid / owed
    re.compile(
        r"\bhow\s+much\s+did\s+i\s+pay\b",
        re.IGNORECASE,
    ),
)


def is_live_data_question(question: str) -> bool:
    """True if the question requires current marketplace state."""
    if not question:
        return False
    return any(p.search(question) for p in _PATTERNS)
