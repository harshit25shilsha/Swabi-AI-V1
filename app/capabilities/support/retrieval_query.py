"""Prepare support questions for retrieval without changing the user prompt."""

from __future__ import annotations

import re

from app.capabilities.support.live_data_check import is_general_process_question

_BOOKING_TERMS = re.compile(r"\b(book|booking|reserve|reservation)\b", re.IGNORECASE)
_BOOKING_FLOW_TERMS = re.compile(
    r"\b(packages?|enquir(?:y|ies)|bids?)\b", re.IGNORECASE
)


def prepare_retrieval_query(question: str) -> str:
    """Expand package-booking questions into the documented enquiry workflow."""
    if (
        is_general_process_question(question)
        and _BOOKING_TERMS.search(question)
        and _BOOKING_FLOW_TERMS.search(question)
    ):
        return (
            "Swabi customer booking process: post a trip enquiry with "
            "requirements, vendors place bids, customer reviews bids and "
            "chats or negotiates, customer accepts a bid, booking is created "
            "and payment starts."
        )
    return question
