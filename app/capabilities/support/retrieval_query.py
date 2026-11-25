"""Prepare support questions for retrieval without changing the user prompt."""

from __future__ import annotations

import re

from app.capabilities.support.live_data_check import is_general_process_question

_BOOKING_TERMS = re.compile(r"\b(book|booking|reserve|reservation)\b", re.IGNORECASE)
_BOOKING_FLOW_TERMS = re.compile(r"\b(packages?|enquir(?:y|ies)|bids?)\b", re.IGNORECASE)
_PLATFORM_IDENTITY_QUESTION = re.compile(
    r"\bwhat\s+is\s+(?:swabi|(?:this|the)\s+platform)(?:\s+is)?\s+for\b"
    r"|\bwhat\s+is\s+(?:swabi|(?:this|the)\s+platform)\s*[?.!]*$"
    r"|\bwhat\s+does\s+(?:swabi|(?:this|the)\s+platform)\s+do\b"
    r"|\b(?:tell\s+me\s+about|describe)\s+swabi\s*[?.!]*$"
    r"|\bpurpose\s+of\s+(?:swabi|(?:this|the)\s+platform)\b",
    re.IGNORECASE,
)


def prepare_retrieval_query(question: str) -> str:
    """Expand short product-definition and documented booking questions."""
    if _PLATFORM_IDENTITY_QUESTION.search(question):
        return (
            "Swabi platform overview and purpose: online travel marketplace "
            "connecting customers or travelers with independent travel-service "
            "vendors. Customers post trip enquiries, review vendor bids, browse "
            "travel packages and activities, and use Swabi's booking and payment "
            "process, including escrow where applicable. Swabi acts as an "
            "intermediary and payment facilitator; vendors provide the travel "
            "services."
        )

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
