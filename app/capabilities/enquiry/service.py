"""Enquiry Understanding orchestrator.

Steps:
    1. Call the LLM via LLMGateway.generate_structured_sync.
    2. Validate against EnquiryExtraction.
    3. Apply the currency hint if the model left it null.
    4. Deterministically compute the missing-field list.
    5. Build clarifying questions for missing required + askable fields.
    6. Assemble the response with a completeness score.

Synchronous, matching Chat Safety.
"""

from __future__ import annotations

import logging

from app.capabilities.enquiry.constants import (
    FIELD_ACCOMMODATION,
    FIELD_ACTIVITIES,
    FIELD_BUDGET,
    FIELD_DESCRIPTION,
    FIELD_DESTINATION,
    FIELD_DURATION,
    FIELD_MEALS,
    FIELD_SPECIAL_INSTRUCTIONS,
    FIELD_TOUR_GUIDE,
    FIELD_TRANSPORT,
    FIELD_TRAVEL_DATE,
    FIELD_TRAVELERS,
    FIELD_WHEN,
    MAX_CLARIFYING_QUESTIONS,
    REQUIRED_FIELDS,
)
from app.capabilities.enquiry.prompts import (
    ENQUIRY_PROMPT_VERSION,
    ENQUIRY_SYSTEM_PROMPT,
)
from app.capabilities.enquiry.schemas import (
    ClarificationQuestion,
    EnquiryExtraction,
    EnquiryUnderstandResponse,
)
from app.llm.gateway import LLMGateway

logger = logging.getLogger(__name__)


_QUESTION_TEMPLATES: dict[str, str] = {
    FIELD_DESTINATION: "Where would you like to go?",
    FIELD_WHEN: "When are you planning to travel, and for how long?",
    FIELD_DURATION: "How many nights and days?",
    FIELD_TRAVEL_DATE: "What date would you like to start?",
    FIELD_TRAVELERS: "How many travellers?",
    FIELD_BUDGET: "Do you have a budget range in mind?",
    FIELD_ACCOMMODATION: "What type of accommodation — hotel, villa, houseboat?",
    FIELD_TRANSPORT: "What transport would you prefer?",
    FIELD_MEALS: "Which meals should be included?",
    FIELD_TOUR_GUIDE: "Would you like a tour guide?",
    FIELD_ACTIVITIES: "Any activities you'd like to include?",
    FIELD_SPECIAL_INSTRUCTIONS: "Any special instructions or requests?",
    FIELD_DESCRIPTION: "Could you describe the trip briefly?",
}


# Missing field detection — deterministic


def detect_missing_fields(extraction: EnquiryExtraction) -> list[str]:
    """Return field names that are absent or unusable.

    Deterministic order: required fields (in REQUIRED_FIELDS order),
    then askable fields.
    """
    missing: list[str] = []
    req = extraction.requirements

    # Required
    if not extraction.destination:
        missing.append(FIELD_DESTINATION)

    if (
        extraction.travel_date is None
        and extraction.duration_days is None
        and extraction.duration_nights is None
    ):
        missing.append(FIELD_WHEN)

    if extraction.travelers is None:
        missing.append(FIELD_TRAVELERS)

    if not req.accommodation:
        missing.append(FIELD_ACCOMMODATION)

    if not req.transport:
        missing.append(FIELD_TRANSPORT)

    if not req.meals:
        missing.append(FIELD_MEALS)

    if not req.tour_guide:
        missing.append(FIELD_TOUR_GUIDE)

    if not req.activities:
        missing.append(FIELD_ACTIVITIES)

    if not req.special_requests:
        missing.append(FIELD_SPECIAL_INSTRUCTIONS)

    if not extraction.description:
        missing.append(FIELD_DESCRIPTION)

    # Askable
    if extraction.budget is None or extraction.budget.amount is None:
        missing.append(FIELD_BUDGET)

    if extraction.duration_nights is None and extraction.duration_days is None:
        missing.append(FIELD_DURATION)

    if extraction.travel_date is None:
        missing.append(FIELD_TRAVEL_DATE)

    return missing


def generate_clarifying_questions(
    missing: list[str],
) -> list[ClarificationQuestion]:
    """Turn missing field names into questions, capped at MAX."""
    questions: list[ClarificationQuestion] = []
    for field in missing:
        template = _QUESTION_TEMPLATES.get(field)
        if template is None:
            continue
        questions.append(ClarificationQuestion(field=field, question=template))
        if len(questions) >= MAX_CLARIFYING_QUESTIONS:
            break
    return questions


def _completeness(missing: list[str]) -> float:
    """Fraction of required fields present. 0.0 - 1.0."""
    required_missing = sum(1 for f in missing if f in REQUIRED_FIELDS)
    return round(1.0 - (required_missing / len(REQUIRED_FIELDS)), 2)


# LLM interaction


_gateway: LLMGateway | None = None


def _get_gateway() -> LLMGateway:
    global _gateway
    if _gateway is None:
        _gateway = LLMGateway()
    return _gateway


def _apply_currency_hint(extraction: EnquiryExtraction, currency_hint: str | None) -> None:
    """Fill budget.currency from the customer's profile if the model
    left it null and we have a hint. In-place."""
    if currency_hint and extraction.budget is not None and extraction.budget.currency is None:
        extraction.budget.currency = currency_hint.strip().upper()


# Public entry point


def understand(
    message: str,
    *,
    currency_hint: str | None = None,
) -> EnquiryUnderstandResponse:
    """Extract structured enquiry fields from a free-text request.

    Raises AIProviderError / AITimeoutError / AIValidationError. The
    route maps these to HTTP status codes via the app's exception
    handlers.
    """
    gateway = _get_gateway()
    extraction = gateway.generate_structured_sync(
        prompt=message,
        response_model=EnquiryExtraction,
        system=ENQUIRY_SYSTEM_PROMPT,
        temperature=0.0,
    )

    _apply_currency_hint(extraction, currency_hint)

    missing = detect_missing_fields(extraction)
    questions = generate_clarifying_questions(missing)
    score = _completeness(missing)

    logger.info(
        "enquiry_understood",
        extra={
            "event": "enquiry_understood",
            "missing_fields": missing,
            "questions_count": len(questions),
            "completeness": score,
            "prompt_version": ENQUIRY_PROMPT_VERSION,
        },
    )

    return EnquiryUnderstandResponse(
        extraction=extraction,
        missing_fields=missing,
        clarifying_questions=questions,
        completeness=score,
        provider="groq",
        prompt_version=ENQUIRY_PROMPT_VERSION,
    )
