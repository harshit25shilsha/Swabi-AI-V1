"""Enquiry Understanding orchestrator.

Steps:
    1. Call the LLM via LLMGateway.generate_structured_sync.
    2. Validate against EnquiryExtraction.
    3. Reject unsupported exact dates and apply the currency hint if needed.
    4. Deterministically compute missing required and recommended fields.
    5. Ask about missing required fields first; defer recommendations until
       the core trip details are known.
    6. Assemble the response with a completeness score.

Synchronous, matching Chat Safety.
"""

from __future__ import annotations

import logging
import re
from datetime import date

from app.capabilities.enquiry.constants import (
    FIELD_ACCOMMODATION,
    FIELD_BUDGET,
    FIELD_DESTINATION,
    FIELD_DURATION,
    FIELD_TRANSPORT,
    FIELD_TRAVEL_DATE,
    FIELD_TRAVELERS,
    FIELD_WHEN,
    MAX_CLARIFYING_QUESTIONS,
    RECOMMENDED_FIELDS,
    REQUIRED_FIELDS,
    PropertyType,
    Transport,
    TripType,
)
from app.capabilities.enquiry.prompts import (
    ENQUIRY_PROMPT_VERSION,
    ENQUIRY_SYSTEM_PROMPT,
)
from app.capabilities.enquiry.schemas import (
    ClarificationQuestion,
    DateFlexibility,
    EnquiryExtraction,
    EnquiryUnderstandResponse,
)
from app.llm.gateway import LLMGateway

logger = logging.getLogger(__name__)

_MONTH = (
    r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?"
)
_SHARED_MONTH_DATE_RANGE = (
    rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s*(?:to|through|until|[-–])\s*"
    rf"\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:{_MONTH})\b"
)
_SHARED_MONTH_DATE_RANGE_PATTERN = re.compile(_SHARED_MONTH_DATE_RANGE, re.IGNORECASE)
_EXPLICIT_DATE_PATTERN = re.compile(
    rf"\b(?:"
    rf"\d{{1,2}}/\d{{1,2}}(?:/\d{{2,4}})?|"
    rf"\d{{1,2}}-\d{{1,2}}-\d{{2,4}}|"
    rf"{_SHARED_MONTH_DATE_RANGE}|"
    rf"(?:{_MONTH})\s+\d{{1,2}}(?:st|nd|rd|th)?|"
    rf"\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:{_MONTH})|"
    rf"(?:the\s+)?(?:day\s+after\s+tomorrow|tomorrow|today)|"
    rf"(?:in|after)\s+(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
    rf"(?:days?|weeks?)|"
    rf"(?:(?:next|this|coming|upcoming|on)\s+)"
    rf"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)|"
    rf"(?:this|next|coming|upcoming)\s+weekend|"
    rf"(?:first|second|third|fourth|last)\s+"
    rf"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\s+of\s+(?:{_MONTH})"
    rf")\b",
    re.IGNORECASE,
)
_EXPLICIT_DATE_RANGE_PATTERN = re.compile(
    rf"\b(?:from\s+)?(?:"
    rf"(?:{_MONTH})\s+\d{{1,2}}(?:st|nd|rd|th)?|"
    rf"\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:{_MONTH})|"
    rf"\d{{1,2}}/\d{{1,2}}(?:/\d{{2,4}})?|"
    rf"\d{{1,2}}-\d{{1,2}}-\d{{2,4}}"
    rf")\s*(?:to|through|until|[-–])\s*"
    rf"(?:(?:{_MONTH})\s*)?\d{{1,2}}(?:st|nd|rd|th)?\b",
    re.IGNORECASE,
)
_WEEKEND_PATTERN = re.compile(
    r"\b(?:this|next|coming|upcoming)\s+weekend\b",
    re.IGNORECASE,
)
_BROAD_DATE_PATTERN = re.compile(
    rf"\b(?:"
    rf"(?:early|mid|late|beginning|start|end)\s+(?:part\s+of\s+)?(?:{_MONTH})|"
    rf"(?:first|second|third|fourth|last)\s+week\s+(?:of|in)\s+(?:{_MONTH})|"
    rf"(?:first|last)\s+weekend\s+(?:of|in)\s+(?:{_MONTH})|"
    rf"(?:(?:first|last)\s+week\s+of\s+)?(?:{_MONTH})|"
    rf"(?:this|next|coming|upcoming)?\s*weekend|"
    rf"(?:Christmas|New\s+Year(?:'s)?|Easter|holiday\s+season|"
    rf"summer|winter|spring|autumn|fall|monsoon)|"
    rf"(?:in|after)\s+\d+\s+(?:days?|weeks?|months?)"
    rf")\b",
    re.IGNORECASE,
)
_STAR_RATING_PATTERN = re.compile(r"\b([3-5])\s*[- ]?\s*star\b", re.IGNORECASE)
_FLEX_WINDOW_PATTERN = re.compile(
    r"\bflexible\b[^.;]{0,80}?(?:\b(?:by|within|plus\s+or\s+minus)\s*|(?:\+/-|±)\s*)"
    r"(\d+)(?:\s*(?:-|to)\s*(\d+))?\s*days?\b",
    re.IGNORECASE,
)
_ROOM_COUNT_PATTERN = re.compile(
    r"\b(?P<count>\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
    r"(?:(?P<family>family)\s+)?rooms?\b",
    re.IGNORECASE,
)
_NUMBER_WORDS = {
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
}


_QUESTION_TEMPLATES: dict[str, str] = {
    FIELD_DESTINATION: "Where would you like to go?",
    FIELD_WHEN: "When are you planning to travel, and for how long?",
    FIELD_DURATION: "How many nights and days?",
    FIELD_TRAVEL_DATE: "What date would you like to start?",
    FIELD_TRAVELERS: "How many travellers?",
    FIELD_BUDGET: "Do you have a budget range in mind?",
    FIELD_ACCOMMODATION: "Any accommodation preference, such as a hotel, resort, or houseboat?",
    FIELD_TRANSPORT: "What transport would you prefer?",
}


# Missing field detection — deterministic


def _apply_date_guard(message: str, extraction: EnquiryExtraction) -> None:
    """Reject exact dates unsupported by the user's wording.

    Relative calendar expressions are left to the prompt's reference-date
    instructions; broad periods are retained as date-flexibility notes.
    """
    date_expressions = list(_EXPLICIT_DATE_PATTERN.finditer(message))
    weekend = _WEEKEND_PATTERN.search(message) is not None
    has_date_range = (
        len(date_expressions) > 1
        or weekend
        or _SHARED_MONTH_DATE_RANGE_PATTERN.search(message) is not None
        or _EXPLICIT_DATE_RANGE_PATTERN.search(message) is not None
    )

    if not date_expressions:
        extraction.travel_date = None
        extraction.travel_date_end = None
        broad_period = _BROAD_DATE_PATTERN.search(message)
        if broad_period is None:
            if extraction.date_flexibility is not None:
                extraction.date_flexibility.note = None
            return

        flexibility = extraction.date_flexibility
        if flexibility is None:
            extraction.date_flexibility = DateFlexibility(
                is_flexible=True,
                note=broad_period.group().strip(),
            )
        else:
            flexibility.is_flexible = True
            flexibility.window_days = None
            flexibility.note = broad_period.group().strip()
        return

    if not has_date_range:
        extraction.travel_date_end = None


def _has_date_window(extraction: EnquiryExtraction) -> bool:
    note = (
        extraction.date_flexibility.note
        if extraction.date_flexibility is not None
        else None
    )
    return bool(
        note
        and (
            _BROAD_DATE_PATTERN.search(note)
            or _EXPLICIT_DATE_RANGE_PATTERN.search(note)
        )
    )


def _apply_flexibility_window(message: str, extraction: EnquiryExtraction) -> None:
    match = _FLEX_WINDOW_PATTERN.search(message)
    if match is None:
        return

    flexibility = extraction.date_flexibility or DateFlexibility()
    flexibility.is_flexible = True
    flexibility.window_days = int(match.group(2) or match.group(1))
    phrase = match.group().strip()
    if flexibility.note and phrase.casefold() not in flexibility.note.casefold():
        flexibility.note = f"{flexibility.note}; {phrase}"
    elif not flexibility.note:
        flexibility.note = phrase
    extraction.date_flexibility = flexibility


def _append_unique(values: list[str], value: str) -> None:
    if value.casefold() not in {item.casefold() for item in values}:
        values.append(value)


def _append_special_request(current: str, detail: str) -> str:
    if detail.casefold() in current.casefold():
        return current
    return f"{current}; {detail}".strip("; ")


def _preserve_named_stops(message: str, extraction: EnquiryExtraction) -> None:
    match = re.search(
        r"\b(?:visit|explore|see|tour)\s+(.+?)"
        r"(?=[.;!?]|,\s*(?:and\s+)?(?:activities|experiences|we|please|enjoy)\b|"
        r"\s+(?:and\s+)?(?:enjoy|activities|experiences|we|please|need|prefer)\b|$)",
        message,
        re.IGNORECASE,
    )
    summary = extraction.description or f"Trip to {extraction.destination or 'the destination'}"
    details: list[str] = []
    if match:
        stops = match.group(1).strip(" ,")
        if stops and stops.casefold() not in summary.casefold():
            details.append(f"visiting {stops}")

    if extraction.adults is not None and extraction.children:
        ages = [str(child.age) for child in extraction.children if child.age is not None]
        composition = (
            f"{extraction.adults} adults and {len(extraction.children)} children"
        )
        if ages:
            composition += f" (ages {', '.join(ages)})"
        if composition.casefold() not in summary.casefold():
            details.append(composition)

    if details:
        extraction.description = f"{summary.rstrip(' .;')}; {'; '.join(details)}"


def _transport_location_details(message: str) -> list[str]:
    details: list[str] = []
    for sentence in re.split(r"(?<=[.!?;])\s+", message):
        if not re.search(
            r"\b(?:cab|taxi|tempo\s+traveller|transport(?:ation)?|pickup|pick-up|vehicle)\b",
            sentence,
            re.IGNORECASE,
        ):
            continue
        pickup = re.search(
            r"\bfrom\s+(.+?)(?=\s+(?:for|during|throughout|and|to)\b|[,.;!?]|$)",
            sentence,
            re.IGNORECASE,
        )
        dropoff = re.search(
            r"\bto\s+(.+?)(?=\s+(?:for|during|throughout|and|from)\b|[,.;!?]|$)",
            sentence,
            re.IGNORECASE,
        )
        if pickup:
            details.append(f"Pickup from {pickup.group(1).strip()}")
        if dropoff:
            details.append(f"Drop-off at {dropoff.group(1).strip()}")
    return details


def _apply_semantic_preservation(message: str, extraction: EnquiryExtraction) -> None:
    """Preserve high-signal trip intent and experience details in existing fields."""
    normalized = message.casefold()
    req = extraction.requirements
    has_resort = re.search(r"\bresorts?\b", message, re.IGNORECASE) is not None
    has_hotel = re.search(r"\bhotels?\b", message, re.IGNORECASE) is not None
    has_houseboat = re.search(r"\bhouse\s*boats?\b", message, re.IGNORECASE) is not None

    _preserve_named_stops(message, extraction)

    if re.search(r"\bhoneymoon\b", normalized):
        extraction.trip_type = TripType.HONEYMOON.value
    elif re.search(r"\b(?:family|families)\s+(?:trip|holiday|vacation)\b", normalized):
        extraction.trip_type = TripType.FAMILY.value
        _append_unique(extraction.tags, "family")
    elif re.search(r"\b(?:business|work)\s+(?:trip|travel)\b", normalized):
        extraction.trip_type = TripType.BUSINESS.value
        _append_unique(extraction.tags, "business")
    elif re.search(r"\badventure\s+(?:trip|holiday|vacation)\b", normalized):
        extraction.trip_type = TripType.ADVENTURE.value
        _append_unique(extraction.tags, "adventure")
    elif re.search(r"\b(?:friends|group of friends)\s+(?:trip|holiday|vacation)\b", normalized):
        extraction.trip_type = TripType.FRIENDS.value
        _append_unique(extraction.tags, "friends")
    elif re.search(r"\bromantic\b", normalized):
        extraction.trip_type = TripType.ROMANTIC.value

    for descriptor in ("honeymoon", "romantic", "luxury", "peaceful"):
        if re.search(rf"\b{descriptor}\b", normalized):
            _append_unique(extraction.tags, descriptor)

    ratings = list(dict.fromkeys(_STAR_RATING_PATTERN.findall(message)))
    if has_hotel and has_resort:
        hotel_rating = f"{ratings[0]}-star " if ratings else ""
        comfort = "comfortable " if re.search(r"\bcomfortable\b", normalized) else ""
        mountain = "Mountain " if re.search(r"\bmountain\b", normalized) else ""
        separator = " or " if re.search(r"\bor\b", normalized) else " and "
        req.accommodation = (
            f"{hotel_rating}Hotel{separator}{comfort}{mountain}Resort"
        )
        extraction.property_type = (
            PropertyType.HOTEL_RESORT_AND_HOUSEBOAT.value
            if has_houseboat
            else PropertyType.HOTEL_AND_RESORT.value
        )
        if re.search(r"\bmountain\b", normalized):
            _append_unique(extraction.tags, "mountain")
    elif has_resort:
        rating_text = " or ".join(f"{rating}-star" for rating in ratings)
        luxury = "Luxury " if re.search(r"\bluxury\b", normalized) else ""
        rating = f"{rating_text} " if rating_text else ""
        req.accommodation = f"{luxury}{rating}Resort"
        extraction.property_type = (
            PropertyType.RESORT_AND_HOUSEBOAT.value
            if has_houseboat
            else PropertyType.RESORT.value
        )
        for rating_value in ratings:
            _append_unique(extraction.tags, f"{rating_value}-star")
    elif has_hotel:
        rating_text = " or ".join(f"{rating}-star" for rating in ratings)
        req.accommodation = f"{rating_text + ' ' if rating_text else ''}Hotel"
        extraction.property_type = "hotel"
        for rating_value in ratings:
            _append_unique(extraction.tags, f"{rating_value}-star")
    elif has_houseboat:
        if not req.accommodation or "houseboat" in req.accommodation.casefold():
            req.accommodation = "Houseboat"
        extraction.property_type = PropertyType.HOUSEBOAT.value

    if has_houseboat:
        _append_unique(req.activities, "Houseboat Stay")

    requested_activities = (
        (r"\bcandle[\s-]?light dinner\b", "Candlelight Dinner"),
        (r"\bscenic views?\b", "Scenic Views"),
    )
    for pattern, activity in requested_activities:
        if re.search(pattern, message, re.IGNORECASE):
            _append_unique(req.activities, activity)

    if re.search(r"\bphotography\s+spots?\b", message, re.IGNORECASE):
        req.activities = [
            activity for activity in req.activities if activity.casefold() != "photography"
        ]
        _append_unique(req.activities, "Photography Spots")
    elif re.search(r"\bphotograph(?:y|ic)\b", message, re.IGNORECASE):
        _append_unique(req.activities, "Photography")

    if re.search(r"\bsightseeing\b", message, re.IGNORECASE):
        _append_unique(req.activities, "Local Sightseeing")

    if re.search(r"\bprivate\s+(?:cab|taxi)\b", message, re.IGNORECASE) and not re.search(
        r"\b(?:sedan|suv|van|tempo|coach)\b",
        message,
        re.IGNORECASE,
    ):
        req.transport = Transport.PRIVATE_CAB.value
    elif re.search(r"\bprivate\s+tempo\s+traveller\b", message, re.IGNORECASE) and not re.search(
        r"\b\d+\s*[- ]?\s*seater\b",
        message,
        re.IGNORECASE,
    ):
        req.transport = Transport.PRIVATE_TEMPO_TRAVELLER.value

    room_match = _ROOM_COUNT_PATTERN.search(message)
    if room_match:
        count = room_match.group("count").casefold()
        extraction.rooms = int(count) if count.isdigit() else _NUMBER_WORDS[count]
        room_detail = re.match(
            r"[^.;!?]+",
            message[room_match.start() :],
        )
        if room_detail:
            special_detail = room_detail.group().strip(" ,")
        else:
            special_detail = ""
    else:
        special_detail = ""

    if extraction.duration_days is not None and extraction.duration_days > 0:
        if extraction.duration_nights is None:
            extraction.duration_nights = extraction.duration_days - 1

    special_requests = req.special_requests or ""
    for activity_pattern, _ in requested_activities:
        special_requests = re.sub(activity_pattern, "", special_requests, flags=re.IGNORECASE)
    special_requests = re.sub(
        r"\bphotography\s+spots?\b|\bphotograph(?:y|ic)\b",
        "",
        special_requests,
        flags=re.IGNORECASE,
    )
    special_requests = re.sub(r"\s+", " ", special_requests)
    special_requests = re.sub(r"[,;]\s*[,;]+", ", ", special_requests)
    special_requests = re.sub(
        r"^(?:\s*(?:and|or)\s+)+|(?:\s+(?:and|or))\s*(?=[,;]|$)",
        "",
        special_requests,
        flags=re.IGNORECASE,
    )
    special_requests = special_requests.strip(" ,;.")

    if has_houseboat:
        if re.search(r"\bprivate\b", normalized) and re.search(
            r"\b(?:one|1)\s+nights?\b", normalized
        ):
            detail = "Private houseboat stay for one night"
        elif re.search(r"\bprivate\b", normalized):
            detail = "Private houseboat stay"
        elif re.search(r"\b(?:one|1)\s+nights?\b", normalized):
            detail = "Houseboat stay for one night"
        else:
            detail = ""
        if detail and detail.casefold() not in special_requests.casefold():
            special_requests = _append_special_request(special_requests, detail)

    if special_detail:
        special_requests = _append_special_request(special_requests, special_detail)

    for detail in _transport_location_details(message):
        special_requests = _append_special_request(special_requests, detail)

    preference_patterns = (
        r"\bpeaceful(?:\s+locations?)?\b",
        r"\bnot\s+too\s+crowded\b",
        r"\bless\s+crowded\b",
        r"\bavoid(?:ing)?\s+crowds?\b",
    )
    for pattern in preference_patterns:
        match = re.search(pattern, message, re.IGNORECASE)
        if match and not re.search(pattern, special_requests, re.IGNORECASE):
            special_requests = f"{special_requests}; {match.group().strip()}".strip("; ")

    req.special_requests = special_requests or None


def detect_missing_fields(extraction: EnquiryExtraction) -> list[str]:
    """Return field names that are absent or unusable.

    Return missing required fields first, followed by recommended planning
    details. Optional preferences are not treated as missing.
    """
    missing: list[str] = []
    req = extraction.requirements

    if not extraction.destination:
        missing.append(FIELD_DESTINATION)

    if (
        extraction.travel_date is None
        and extraction.travel_date_end is None
        and extraction.duration_days is None
        and extraction.duration_nights is None
        and not _has_date_window(extraction)
    ):
        missing.append(FIELD_WHEN)

    if extraction.travelers is None:
        missing.append(FIELD_TRAVELERS)

    if extraction.budget is None or extraction.budget.amount is None:
        missing.append(FIELD_BUDGET)

    if not req.accommodation:
        missing.append(FIELD_ACCOMMODATION)

    if not req.transport:
        missing.append(FIELD_TRANSPORT)

    if extraction.duration_nights is None and extraction.duration_days is None:
        missing.append(FIELD_DURATION)

    has_date_window = _has_date_window(extraction)
    if (
        extraction.travel_date is None
        and extraction.travel_date_end is None
        and not has_date_window
        and (extraction.duration_days is not None or extraction.duration_nights is not None)
    ):
        missing.append(FIELD_TRAVEL_DATE)

    return missing


def generate_clarifying_questions(
    missing: list[str],
) -> list[ClarificationQuestion]:
    """Ask core questions first, then recommended questions, capped at MAX."""
    missing_set = set(missing)
    required_missing = [field for field in REQUIRED_FIELDS if field in missing_set]
    fields_to_ask = required_missing or [
        field for field in RECOMMENDED_FIELDS if field in missing_set
    ]

    questions: list[ClarificationQuestion] = []
    for field in fields_to_ask:
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
        system=ENQUIRY_SYSTEM_PROMPT.replace(
            "{reference_date}",
            date.today().strftime("%d/%m/%Y"),
        ),
        temperature=0.0,
    )

    _apply_date_guard(message, extraction)
    _apply_flexibility_window(message, extraction)
    _apply_semantic_preservation(message, extraction)
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
