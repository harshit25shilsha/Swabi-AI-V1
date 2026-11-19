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
from datetime import date, datetime

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
    EnquiryBudget,
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


def _parse_extracted_date(value: str | None) -> date | None:
    """Parse supported model date formats into the API's date representation."""
    if not value:
        return None

    for fmt in (
        "%d/%m/%y",
        "%d/%m/%Y",
        "%Y-%m-%d",
        "%d-%m-%Y",
        "%d-%m-%y",
        "%d %B %Y",
        "%d %b %Y",
    ):
        try:
            return datetime.strptime(value.strip(), fmt).date()
        except ValueError:
            continue

    return None


def _validate_date_values(extraction: EnquiryExtraction) -> None:
    """Reject malformed dates and impossible date ordering."""
    start = _parse_extracted_date(extraction.travel_date)
    end = _parse_extracted_date(extraction.travel_date_end)

    if extraction.travel_date and start is None:
        extraction.travel_date = None
    elif start:
        extraction.travel_date = start.strftime("%d/%m/%y")

    if extraction.travel_date_end and end is None:
        extraction.travel_date_end = None
    elif end:
        extraction.travel_date_end = end.strftime("%d/%m/%y")

    if start and end and end < start:
        extraction.travel_date = None
        extraction.travel_date_end = None
        return

    # A date range is explicit trip information. Derive duration only
    # when the user/model has not already supplied a duration.
    if start and end:
        range_days = (end - start).days + 1
        range_nights = (end - start).days

        if extraction.duration_days is None:
            extraction.duration_days = range_days

        if extraction.duration_nights is None:
            extraction.duration_nights = range_nights


def _message_supports_numeric_date(message: str, value: date | None) -> bool:
    """Check date evidence using Unicode digits, independent of language.

    The model translates month names into the API's numeric date format.  This
    guard must therefore not depend on an English month-name regex: ``जनवरी``,
    ``janvier``, and ``January`` can all accompany the same numeric day/year.
    ``int`` accepts Unicode decimal digits, including Devanagari and Arabic-
    Indic numerals.
    """
    if value is None:
        return False

    numbers: set[int] = set()
    for token in re.findall(r"\d+", message):
        try:
            numbers.add(int(token))
        except ValueError:
            continue

    return value.day in numbers and (
        value.year in numbers or value.year % 100 in numbers
    )


def _build_enquiry_title(extraction: EnquiryExtraction) -> str:
    """Build a short, consistent title from extracted trip information."""
    destination = (extraction.destination or "").strip()
    trip_type = (extraction.trip_type or "").strip().casefold()

    trip_labels = {
        "honeymoon": "honeymoon",
        "romantic": "romantic",
        "family": "family",
        "adventure": "adventure",
        "business": "business",
        "friends": "friends",
    }

    descriptor = trip_labels.get(trip_type)

    if extraction.duration_days:
        duration = f"{extraction.duration_days}-Day"
    elif extraction.duration_nights:
        duration = f"{extraction.duration_nights}-Night"
    else:
        duration = None

    parts = [part for part in (duration, descriptor) if part]

    if destination:
        if parts:
            # ``str.capitalize`` lowercases every following character, which
            # makes titles such as ``6-day honeymoon`` inconsistent. Capitalize
            # only the leading duration/descriptor instead.
            title_prefix = " ".join(parts)
            return f"{title_prefix[:1].upper()}{title_prefix[1:]} trip to {destination}"
        return f"Trip to {destination}"

    if parts:
        return f"{' '.join(parts).capitalize()} trip"

    return "Travel enquiry"


# Missing field detection — deterministic


def _apply_date_guard(message: str, extraction: EnquiryExtraction) -> None:
    """Keep only dates supported by the user's message."""
    model_start = _parse_extracted_date(extraction.travel_date)
    model_end = _parse_extracted_date(extraction.travel_date_end)

    # First accept an exact date when its day and year are explicitly present
    # in any Unicode numeral system. English-only date patterns remain below
    # for relative dates and broad-period handling.
    if _message_supports_numeric_date(message, model_start):
        if model_end is not None and not _message_supports_numeric_date(message, model_end):
            extraction.travel_date_end = None
        _validate_date_values(extraction)
        return

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

    # A single start date must never produce an invented end date.
    if not has_date_range:
        extraction.travel_date_end = None

    _validate_date_values(extraction)


def _has_date_window(extraction: EnquiryExtraction) -> bool:
    note = extraction.date_flexibility.note if extraction.date_flexibility is not None else None
    return bool(
        note and (_BROAD_DATE_PATTERN.search(note) or _EXPLICIT_DATE_RANGE_PATTERN.search(note))
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


def _normalize_special_requests(value: str) -> str | None:
    """Remove malformed and equivalent request details from model output."""
    normalized: list[str] = []
    seen: set[str] = set()

    for raw_detail in value.split(";"):
        detail = re.sub(r"\s+", " ", raw_detail).strip(" ,.")
        room_and_preference = re.match(
            r"^((?:\d+|one|two|three|four|five|six|seven|eight|nine|ten) "
            r"(?:family )?rooms?),\s*(.+)$",
            detail,
            re.IGNORECASE,
        )
        details = list(room_and_preference.groups()) if room_and_preference else [detail]

        for detail in details:
            is_invalid_dropoff = re.match(
                r"^drop-off at (?:visit|explore|see|tour)\b",
                detail,
                re.IGNORECASE,
            )
            if not detail or is_invalid_dropoff:
                continue

            pickup = re.match(r"^pick(?:up)?\s+(?:at|from)\s+(.+)$", detail, re.IGNORECASE)
            sightseeing = re.match(
                r"^(?:explore|visiting|sightseeing in)\s+(.+)$",
                detail,
                re.IGNORECASE,
            )
            if pickup:
                location = pickup.group(1).strip()
                detail = f"Pickup from {location}"
                key = f"pickup:{location.casefold()}"
            elif sightseeing:
                stops = sightseeing.group(1).strip()
                detail = f"Sightseeing in {stops}"
                key = f"sightseeing:{stops.casefold()}"
            else:
                key = detail.casefold()

            if key not in seen:
                seen.add(key)
                normalized.append(detail)

    room_only = [
        detail
        for detail in normalized
        if re.fullmatch(
            r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten) "
            r"(?:family )?rooms?",
            detail,
            re.IGNORECASE,
        )
    ]
    if room_only:
        normalized = [
            detail
            for detail in normalized
            if not any(
                detail.casefold().startswith(f"{room.casefold()} in ")
                for room in room_only
            )
        ]

    return "; ".join(normalized) or None


def _preserve_extraction_details(extraction: EnquiryExtraction) -> None:
    """Add only deterministic, language-neutral derived details.

    Do not copy prose from the user's message here. The LLM translates and
    extracts free-text details for every supported input language; replaying a
    substring from the original request can reintroduce its source language.
    """
    summary = extraction.description or f"Trip to {extraction.destination or 'the destination'}"
    details: list[str] = []

    if extraction.adults is not None and extraction.children:
        ages = [str(child.age) for child in extraction.children if child.age is not None]
        composition = f"{extraction.adults} adults and {len(extraction.children)} children"
        if ages:
            composition += f" (ages {', '.join(ages)})"
        if composition.casefold() not in summary.casefold():
            details.append(composition)

    if details:
        extraction.description = f"{summary.rstrip(' .;')}; {'; '.join(details)}"


def _apply_semantic_preservation(message: str, extraction: EnquiryExtraction) -> None:
    """Preserve high-signal trip intent and experience details in existing fields."""
    normalized = message.casefold()
    req = extraction.requirements
    has_resort = re.search(r"\bresorts?\b", message, re.IGNORECASE) is not None
    has_hotel = re.search(r"\bhotels?\b", message, re.IGNORECASE) is not None
    has_houseboat = re.search(r"\bhouse\s*boats?\b", message, re.IGNORECASE) is not None

    _preserve_extraction_details(extraction)

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
        hotel_rating_match = re.search(
            r"\b([3-5])\s*[- ]?star\s+hotels?\b",
            message,
            re.IGNORECASE,
        )
        resort_rating_match = re.search(
            r"\b([3-5])\s*[- ]?star\s+resorts?\b",
            message,
            re.IGNORECASE,
        )

        hotel_label = (
            f"{hotel_rating_match.group(1)}-star Hotel"
            if hotel_rating_match
            else "Hotel"
        )

        resort_label = (
            f"{resort_rating_match.group(1)}-star Resort"
            if resort_rating_match
            else "Resort"
        )

        if not resort_rating_match and ratings and not hotel_rating_match:
            resort_label = " or ".join(f"{rating}-star" for rating in ratings) + " Resort"

        if re.search(r"\bluxury\b", normalized):
            resort_label = f"Luxury {resort_label}"

        if re.search(r"\bcomfortable\b", normalized):
            resort_label = f"comfortable {resort_label}"

        if re.search(r"\bmountain\b", normalized):
            resort_label = resort_label.replace("Resort", "Mountain Resort")
            _append_unique(extraction.tags, "mountain")

        separator = (
            " or "
            if re.search(r"\bor\b", normalized)
            else " and "
        )

        req.accommodation = f"{hotel_label}{separator}{resort_label}"

        extraction.property_type = (
            PropertyType.HOTEL_RESORT_AND_HOUSEBOAT.value
            if has_houseboat
            else PropertyType.HOTEL_AND_RESORT.value
        )

    elif has_resort:
        rating_text = " or ".join(f"{rating}-star" for rating in ratings)
        luxury = "Luxury " if re.search(r"\bluxury\b", normalized) else ""
        req.accommodation = f"{luxury}{rating_text + ' ' if rating_text else ''}Resort"

        extraction.property_type = (
            PropertyType.RESORT_AND_HOUSEBOAT.value
            if has_houseboat
            else PropertyType.RESORT.value
        )

    elif has_hotel:
        rating_text = " or ".join(f"{rating}-star" for rating in ratings)
        req.accommodation = f"{rating_text + ' ' if rating_text else ''}Hotel"
        extraction.property_type = PropertyType.HOTEL.value

    elif has_houseboat:
        if not req.accommodation or "houseboat" in req.accommodation.casefold():
            req.accommodation = "Houseboat"
        extraction.property_type = PropertyType.HOUSEBOAT.value

    for rating in ratings:
        _append_unique(extraction.tags, f"{rating}-star")



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
    elif re.search(r"\bairport\s+transfers?\b", message, re.IGNORECASE):
        req.transport = "Airport Transfers"
    elif re.search(r"\bprivate\s+transport(?:ation)?\b", message, re.IGNORECASE):
        req.transport = "Private Transportation"
    elif re.search(r"\blocal\s+transport(?:ation)?\b", message, re.IGNORECASE):
        req.transport = "Local Transportation"

    # A specific photography request subsumes the generic activity. Keep one
    # canonical entry while retaining the more useful detail.
    if any(activity.casefold() == "couple photography" for activity in req.activities):
        req.activities = [
            activity for activity in req.activities if activity.casefold() != "photography"
        ]

    unique_activities: list[str] = []
    for activity in req.activities:
        _append_unique(unique_activities, activity)
    req.activities = unique_activities

    room_match = _ROOM_COUNT_PATTERN.search(message)
    if room_match:
        count = room_match.group("count").casefold()
        extraction.rooms = int(count) if count.isdigit() else _NUMBER_WORDS[count]

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
        houseboat_context = next(
            (
                sentence
                for sentence in re.split(r"(?<=[.!?;])\s+", message)
                if re.search(r"\bhouse\s*boats?\b", sentence, re.IGNORECASE)
            ),
            "",
        )

        private_houseboat = bool(
            re.search(
                r"\bprivate\s+(?:\w+\s+){0,2}house\s*boats?\b"
                r"|\bhouse\s*boats?\b[^.;!?]{0,40}\bprivate\b",
                houseboat_context,
                re.IGNORECASE,
            )
        )

        one_night_houseboat = bool(
            re.search(
                r"\b(?:one|1)\s+nights?\b[^.;!?]{0,35}\bhouse\s*boats?\b"
                r"|\bhouse\s*boats?\b[^.;!?]{0,35}\b(?:one|1)\s+nights?\b",
                houseboat_context,
                re.IGNORECASE,
            )
        )

        if private_houseboat and one_night_houseboat:
            detail = "Private houseboat stay for one night"
        elif private_houseboat:
            detail = "Private houseboat stay"
        elif one_night_houseboat:
            detail = "Houseboat stay for one night"
        else:
            detail = ""

        if detail:
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

    req.special_requests = _normalize_special_requests(special_requests)

_MONEY_TOKEN = (
    r"(?:₹\s*)?"
    r"(?P<amount>\d+(?:,\d{3})*(?:\.\d+)?)"
    r"\s*(?P<scale>k|thousand|lakh|lac)?"
)


def _money_value(match: re.Match[str]) -> float:
    amount = float(match.group("amount").replace(",", ""))
    scale = (match.group("scale") or "").casefold()

    if scale in {"k", "thousand"}:
        amount *= 1_000
    elif scale in {"lakh", "lac"}:
        amount *= 100_000

    return amount

def _apply_budget_fallback(
    message: str,
    extraction: EnquiryExtraction,
) -> None:
    """Recover an explicit budget when the structured model omitted it.

    Never overwrite an amount already extracted by the model.
    """
    budget_match = re.search(
        r"\bbudget\b[^.;!?]{0,180}",
        message,
        re.IGNORECASE,
    )
    if budget_match is None:
        return

    context = budget_match.group()
    amounts = list(re.finditer(_MONEY_TOKEN, context, re.IGNORECASE))

    if not amounts:
        return

    currency = None

    if re.search(
        r"\bINR\b|₹|\brupees?\b|\bRs\.?\b",
        context,
        re.IGNORECASE,
    ):
        currency = "INR"
    elif re.search(r"\bGBP\b|£", context, re.IGNORECASE):
        currency = "GBP"
    elif re.search(r"\bUSD\b|\$", context, re.IGNORECASE):
        currency = "USD"
    elif re.search(r"\bEUR\b|€", context, re.IGNORECASE):
        currency = "EUR"

    is_per_person = None

    if re.search(r"\bper\s+person\b|\beach\b", context, re.IGNORECASE):
        is_per_person = True
    elif re.search(
        r"\btotal\b|\bfor\s+both\s+of\s+us\b|\bwhole\s+group\b"
        r"|\bentire\s+group\b|\bfor\s+all\s+of\s+us\b",
        context,
        re.IGNORECASE,
    ):
        is_per_person = False

    # Detect ranges without embedding _MONEY_TOKEN into another regex.
    # This avoids duplicate named-group errors if _MONEY_TOKEN defines
    # a named group such as (?P<amount>...).
    range_values = None

    for first, second in zip(amounts, amounts[1:], strict=False):
        connector = context[first.end():second.start()]

        if re.fullmatch(
            r"\s*(?:and|to|[-–—])\s*",
            connector,
            re.IGNORECASE,
        ):
            range_values = (
                _money_value(first),
                _money_value(second),
            )
            break

    if range_values is not None:
        if (
            extraction.budget_range_min is None
            or extraction.budget_range_max is None
        ):
            extraction.budget_range_min = min(range_values)
            extraction.budget_range_max = max(range_values)

        if extraction.budget is None:
            extraction.budget = EnquiryBudget(
                currency=currency,
                is_per_person=is_per_person,
            )
        else:
            if extraction.budget.currency is None:
                extraction.budget.currency = currency

            if extraction.budget.is_per_person is None:
                extraction.budget.is_per_person = is_per_person

        return

    # Preserve an amount already extracted by the LLM.
    if extraction.budget is not None and extraction.budget.amount is not None:
        return

    amount = _money_value(amounts[0])

    if extraction.budget is None:
        extraction.budget = EnquiryBudget(
            amount=amount,
            currency=currency,
            is_per_person=is_per_person,
        )
    else:
        extraction.budget.amount = amount
        extraction.budget.currency = extraction.budget.currency or currency

        if extraction.budget.is_per_person is None:
            extraction.budget.is_per_person = is_per_person


def _apply_traveler_consistency(extraction: EnquiryExtraction) -> None:
    """Keep traveller totals consistent, defaulting unspecified people to adults."""
    if extraction.adults is None and extraction.travelers is not None and not extraction.children:
        extraction.adults = extraction.travelers
    elif extraction.adults is not None:
        extraction.travelers = extraction.adults + len(extraction.children)


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

    has_budget = (
        extraction.budget is not None
        and extraction.budget.amount is not None
    ) or (
        extraction.budget_range_min is not None
        and extraction.budget_range_max is not None
    )

    if not has_budget:
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
    _apply_budget_fallback(message, extraction)
    _apply_traveler_consistency(extraction)
    _apply_currency_hint(extraction, currency_hint)

    # Always use a deterministic title instead of trusting inconsistent
    # free-form LLM title generation.
    extraction.title = _build_enquiry_title(extraction)


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
