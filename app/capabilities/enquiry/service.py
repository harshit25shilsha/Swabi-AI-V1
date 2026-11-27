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
    FIELD_ACTIVITIES,
    FIELD_MEALS,
    FIELD_SPECIAL_INSTRUCTIONS,
    FIELD_TOUR_GUIDE,
    FIELD_TRANSPORT,
    FIELD_TRAVEL_DATE,
    FIELD_TRAVEL_DATE_END,
    FIELD_TRAVELERS,
    FIELD_WHEN,
    MAX_CLARIFYING_QUESTIONS,
    REQUIRED_FIELDS,
    UNCONFIRMED_NEGATIVE_VALUES,
    EnquiryFlag,
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
    Child,
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
    FIELD_TRAVEL_DATE: "What date would you like to start your trip?",
    FIELD_TRAVEL_DATE_END: "What date would you like to end your trip?",
    FIELD_TRAVELERS: "How many travellers?",
    FIELD_BUDGET: "Do you have a budget range in mind?",
    FIELD_ACCOMMODATION: "What type of accommodation would you prefer: budget hotel, 3-star, 4-star, luxury hotel, resort, or homestay?",
    FIELD_TRANSPORT: "What transport would you prefer?",
    FIELD_MEALS: "What meal arrangement would you prefer: breakfast only, breakfast and dinner, all meals, or no meals?",
    FIELD_TOUR_GUIDE: "Would you like a professional tour guide, a local guide, or would you prefer to explore independently?",
    FIELD_ACTIVITIES: "What sightseeing or activities would you like to include?",
    FIELD_SPECIAL_INSTRUCTIONS: "Do you have any special requirements or travel instructions, or should we proceed without any special requests?",
}

_NEGATIVE_CONFIRMATION_TEMPLATES: dict[str, str] = {
    FIELD_ACCOMMODATION: "Please confirm that you do not require accommodation for this trip.",
    FIELD_TRANSPORT: "Please confirm that you do not require transportation for this trip.",
    FIELD_MEALS: "Please confirm that you do not require any meal arrangement.",
    FIELD_TOUR_GUIDE: "Please confirm that you would prefer to travel without a tour guide.",
    FIELD_SPECIAL_INSTRUCTIONS: "Please confirm that you have no special requirements or travel instructions.",
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

    return value.day in numbers and (value.year in numbers or value.year % 100 in numbers)


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
            if not any(detail.casefold().startswith(f"{room.casefold()} in ") for room in room_only)
        ]

    return "; ".join(normalized) or None


def _preserve_extraction_details(extraction: EnquiryExtraction) -> None:
    """Add only deterministic, language-neutral derived details.

    Do not copy prose from the user's message here. The LLM translates and
    extracts free-text details for every supported input language; replaying a
    substring from the original request can reintroduce its source language.
    """
    summary = extraction.description
    details: list[str] = []

    if summary is None and extraction.destination:
        summary = f"Trip to {extraction.destination}"
    elif summary is None and extraction.duration_days:
        summary = f"{extraction.duration_days}-day trip"
    elif summary is None and extraction.travelers is not None:
        summary = f"Trip for {extraction.travelers} travellers"

    if summary is None:
        return

    if extraction.travel_date and extraction.travel_date_end:
        details.append(f"from {extraction.travel_date} to {extraction.travel_date_end}")
    elif extraction.duration_days:
        details.append(f"{extraction.duration_days}-day trip")

    if extraction.travelers is not None:
        details.append(f"{extraction.travelers} travellers")

    if extraction.adults is not None and extraction.children:
        ages = [str(child.age) for child in extraction.children if child.age is not None]
        composition = f"{extraction.adults} adults and {len(extraction.children)} children"
        if ages:
            composition += f" (ages {', '.join(ages)})"
        if composition.casefold() not in summary.casefold():
            details.append(composition)

    if details:
        for detail in details:
            if detail.casefold() not in summary.casefold():
                summary = f"{summary.rstrip(' .;')}; {detail}"
    extraction.description = summary


def _apply_semantic_preservation(message: str, extraction: EnquiryExtraction) -> None:
    """Preserve high-signal trip intent and experience details in existing fields."""
    normalized = message.casefold()
    req = extraction.requirements
    has_resort = re.search(r"\bresorts?\b", message, re.IGNORECASE) is not None
    has_hotel = re.search(r"\bhotels?\b", message, re.IGNORECASE) is not None
    has_houseboat = re.search(r"\bhouse\s*boats?\b", message, re.IGNORECASE) is not None

    _preserve_extraction_details(extraction)
    _apply_explicit_negative_preferences(message, extraction)
    _validate_tour_guide_evidence(message, extraction)
    _extract_property_preferences(message, extraction)
    _infer_city_sightseeing(message, extraction)

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

        hotel_label = f"{hotel_rating_match.group(1)}-star Hotel" if hotel_rating_match else "Hotel"

        resort_label = (
            f"{resort_rating_match.group(1)}-star Resort" if resort_rating_match else "Resort"
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

        separator = " or " if re.search(r"\bor\b", normalized) else " and "

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
            PropertyType.RESORT_AND_HOUSEBOAT.value if has_houseboat else PropertyType.RESORT.value
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
    for activity in _canonicalize_activities(req.activities):
        _append_unique(unique_activities, activity)
    req.activities = unique_activities
    _flag_activity_destination_review(extraction)

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


def _canonicalize_activities(activities: list[str]) -> list[str]:
    """Collapse equivalent generic activity labels without losing specificity."""
    canonical: list[str] = []
    for activity in activities:
        normalized = activity.strip()
        if normalized.casefold() in {"sightseeing", "local sightseeing", "city sightseeing"}:
            normalized = "Local Sightseeing"
        canonical.append(normalized)
    return canonical


def _validate_tour_guide_evidence(message: str, extraction: EnquiryExtraction) -> None:
    """Keep guide preferences only when the message supports their detail."""
    req = extraction.requirements
    if not req.tour_guide:
        return

    normalized = message.casefold()
    if not re.search(r"\b(?:tour\s+)?guide\b|\bexplore\s+independently\b", normalized):
        req.tour_guide = None
        return
    if re.search(r"\b(?:no|without|don't need|do not need)\s+(?:a\s+)?(?:tour\s+)?guide\b", normalized):
        req.tour_guide = "No"
    elif re.search(r"\benglish(?:[- ]speaking)?\s+guide\b", normalized):
        req.tour_guide = "English Speaking Guide"
    elif re.search(r"\blocal(?:[- ]language)?\s+(?:tour\s+)?guide\b", normalized):
        req.tour_guide = "Local Language Guide"
    elif re.search(r"\b(?:photography|adventure)\s+guide\b", normalized):
        # The model may preserve this supported, explicitly qualified value.
        pass
    else:
        # A requested guide is affirmative, but its language/type is unknown.
        req.tour_guide = "Yes"


def _extract_property_preferences(message: str, extraction: EnquiryExtraction) -> None:
    """Add only explicitly stated optional property preferences."""
    patterns = (
        (r"\b(?:travel(?:ling)?|trip|stay)\s+(?:for\s+)?work\b|\bwork\s+trip\b", "Travel for Work"),
        (r"\b(?:travel(?:ling)?|trip|stay)\s+with\s+(?:(?:my|a)\s+)?pets?\b|\bwith\s+(?:my\s+)?(?:dog|cat|pet)\b|\b(?:pet[- ]?friendly|bring(?:ing)?\s+(?:a\s+)?(?:dog|cat|pet))\b", "Travel with Pets"),
        (r"\b(?:an?\s+)?entire\s+(?:apartment|flat|home)\b", "Entire Apartment"),
        (r"\bcouple[- ]?friendly\b", "Couple Friendly"),
    )
    for pattern, preference in patterns:
        if re.search(pattern, message, re.IGNORECASE):
            _append_unique(extraction.property_preferences, preference)


def _flag_activity_destination_review(extraction: EnquiryExtraction) -> None:
    """Flag only clear geographic mismatches; never discard stated activities."""
    destination = (extraction.destination or "").casefold()
    if not destination or not extraction.requirements.activities:
        return

    activity_regions = {
        "Shikara Ride": ("kashmir", "srinagar"),
        "Gulmarg Gondola": ("kashmir", "gulmarg"),
        "Camel Safari": ("rajasthan", "jaisalmer", "bikaner"),
        "Scuba Diving": ("goa", "andaman", "lakshadweep", "kerala"),
        "Backwater Cruise": ("kerala", "alleppey", "alappuzha", "kumarakom"),
    }
    for activity in extraction.requirements.activities:
        regions = activity_regions.get(activity)
        if regions and not any(region in destination for region in regions):
            _append_unique(extraction.flags, EnquiryFlag.ACTIVITY_DESTINATION_REVIEW.value)
            return


def _apply_explicit_negative_preferences(message: str, extraction: EnquiryExtraction) -> None:
    """Represent an explicit refusal as a completed preference, never null.

    Null/empty means unknown in this API. These canonical values make that
    distinction survive both LLM variation and subsequent conversation merges.
    """
    normalized = message.casefold()
    req = extraction.requirements
    if re.search(r"\b(?:no|without|don't need|do not need)\s+(?:any\s+)?(?:hotel|accommodation|stay)\b", normalized):
        req.accommodation = "No accommodation required"
    if re.search(r"\b(?:no|without|don't need|do not need)\s+(?:any\s+)?(?:transport|transportation|cab|transfer)s?\b", normalized):
        req.transport = "No transport required"
    if re.search(r"\b(?:no|without|don't need|do not need)\s+(?:any\s+)?meals?\b", normalized):
        req.meals = ["No Meals Required"]
    if re.search(r"\b(?:no|without|don't need|do not need)\s+(?:a\s+)?(?:tour\s+)?guide\b", normalized):
        req.tour_guide = "No"
    if re.search(
        r"\b(?:no|none|without)\s+(?:special\s+)?(?:requests?|requirements?|instructions?)\b"
        r"|\bno\s+special\s+(?:requirements?|instructions?)\b",
        normalized,
    ):
        req.special_requests = "No special requests"


def _infer_city_sightseeing(message: str, extraction: EnquiryExtraction) -> None:
    """Turn explicitly named itinerary stops into generic sightseeing only.

    This intentionally does not look up attractions: a named stop supports
    sightseeing in that city, but not a claim that the traveller wants any
    particular monument or experience.
    """
    pattern = re.compile(
        r"\b(?:visit|visiting|explore|exploring|cover|covering)\s+"
        r"(?P<stops>[A-Z][A-Za-z .'-]*(?:\s*,\s*[A-Z][A-Za-z .'-]*)*(?:\s*(?:,?\s+and|\s+& )\s*[A-Z][A-Za-z .'-]*)?)",
        re.IGNORECASE,
    )
    for match in pattern.finditer(message):
        stops = re.split(r"\s*,\s*|\s+(?:and|&)\s+", match.group("stops"))
        for stop in stops:
            # "visit Rajasthan from 15 January" names the overall trip before
            # its dates, not a discrete sightseeing stop.
            if re.search(r"\s+from\b", stop, re.IGNORECASE):
                continue
            # Stop before common sentence continuations such as "with private".
            stop = re.split(r"\s+\b(?:with|for|from|on|in)\b\s+", stop, maxsplit=1)[0]
            stop = stop.strip(" ,.;")
            if stop and re.fullmatch(r"[A-Z][A-Za-z .'-]*", stop):
                _append_unique(extraction.requirements.activities, f"Sightseeing in {stop}")


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
        connector = context[first.end() : second.start()]

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
        if extraction.budget_range_min is None or extraction.budget_range_max is None:
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


def _apply_traveler_consistency(message: str, extraction: EnquiryExtraction) -> None:
    """Reconcile traveller fields using only explicit composition evidence.

    A total party size is not evidence that everyone is an adult. Child ages
    are retained when stated and otherwise represented as age-null children.
    """
    number_pattern = r"\d+|one|two|three|four|five|six|seven|eight|nine|ten"

    def number(value: str) -> int:
        return int(value) if value.isdigit() else _NUMBER_WORDS[value.casefold()]

    adult_match = re.search(rf"\b(?P<count>{number_pattern})\s+adults?\b", message, re.IGNORECASE)
    child_match = re.search(rf"\b(?P<count>{number_pattern})\s+children?\b", message, re.IGNORECASE)
    total_match = re.search(
        rf"\b(?:family|group|party)\s+of\s+(?P<count>{number_pattern})\b"
        rf"|\b(?P<plain>{number_pattern})\s+(?:people|travellers?|travelers?|guests?|persons?)\b",
        message,
        re.IGNORECASE,
    )

    if adult_match:
        extraction.adults = number(adult_match.group("count"))

    explicit_child_count = number(child_match.group("count")) if child_match else None
    ages_match = re.search(
        r"\bchildren?\s+(?:aged?|ages?)\s+(?P<ages>\d{1,2}(?:\s*(?:,|and)\s*\d{1,2})*)",
        message,
        re.IGNORECASE,
    )
    ages = [int(age) for age in re.findall(r"\d{1,2}", ages_match.group("ages"))] if ages_match else []
    if explicit_child_count is not None or ages:
        child_count = explicit_child_count if explicit_child_count is not None else len(ages)
        extraction.children = [Child(age=age) for age in ages[:child_count]]
        extraction.children.extend(Child(age=None) for _ in range(child_count - len(ages)))

    explicit_total = None
    if total_match:
        explicit_total = number(total_match.group("count") or total_match.group("plain"))

    composition_total = None
    if extraction.adults is not None and (explicit_child_count is not None or ages):
        composition_total = extraction.adults + len(extraction.children)

    if composition_total is not None:
        extraction.travelers = composition_total
        if explicit_total is not None and explicit_total != composition_total:
            _append_unique(extraction.flags, EnquiryFlag.TRAVELLER_COUNT_REVIEW.value)
    elif explicit_total is not None:
        extraction.travelers = explicit_total


def detect_missing_fields(extraction: EnquiryExtraction) -> list[str]:
    """Return every genuinely unanswered required field, then recommendations.

    Explicit negative preferences are retained but remain pending until the
    traveller confirms the refusal. This keeps a one-question confirmation
    loop without discarding what the traveller already said.
    """
    missing: list[str] = []
    req = extraction.requirements

    if not extraction.destination:
        missing.append(FIELD_DESTINATION)

    if not extraction.travel_date:
        missing.append(FIELD_TRAVEL_DATE)

    if not extraction.travel_date_end:
        missing.append(FIELD_TRAVEL_DATE_END)

    if _field_needs_confirmation(FIELD_ACCOMMODATION, req.accommodation):
        missing.append(FIELD_ACCOMMODATION)

    if _field_needs_confirmation(FIELD_TRANSPORT, req.transport):
        missing.append(FIELD_TRANSPORT)

    if _field_needs_confirmation(FIELD_MEALS, req.meals):
        missing.append(FIELD_MEALS)

    if _field_needs_confirmation(FIELD_TOUR_GUIDE, req.tour_guide):
        missing.append(FIELD_TOUR_GUIDE)

    if not req.activities:
        missing.append(FIELD_ACTIVITIES)

    if _field_needs_confirmation(FIELD_SPECIAL_INSTRUCTIONS, req.special_requests):
        missing.append(FIELD_SPECIAL_INSTRUCTIONS)

    return missing


def _field_needs_confirmation(field: str, value: object) -> bool:
    """Return true for absent values and explicit refusals awaiting confirmation."""
    if value is None or value == "" or value == []:
        return True
    values = UNCONFIRMED_NEGATIVE_VALUES.get(field, ())
    if isinstance(value, str):
        return value.casefold() in values
    if isinstance(value, list):
        return any(isinstance(item, str) and item.casefold() in values for item in value)
    return False


def generate_clarifying_questions(
    missing: list[str],
    extraction: EnquiryExtraction | None = None,
) -> list[ClarificationQuestion]:
    """Generate exactly one question for every missing required field."""
    missing_set = set(missing)
    required_missing = [field for field in REQUIRED_FIELDS if field in missing_set]
    fields_to_ask = required_missing

    questions: list[ClarificationQuestion] = []
    for field in fields_to_ask:
        value = _required_field_value(extraction, field) if extraction else None
        template = (
            _NEGATIVE_CONFIRMATION_TEMPLATES.get(field)
            if extraction is not None and not _field_is_absent(value)
            else _QUESTION_TEMPLATES.get(field)
        )
        if template is None:
            continue
        questions.append(ClarificationQuestion(field=field, question=template))
        if len(questions) >= MAX_CLARIFYING_QUESTIONS:
            break
    return questions


def _field_is_absent(value: object) -> bool:
    return value is None or value == "" or value == []


def _required_field_value(extraction: EnquiryExtraction | None, field: str) -> object:
    if extraction is None:
        return None
    requirements = extraction.requirements
    values: dict[str, object] = {
        FIELD_ACCOMMODATION: requirements.accommodation,
        FIELD_TRANSPORT: requirements.transport,
        FIELD_MEALS: requirements.meals,
        FIELD_TOUR_GUIDE: requirements.tour_guide,
        FIELD_SPECIAL_INSTRUCTIONS: requirements.special_requests,
    }
    return values.get(field)


def _completeness(missing: list[str]) -> float:
    """Fraction of the nine required fields that are complete."""
    required_missing = sum(1 for f in missing if f in REQUIRED_FIELDS)
    return round(1.0 - (required_missing / len(REQUIRED_FIELDS)), 4)


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


def _merge_with_previous(
    previous: EnquiryExtraction,
    update: EnquiryExtraction,
) -> EnquiryExtraction:
    """Overlay a follow-up extraction onto the prior enquiry without loss.

    Scalars from a follow-up win only when they contain an answer; list fields
    are combined case-insensitively so a traveller can add preferences over
    several turns.  This is deterministic and deliberately independent of
    wording returned by the LLM.
    """
    merged = previous.model_copy(deep=True)
    for field in (
        "title", "description", "destination", "travel_date", "travel_date_end",
        "date_flexibility", "duration_days", "duration_nights", "travelers", "adults",
        "budget", "budget_range_min", "budget_range_max", "rooms", "property_type",
        "trip_type",
    ):
        value = getattr(update, field)
        if value is not None:
            setattr(merged, field, value)

    for field in ("children", "property_preferences", "flags", "tags"):
        existing = getattr(merged, field)
        for item in getattr(update, field):
            if item not in existing:
                existing.append(item)

    for field in (
        "accommodation", "transport", "tour_guide", "special_requests",
    ):
        value = getattr(update.requirements, field)
        if value:
            setattr(merged.requirements, field, value)
    for field in ("meals", "activities"):
        existing = getattr(merged.requirements, field)
        for item in getattr(update.requirements, field):
            _append_unique(existing, item)
    return merged


# Public entry point


def understand(
    message: str,
    *,
    currency_hint: str | None = None,
    previous_extraction: EnquiryExtraction | None = None,
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
    _apply_budget_fallback(message, extraction)
    _apply_traveler_consistency(message, extraction)
    _apply_semantic_preservation(message, extraction)
    _apply_currency_hint(extraction, currency_hint)
    if previous_extraction is not None:
        extraction = _merge_with_previous(previous_extraction, extraction)
        _apply_traveler_consistency(message, extraction)
        _preserve_extraction_details(extraction)

    # Always use a deterministic title instead of trusting inconsistent
    # free-form LLM title generation.
    extraction.title = _build_enquiry_title(extraction)

    missing = detect_missing_fields(extraction)
    questions = generate_clarifying_questions(missing, extraction)
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
