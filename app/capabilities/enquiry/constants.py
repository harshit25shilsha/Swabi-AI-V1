"""Enums, field names, and thresholds for the enquiry capability.

Enum values are aligned with:
    * the live Enquiry form (source of truth for the extraction surface)
    * the CreateEnquiryDto in the NestJS Swagger contract
    * the Swabi BRD (CUST-ENQ-*) as a superset for future UI additions

Where the DTO and UI disagree (e.g. accommodation is single-select in
the DTO, multi-select in the UI), the extraction follows the DTO to
avoid a translation layer. See the Phase 1 report for the mapping.
"""

from __future__ import annotations

from enum import StrEnum


class Accommodation(StrEnum):
    """UI accommodation chips. DTO stores this as a single string."""

    THREE_STAR_HOTEL = "3-Star Hotel"
    FOUR_STAR_HOTEL = "4-Star Hotel"
    FIVE_STAR_HOTEL = "5-Star Hotel"
    LUXURY_RESORT = "Luxury Resort"
    HOUSEBOAT = "Houseboat"
    HOMESTAY = "Homestay"
    VILLA = "Villa"


class Transport(StrEnum):
    """Transport values; keep generic private cabs distinct from sedans."""

    PRIVATE_CAB = "Private Cab"
    PRIVATE_CAB_SEDAN = "Private Cab (Sedan)"
    PRIVATE_TEMPO_TRAVELLER = "Private Tempo Traveller"
    PRIVATE_SUV = "Private SUV"
    TEMPO_TRAVELLER_12 = "Tempo Traveller (12-Seater)"
    LUXURY_SEDAN = "Luxury Sedan"
    SHARED_COACH = "Shared Coach"


class Meal(StrEnum):
    BREAKFAST = "Breakfast"
    LUNCH = "Lunch"
    DINNER = "Dinner"
    ALL_MEALS = "All Meals"


class TourGuide(StrEnum):
    ENGLISH = "English Speaking Guide"
    LOCAL_LANGUAGE = "Local Language Guide"
    PHOTOGRAPHY = "Photography Guide"
    ADVENTURE = "Adventure Guide"
    OPTIONAL = "Optional"


# Known activities from the UI's activity picker. Custom activities
# (free strings) are returned alongside these.
KNOWN_ACTIVITIES: tuple[str, ...] = (
    "Shikara Ride",
    "Gulmarg Gondola",
    "Houseboat Stay",
    "Skiing",
    "Trekking",
    "Paragliding",
    "Local Sightseeing",
    "River Rafting",
    "Camel Safari",
    "Scuba Diving",
)


# BRD superset (fields the UI does not currently collect but the BRD defines).
# Extraction includes these when the user mentions them; they are dormant
# for the current UI and consumed when the UI exposes them.


class PropertyType(StrEnum):
    """BRD CUST-ENQ-14; composite values represent multi-stay enquiries."""

    VILLA = "villa"
    APARTMENT = "apartment"
    HOTEL = "hotel"
    HOMESTAY = "homestay"
    RESORT = "resort"
    HOUSEBOAT = "houseboat"
    RESORT_AND_HOUSEBOAT = "resort+houseboat"
    HOTEL_AND_RESORT = "hotel+resort"
    HOTEL_RESORT_AND_HOUSEBOAT = "hotel+resort+houseboat"
    GUESTHOUSE = "guesthouse"
    OTHER = "other"


class TripType(StrEnum):
    """Trip intent, including existing BRD values and common travel intents."""

    MYSELF = "myself"
    FAMILY = "family"
    HONEYMOON = "honeymoon"
    ROMANTIC = "romantic"
    ADVENTURE = "adventure"
    BUSINESS = "business"
    FRIENDS = "friends"
    OTHERS = "others"


class EnquiryFlag(StrEnum):
    """BRD CUST-ENQ-13."""

    WORK = "work"
    PETS = "pets"
    COUPLE_FRIENDLY = "couple_friendly"
    ENTIRE_APARTMENT = "entire_apartment"


# Field names for missing_fields / clarifying_questions.


FIELD_DESTINATION = "destination"
FIELD_DESCRIPTION = "description"
FIELD_WHEN = "when"  # covers travel_date OR duration
FIELD_DURATION = "duration"
FIELD_TRAVEL_DATE = "travel_date"
FIELD_TRAVELERS = "travelers"
FIELD_BUDGET = "budget"
FIELD_ACCOMMODATION = "accommodation"
FIELD_TRANSPORT = "transport"
FIELD_MEALS = "meals"
FIELD_TOUR_GUIDE = "tour_guide"
FIELD_ACTIVITIES = "activities"
FIELD_SPECIAL_INSTRUCTIONS = "special_instructions"

# Minimum information needed to understand the trip.
REQUIRED_FIELDS: tuple[str, ...] = (
    FIELD_DESTINATION,
    FIELD_WHEN,
    FIELD_TRAVELERS,
)

# Useful planning preferences, but not blockers for creating an enquiry.
RECOMMENDED_FIELDS: tuple[str, ...] = (
    FIELD_BUDGET,
    FIELD_ACCOMMODATION,
    FIELD_TRANSPORT,
    FIELD_DURATION,
    FIELD_TRAVEL_DATE,
)

# Preference details that should not be requested unless the user raises them.
OPTIONAL_FIELDS: tuple[str, ...] = (
    FIELD_MEALS,
    FIELD_TOUR_GUIDE,
    FIELD_ACTIVITIES,
    FIELD_SPECIAL_INSTRUCTIONS,
    FIELD_DESCRIPTION,
)

# Kept as the existing askable-field grouping for internal callers.
ASKABLE_FIELDS: tuple[str, ...] = RECOMMENDED_FIELDS

MAX_CLARIFYING_QUESTIONS = 4
