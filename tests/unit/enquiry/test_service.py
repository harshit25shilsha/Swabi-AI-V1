"""Tests for the enquiry service with a mocked LLM gateway."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.capabilities.enquiry import service as svc
from app.capabilities.enquiry.constants import (
    FIELD_ACCOMMODATION,
    FIELD_DESTINATION,
)
from app.capabilities.enquiry.prompts import ENQUIRY_PROMPT_VERSION
from app.capabilities.enquiry.schemas import (
    DateFlexibility,
    EnquiryBudget,
    EnquiryExtraction,
    EnquiryRequirements,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _full_extraction() -> EnquiryExtraction:
    return EnquiryExtraction(
        title="5-day family trip to Manali",
        description="5-day family trip",
        destination="Manali",
        travel_date="15/12/26",
        duration_days=5,
        travelers=4,
        budget=EnquiryBudget(amount=60000, currency="INR"),
        requirements=EnquiryRequirements(
            accommodation="5-Star Hotel",
            meals=["All Meals"],
            transport="Private SUV",
            activities=["Skiing"],
            tour_guide="English Speaking Guide",
            special_requests="Senior citizen friendly",
        ),
    )


@pytest.fixture
def mock_gateway():
    gateway = MagicMock()
    gateway.generate_structured_sync.return_value = _full_extraction()
    return gateway


@pytest.fixture
def patch_gateway(mock_gateway):
    """Patch the service-level gateway getter for the duration of a test."""
    with patch(
        "app.capabilities.enquiry.service._get_gateway",
        return_value=mock_gateway,
    ) as p:
        yield p


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_understand_returns_response(patch_gateway, mock_gateway):
    r = svc.understand("5-day family trip to Manali on 15/12/26")
    assert r.extraction.destination == "Manali"
    assert r.extraction.duration_days == 5
    assert r.completeness == 1.0
    assert r.missing_fields == []


def test_understand_passes_system_prompt(patch_gateway, mock_gateway):
    svc.understand("test message")
    kwargs = mock_gateway.generate_structured_sync.call_args.kwargs
    assert "system" in kwargs
    assert "structured-data extractor" in kwargs["system"]
    assert "{reference_date}" not in kwargs["system"]


def test_understand_passes_temperature_zero(patch_gateway, mock_gateway):
    svc.understand("test message")
    kwargs = mock_gateway.generate_structured_sync.call_args.kwargs
    assert kwargs.get("temperature") == 0.0


def test_understand_sets_prompt_version(patch_gateway):
    r = svc.understand("test message")
    assert r.prompt_version == ENQUIRY_PROMPT_VERSION
    assert r.prompt_version.startswith("enquiry_")


def test_understand_sets_provider(patch_gateway):
    r = svc.understand("test message")
    assert r.provider == "groq"


# ---------------------------------------------------------------------------
# Date assumptions
# ---------------------------------------------------------------------------


def test_broad_date_period_clears_unsupported_exact_dates(patch_gateway, mock_gateway):
    extraction = _full_extraction()
    extraction.travel_date = "24/12/26"
    extraction.travel_date_end = "29/12/26"
    extraction.date_flexibility = DateFlexibility(
        is_flexible=False,
        window_days=7,
        note="24/12/26",
    )
    mock_gateway.generate_structured_sync.return_value = extraction

    r = svc.understand("6-day honeymoon trip in the last week of December")

    assert r.extraction.travel_date is None
    assert r.extraction.travel_date_end is None
    assert r.extraction.date_flexibility is not None
    assert r.extraction.date_flexibility.is_flexible is True
    assert r.extraction.date_flexibility.window_days is None
    assert r.extraction.date_flexibility.note == "last week of December"


def test_unsupported_date_flexibility_note_is_not_trusted(patch_gateway, mock_gateway):
    extraction = EnquiryExtraction(
        date_flexibility=DateFlexibility(
            is_flexible=True,
            note="late December",
        )
    )
    mock_gateway.generate_structured_sync.return_value = extraction

    r = svc.understand("I would like to plan a trip")

    assert r.extraction.date_flexibility is not None
    assert r.extraction.date_flexibility.note is None
    assert "when" in r.missing_fields


def test_honeymoon_example_does_not_ask_for_unmentioned_preferences(
    patch_gateway,
    mock_gateway,
):
    extraction = EnquiryExtraction(
        destination="Kerala",
        travel_date="24/12/26",
        travel_date_end="29/12/26",
        duration_days=6,
        travelers=2,
        budget=EnquiryBudget(amount=85000, currency="INR", is_per_person=False),
        requirements=EnquiryRequirements(
            accommodation="Luxury Resort",
            meals=["Breakfast", "Dinner"],
            transport="Private Cab (Sedan)",
            activities=[
                "Houseboat Stay",
                "Local Sightseeing",
                "Photography",
            ],
            special_requests=(
                "candlelight dinner, scenic views, peaceful locations, not too crowded"
            ),
        ),
        property_type="resort",
        tags=["romantic", "honeymoon", "luxury", "peaceful"],
    )
    mock_gateway.generate_structured_sync.return_value = extraction

    message = (
        "We are planning a romantic 6-day honeymoon trip to Kerala in the last week "
        "of December. We are 2 adults and our budget is approximately 85,000 INR for "
        "both of us. We prefer luxury 4-star or 5-star resorts with breakfast and "
        "dinner included. We want to explore Munnar, Alleppey backwaters and enjoy "
        "a private houseboat stay for one night. Please arrange private cab "
        "transportation from Kochi airport. We would also love candlelight dinner, "
        "scenic views and some photography spots. Prefer peaceful locations, not "
        "too crowded."
    )
    r = svc.understand(message)

    assert r.extraction.travel_date is None
    assert r.extraction.travel_date_end is None
    assert r.extraction.date_flexibility is not None
    assert r.extraction.trip_type == "honeymoon"
    assert r.extraction.requirements.accommodation == "Luxury 4-star or 5-star Resort"
    assert r.extraction.property_type == "resort+houseboat"
    assert r.extraction.duration_nights == 5
    assert r.extraction.requirements.transport == "Private Cab"
    assert {
        "Houseboat Stay",
        "Local Sightseeing",
        "Candlelight Dinner",
        "Scenic Views",
        "Photography Spots",
    }.issubset(r.extraction.requirements.activities)
    special_requests = r.extraction.requirements.special_requests
    assert special_requests is not None
    assert "peaceful locations" in special_requests
    assert "not too crowded" in special_requests
    assert "private houseboat stay for one night" in special_requests.casefold()
    assert "candlelight dinner" not in special_requests.casefold()
    assert "scenic views" not in special_requests.casefold()
    assert {"4-star", "5-star"}.issubset(r.extraction.tags)
    assert r.missing_fields == []
    assert r.clarifying_questions == []
    assert r.completeness == 1.0
    expected_extraction = r.extraction.model_dump()
    repeated = svc.understand(message)
    assert repeated.extraction.model_dump() == expected_extraction


def test_family_example_preserves_dates_destinations_transport_and_room_type(
    patch_gateway,
    mock_gateway,
):
    extraction = EnquiryExtraction(
        title="7-day family vacation to Himachal Pradesh",
        description="7-day family vacation to Himachal Pradesh for a family of 6",
        destination="Himachal Pradesh",
        date_flexibility=DateFlexibility(is_flexible=True, note="June"),
        duration_days=7,
        duration_nights=6,
        travelers=6,
        adults=4,
        children=[{"age": 5}, {"age": 10}],
        budget=EnquiryBudget(amount=120000, currency="INR", is_per_person=False),
        requirements=EnquiryRequirements(
            accommodation="4-star Resort",
            meals=["Breakfast"],
            transport="Tempo Traveller (12-Seater)",
            activities=[
                "Paragliding",
                "River Rafting",
                "Snow Activities",
                "Local Sightseeing",
            ],
            special_requests=(
                "Ensure comfortable travel for senior citizen and avoid excessive "
                "walking. Prefer rooms with mountain views"
            ),
        ),
        rooms=2,
        property_type="resort",
        trip_type="family",
        tags=["family", "7-day", "4-star", "mountain"],
    )
    mock_gateway.generate_structured_sync.return_value = extraction
    message = (
        "We are planning a 7-day family vacation to Himachal Pradesh sometime in "
        "the second week of June. We are 4 adults and 2 children aged 5 and 10. "
        "Our total budget is around 1.2 lakh INR. We prefer 4-star hotels or "
        "comfortable mountain resorts with breakfast included. We would like to "
        "visit Shimla, Manali and Solang Valley. Activities should include "
        "paragliding, river rafting, snow activities if available, and local "
        "sightseeing. We need a private tempo traveller from Chandigarh for the "
        "entire trip. We prefer two family rooms with mountain views. One "
        "traveller is a senior citizen, so please ensure comfortable travel and "
        "avoid excessive walking. We are flexible with travel dates by 2-3 days."
    )

    r = svc.understand(message)
    date_flexibility = r.extraction.date_flexibility
    assert date_flexibility is not None
    assert date_flexibility.note is not None
    assert date_flexibility.window_days == 3
    assert "second week of June" in date_flexibility.note
    assert "2-3 days" in date_flexibility.note
    assert r.extraction.travel_date is None
    assert r.extraction.description is not None
    assert "Shimla, Manali and Solang Valley" in r.extraction.description
    assert "4 adults and 2 children (ages 5, 10)" in r.extraction.description
    assert r.extraction.requirements.accommodation == (
        "4-star Hotel or comfortable Mountain Resort"
    )
    assert r.extraction.property_type == "hotel+resort"
    assert r.extraction.requirements.transport == "Private Tempo Traveller"
    assert r.extraction.rooms == 2
    assert "two family rooms with mountain views" in (
        r.extraction.requirements.special_requests or ""
    )
    assert "Pickup from Chandigarh" in (r.extraction.requirements.special_requests or "")
    assert r.missing_fields == []
    assert r.clarifying_questions == []
    assert r.completeness == 1.0


def test_duration_does_not_create_date_end(patch_gateway, mock_gateway):
    extraction = _full_extraction()
    extraction.travel_date_end = "17/12/26"
    mock_gateway.generate_structured_sync.return_value = extraction

    r = svc.understand("Trip to Goa on 15/12/26 for 3-5 nights")

    assert r.extraction.travel_date == "15/12/26"
    assert r.extraction.travel_date_end is None


def test_unambiguous_relative_date_is_not_discarded(patch_gateway, mock_gateway):
    extraction = _full_extraction()
    extraction.travel_date = "30/11/26"
    mock_gateway.generate_structured_sync.return_value = extraction

    r = svc.understand("Trip to Goa in two weeks")

    assert r.extraction.travel_date == "30/11/26"


def test_explicit_date_range_is_preserved(patch_gateway, mock_gateway):
    extraction = _full_extraction()
    extraction.travel_date_end = "20/12/26"
    mock_gateway.generate_structured_sync.return_value = extraction

    r = svc.understand("Goa from 15/12/26 to 20/12/26")

    assert r.extraction.travel_date == "15/12/26"
    assert r.extraction.travel_date_end == "20/12/26"


def test_date_range_with_shared_month_is_preserved(patch_gateway, mock_gateway):
    extraction = _full_extraction()
    extraction.travel_date = "24/12/26"
    extraction.travel_date_end = "29/12/26"
    mock_gateway.generate_structured_sync.return_value = extraction

    r = svc.understand("Kerala from 24 to 29 December")

    assert r.extraction.travel_date == "24/12/26"
    assert r.extraction.travel_date_end == "29/12/26"


# ---------------------------------------------------------------------------
# Missing fields
# ---------------------------------------------------------------------------


def test_partial_extraction_reports_missing(patch_gateway, mock_gateway):
    partial = EnquiryExtraction(
        destination="Goa",
        duration_days=3,
        travelers=2,
        description="Short trip",
        budget=EnquiryBudget(amount=20000),
        requirements=EnquiryRequirements(accommodation="Villa"),
    )
    mock_gateway.generate_structured_sync.return_value = partial

    r = svc.understand("test")
    assert FIELD_DESTINATION not in r.missing_fields
    assert FIELD_ACCOMMODATION not in r.missing_fields
    # Preferences such as meals, activities, guides, and special requests are optional.
    assert "meals" not in r.missing_fields
    assert "activities" not in r.missing_fields
    assert "tour_guide" not in r.missing_fields
    assert "special_instructions" not in r.missing_fields
    assert "transport" in r.missing_fields


def test_completeness_score_partial(patch_gateway, mock_gateway):
    partial = EnquiryExtraction(destination="Goa")
    mock_gateway.generate_structured_sync.return_value = partial
    r = svc.understand("test")
    assert 0.0 < r.completeness < 1.0


def test_completeness_and_questions_use_required_fields_only(patch_gateway, mock_gateway):
    mock_gateway.generate_structured_sync.return_value = EnquiryExtraction(
        destination="Manali",
        travelers=2,
    )

    result = svc.understand("I want to plan a trip to Manali for 2 people.")

    assert result.completeness == 0.67
    assert {question.field for question in result.clarifying_questions} == {"when"}


# ---------------------------------------------------------------------------
# Currency hint
# ---------------------------------------------------------------------------


def test_currency_hint_fills_missing_currency(patch_gateway, mock_gateway):
    extraction = _full_extraction()
    extraction.budget = EnquiryBudget(amount=60000, currency=None)
    mock_gateway.generate_structured_sync.return_value = extraction

    r = svc.understand("test", currency_hint="inr")
    assert r.extraction.budget.currency == "INR"


def test_currency_hint_does_not_overwrite_present_currency(patch_gateway, mock_gateway):
    extraction = _full_extraction()
    extraction.budget = EnquiryBudget(amount=60000, currency="GBP")
    mock_gateway.generate_structured_sync.return_value = extraction

    r = svc.understand("test", currency_hint="INR")
    assert r.extraction.budget.currency == "GBP"


def test_currency_hint_ignored_when_budget_is_none(patch_gateway, mock_gateway):
    extraction = _full_extraction()
    extraction.budget = None
    mock_gateway.generate_structured_sync.return_value = extraction

    r = svc.understand("test", currency_hint="INR")
    assert r.extraction.budget is None


# ---------------------------------------------------------------------------
# Clarifying questions
# ---------------------------------------------------------------------------


def test_questions_returned_for_missing_required(patch_gateway, mock_gateway):
    partial = EnquiryExtraction()
    mock_gateway.generate_structured_sync.return_value = partial
    r = svc.understand("test")
    assert len(r.clarifying_questions) > 0
    fields = {q.field for q in r.clarifying_questions}
    assert "destination" in fields


def test_questions_capped_at_max(patch_gateway, mock_gateway):
    from app.capabilities.enquiry.constants import MAX_CLARIFYING_QUESTIONS

    partial = EnquiryExtraction()
    mock_gateway.generate_structured_sync.return_value = partial
    r = svc.understand("test")
    assert len(r.clarifying_questions) <= MAX_CLARIFYING_QUESTIONS


# ---------------------------------------------------------------------------
# Gateway errors propagate
# ---------------------------------------------------------------------------


def test_gateway_errors_propagate(patch_gateway, mock_gateway):
    from app.core.exceptions import AIProviderError

    mock_gateway.generate_structured_sync.side_effect = AIProviderError("boom")
    with pytest.raises(AIProviderError):
        svc.understand("test")


def test_title_is_deterministic_and_descriptive(patch_gateway, mock_gateway):
    extraction = _full_extraction()
    extraction.title = "Hello, here is your trip!"
    extraction.trip_type = "honeymoon"
    extraction.destination = "Kerala"
    extraction.duration_days = 6
    mock_gateway.generate_structured_sync.return_value = extraction

    result = svc.understand("6-day honeymoon trip to Kerala")

    assert result.extraction.title == "6-Day honeymoon trip to Kerala"


def test_title_falls_back_to_destination(patch_gateway, mock_gateway):
    extraction = EnquiryExtraction(destination="Goa")
    mock_gateway.generate_structured_sync.return_value = extraction

    result = svc.understand("Trip to Goa")

    assert result.extraction.title == "Trip to Goa"


def test_hotel_and_resort_ratings_are_preserved_separately(
    patch_gateway,
    mock_gateway,
):
    extraction = EnquiryExtraction()
    mock_gateway.generate_structured_sync.return_value = extraction

    result = svc.understand(
        "I want a 4-star hotel or a 5-star resort in Goa"
    )

    assert result.extraction.requirements.accommodation == (
        "4-star Hotel or 5-star Resort"
    )


def test_private_cab_does_not_make_houseboat_private(
    patch_gateway,
    mock_gateway,
):
    extraction = EnquiryExtraction(
        requirements=EnquiryRequirements(
            transport="Private Cab",
            accommodation="Houseboat",
        )
    )
    mock_gateway.generate_structured_sync.return_value = extraction

    result = svc.understand(
        "Arrange a private cab to Kochi and a houseboat stay for one night"
    )

    special = result.extraction.requirements.special_requests or ""
    assert "houseboat stay for one night" in special.casefold()
    assert "private houseboat" not in special.casefold()


def test_explicit_private_one_night_houseboat_is_preserved(
    patch_gateway,
    mock_gateway,
):
    extraction = EnquiryExtraction()
    mock_gateway.generate_structured_sync.return_value = extraction

    result = svc.understand(
        "I want a private houseboat stay for one night in Alleppey"
    )

    special = result.extraction.requirements.special_requests or ""
    assert "private houseboat stay for one night" in special.casefold()


def test_budget_range_is_recovered_when_llm_omits_it(
    patch_gateway,
    mock_gateway,
):
    mock_gateway.generate_structured_sync.return_value = EnquiryExtraction(
        destination="Goa",
        duration_days=3,
        travelers=2,
    )

    result = svc.understand(
        "Trip to Goa for 2 people. Our budget is between 40k and 60k INR total."
    )

    assert result.extraction.budget_range_min == 40000
    assert result.extraction.budget_range_max == 60000
    assert "budget" not in result.missing_fields


def test_budget_per_person_is_preserved_by_fallback(
    patch_gateway,
    mock_gateway,
):
    mock_gateway.generate_structured_sync.return_value = EnquiryExtraction(
        destination="Goa",
        duration_days=3,
        travelers=2,
    )

    result = svc.understand(
        "Trip to Goa for 2 people. Budget is 25k INR per person."
    )

    assert result.extraction.budget is not None
    assert result.extraction.budget.amount == 25000
    assert result.extraction.budget.currency == "INR"
    assert result.extraction.budget.is_per_person is True


def test_travelers_are_derived_from_explicit_adults_and_children(
    patch_gateway,
    mock_gateway,
):
    mock_gateway.generate_structured_sync.return_value = EnquiryExtraction(
        destination="Manali",
        adults=2,
        children=[{"age": 7}, {"age": 12}],
        duration_days=5,
    )

    result = svc.understand("Family trip to Manali for 2 adults and 2 children")

    assert result.extraction.travelers == 4


def test_date_range_derives_duration_only_when_missing(
    patch_gateway,
    mock_gateway,
):
    mock_gateway.generate_structured_sync.return_value = EnquiryExtraction(
        destination="Goa",
        travel_date="15/12/26",
        travel_date_end="17/12/26",
        travelers=2,
    )

    result = svc.understand("Trip to Goa from 15/12/26 to 17/12/26")

    assert result.extraction.duration_days == 3
    assert result.extraction.duration_nights == 2


def test_reversed_date_range_is_rejected(patch_gateway, mock_gateway):
    mock_gateway.generate_structured_sync.return_value = EnquiryExtraction(
        destination="Goa",
        travel_date="20/12/26",
        travel_date_end="15/12/26",
        travelers=2,
    )

    result = svc.understand("Trip to Goa from 15/12/26 to 20/12/26")

    assert result.extraction.travel_date is None
    assert result.extraction.travel_date_end is None


def test_duration_does_not_generate_end_date(patch_gateway, mock_gateway):
    mock_gateway.generate_structured_sync.return_value = EnquiryExtraction(
        destination="Goa",
        travel_date="15/12/26",
        duration_days=5,
        travelers=2,
    )

    result = svc.understand("5-day trip to Goa starting 15/12/26")

    assert result.extraction.travel_date == "15/12/26"
    assert result.extraction.travel_date_end is None


def test_transport_phrases_are_preserved_and_do_not_prompt_again(
    patch_gateway,
    mock_gateway,
):
    mock_gateway.generate_structured_sync.return_value = EnquiryExtraction(
        destination="Goa",
        duration_days=3,
        travelers=2,
    )

    result = svc.understand("Plan a 3-day trip to Goa for 2 people with local transportation.")

    assert result.extraction.requirements.transport == "Local Transportation"
    assert "transport" not in result.missing_fields
    assert all(question.field != "transport" for question in result.clarifying_questions)


def test_airport_transfers_are_transport_not_only_special_requests(
    patch_gateway,
    mock_gateway,
):
    mock_gateway.generate_structured_sync.return_value = EnquiryExtraction(
        destination="Goa",
        duration_days=3,
        travelers=2,
    )

    result = svc.understand("Plan a 3-day trip to Goa for 2 people with airport transfers.")

    assert result.extraction.requirements.transport == "Airport Transfers"
    assert "transport" not in result.missing_fields


def test_private_transport_and_requested_stops_are_preserved(
    patch_gateway,
    mock_gateway,
):
    mock_gateway.generate_structured_sync.return_value = EnquiryExtraction(
        destination="Rajasthan",
        travelers=3,
        adults=3,
        travel_date="2027-01-15",
        travel_date_end="2027-01-21",
        requirements=EnquiryRequirements(
            special_requests="Drop-off at visit Jaipur",
        ),
    )

    result = svc.understand(
        "We want to visit Rajasthan from 15 January to 21 January 2027 for "
        "3 adults. Visit Jaipur, Jodhpur and Udaipur with private transportation."
    )

    assert result.extraction.requirements.transport == "Private Transportation"
    assert result.extraction.travel_date == "15/01/27"
    assert result.extraction.travel_date_end == "21/01/27"
    special = result.extraction.requirements.special_requests or ""
    assert "Sightseeing in Jaipur, Jodhpur and Udaipur" in special
    assert "Drop-off at visit Jaipur" not in special


def test_unspecified_people_default_to_adults_and_specific_activity_wins(
    patch_gateway,
    mock_gateway,
):
    mock_gateway.generate_structured_sync.return_value = EnquiryExtraction(
        destination="Kerala",
        duration_days=6,
        travelers=2,
        requirements=EnquiryRequirements(
            activities=["Couple Photography", "Photography", "couple photography"],
        ),
    )

    result = svc.understand("6-day Kerala trip for 2 people with couple photography")

    assert result.extraction.adults == 2
    assert result.extraction.travelers == 2
    assert result.extraction.requirements.activities == ["Couple Photography"]


def test_special_requests_deduplicate_pickups_and_sightseeing(
    patch_gateway,
    mock_gateway,
):
    mock_gateway.generate_structured_sync.return_value = EnquiryExtraction(
        destination="Kerala",
        requirements=EnquiryRequirements(
            special_requests=(
                "pickup at Kochi airport; explore Munnar and Alleppey backwaters"
            ),
        ),
    )

    result = svc.understand(
        "Arrange a private cab from Kochi airport and explore Munnar and "
        "Alleppey backwaters."
    )

    special = result.extraction.requirements.special_requests or ""
    assert special.casefold().count("kochi airport") == 1
    assert special.casefold().count("munnar and alleppey backwaters") == 1


def test_special_requests_omit_redundant_room_sentence(
    patch_gateway,
    mock_gateway,
):
    mock_gateway.generate_structured_sync.return_value = EnquiryExtraction(
        destination="Himachal Pradesh",
        requirements=EnquiryRequirements(
            special_requests="Two family rooms, activities suitable for children",
        ),
    )

    result = svc.understand(
        "We need two family rooms in a 4-star hotel with breakfast included."
    )

    assert result.extraction.requirements.special_requests == (
        "Two family rooms; activities suitable for children"
    )
