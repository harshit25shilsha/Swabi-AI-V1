"""Tests for the enquiry service with a mocked LLM gateway."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.capabilities.enquiry import service as svc
from app.capabilities.enquiry.constants import (
    FIELD_ACCOMMODATION,
    FIELD_DESTINATION,
)
from app.capabilities.enquiry.prompts import (
    ENQUIRY_PROMPT_VERSION,
    ENQUIRY_SYSTEM_PROMPT,
)
from app.capabilities.enquiry.schemas import (
    DateFlexibility,
    EnquiryBudget,
    EnquiryExtraction,
    EnquiryExtractRequest,
    EnquiryRequirements,
)
from app.llm.gateway import _build_structured_prompt
from app.llm.groq import GroqProvider, _estimate_request_tokens

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _full_extraction() -> EnquiryExtraction:
    return EnquiryExtraction(
        title="5-day family trip to Manali",
        description="5-day family trip",
        destination="Manali",
        travel_date="15/12/26",
        travel_date_end="19/12/26",
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


# def test_understand_returns_response(patch_gateway, mock_gateway):
#     r = svc.understand("5-day family trip to Manali from 15/12/26 to 19/12/26")
#     assert r.extraction.destination == "Manali"
#     assert r.extraction.duration_days == 5
#     assert r.completeness == 1.0
#     assert r.missing_fields == []


def test_understand_passes_system_prompt(patch_gateway, mock_gateway):
    svc.understand("test message")
    kwargs = mock_gateway.generate_structured_sync.call_args.kwargs
    assert "system" in kwargs
    assert "structured-data extractor" in kwargs["system"]
    assert (
        "every human-readable string in the JSON response MUST be idiomatic English"
        in kwargs["system"]
    )
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


def test_maximum_enquiry_message_fits_active_groq_budget():
    system = ENQUIRY_SYSTEM_PROMPT.replace("{reference_date}", "17/11/2026")
    max_message_length = EnquiryExtractRequest.model_json_schema()["properties"]["message"][
        "maxLength"
    ]
    prompt = _build_structured_prompt("x" * max_message_length, EnquiryExtraction)
    payload = GroqProvider()._build_payload(
        prompt,
        system,
        temperature=0.0,
        json_mode=True,
        extra={},
    )

    assert _estimate_request_tokens(payload) <= 8000


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
    assert "travel_date" in r.missing_fields
    assert "travel_date_end" in r.missing_fields


def test_honeymoon_example_reports_required_answers_not_given(
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
    assert r.missing_fields == ["travel_date", "travel_date_end", "tour_guide"]
    assert [question.field for question in r.clarifying_questions] == r.missing_fields
    assert r.completeness == 0.6667
    expected_extraction = r.extraction.model_dump()
    repeated = svc.understand(message)
    assert repeated.extraction.model_dump() == expected_extraction


def test_family_example_preserves_dates_destinations_transport_and_room_type(
    patch_gateway,
    mock_gateway,
):
    extraction = EnquiryExtraction(
        title="7-day family vacation to Himachal Pradesh",
        description=(
            "7-day family vacation to Himachal Pradesh for a family of 6, "
            "visiting Shimla, Manali and Solang Valley"
        ),
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
                "walking; Two family rooms with mountain views; Pickup from Chandigarh"
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
    assert (
        "two family rooms with mountain views"
        in (r.extraction.requirements.special_requests or "").casefold()
    )
    assert "Pickup from Chandigarh" in (r.extraction.requirements.special_requests or "")
    assert r.missing_fields == ["travel_date", "travel_date_end", "tour_guide"]
    assert [question.field for question in r.clarifying_questions] == r.missing_fields
    assert r.completeness == 0.6667


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


def test_non_english_month_names_preserve_model_extracted_date_range(
    patch_gateway,
    mock_gateway,
):
    mock_gateway.generate_structured_sync.return_value = EnquiryExtraction(
        destination="Rajasthan",
        travel_date="15/01/27",
        travel_date_end="21/01/27",
        travelers=3,
    )

    result = svc.understand("हम 15 जनवरी से 21 जनवरी 2027 तक राजस्थान की यात्रा की योजना बना रहे हैं।")

    assert result.extraction.travel_date == "15/01/27"
    assert result.extraction.travel_date_end == "21/01/27"


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
    assert "meals" in r.missing_fields
    assert "activities" in r.missing_fields
    assert "tour_guide" in r.missing_fields
    assert "special_requests" in r.missing_fields
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

    assert result.completeness == 0.1111
    assert {question.field for question in result.clarifying_questions} == {
        "travel_date", "travel_date_end", "accommodation", "transport",
        "meals", "tour_guide", "activities", "special_requests",
    }


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

    result = svc.understand("I want a 4-star hotel or a 5-star resort in Goa")

    assert result.extraction.requirements.accommodation == ("4-star Hotel or 5-star Resort")


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

    result = svc.understand("Arrange a private cab to Kochi and a houseboat stay for one night")

    special = result.extraction.requirements.special_requests or ""
    assert "houseboat stay for one night" in special.casefold()
    assert "private houseboat" not in special.casefold()


def test_explicit_private_one_night_houseboat_is_preserved(
    patch_gateway,
    mock_gateway,
):
    extraction = EnquiryExtraction()
    mock_gateway.generate_structured_sync.return_value = extraction

    result = svc.understand("I want a private houseboat stay for one night in Alleppey")

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

    result = svc.understand("Trip to Goa for 2 people. Budget is 25k INR per person.")

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
            special_requests="Sightseeing in Jaipur, Jodhpur and Udaipur",
        ),
    )

    result = svc.understand(
        "We want to visit Rajasthan from 15 January to 21 January 2027 for "
        "3 adults. Visit Jaipur, Jodhpur and Udaipur with private transportation."
    )

    assert result.extraction.requirements.transport == "Private Transportation"
    assert result.extraction.requirements.activities == [
        "Sightseeing in Jaipur",
        "Sightseeing in Jodhpur",
        "Sightseeing in Udaipur",
    ]
    assert result.extraction.travel_date == "15/01/27"
    assert result.extraction.travel_date_end == "21/01/27"
    special = result.extraction.requirements.special_requests or ""
    assert "Sightseeing in Jaipur, Jodhpur and Udaipur" in special


def test_total_people_do_not_imply_adults_and_specific_activity_wins(
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

    assert result.extraction.adults is None
    assert result.extraction.travelers == 2
    assert result.extraction.requirements.activities == ["Couple Photography"]


def test_explicit_traveller_composition_overrides_inconsistent_model_counts(
    patch_gateway,
    mock_gateway,
):
    mock_gateway.generate_structured_sync.return_value = EnquiryExtraction(
        travelers=8,
        adults=8,
        children=[],
    )

    result = svc.understand("Trip for 2 adults and 2 children aged 6 and 9")

    assert result.extraction.adults == 2
    assert result.extraction.travelers == 4
    assert [child.age for child in result.extraction.children] == [6, 9]


def test_generic_guide_request_never_invents_a_language(patch_gateway, mock_gateway):
    mock_gateway.generate_structured_sync.return_value = EnquiryExtraction(
        requirements=EnquiryRequirements(tour_guide="English Speaking Guide"),
    )

    result = svc.understand("We would like a guide for the trip")

    assert result.extraction.requirements.tour_guide == "Yes"


def test_explicit_local_tour_guide_preserves_guide_type(patch_gateway, mock_gateway):
    mock_gateway.generate_structured_sync.return_value = EnquiryExtraction(
        requirements=EnquiryRequirements(tour_guide="Yes"),
    )

    result = svc.understand("We prefer a local tour guide")

    assert result.extraction.requirements.tour_guide == "Local Language Guide"


def test_negative_preference_is_retained_but_requires_confirmation(
    patch_gateway,
    mock_gateway,
):
    mock_gateway.generate_structured_sync.return_value = EnquiryExtraction(
        requirements=EnquiryRequirements(special_requests="No special requests"),
    )

    result = svc.understand("No special requirements")

    assert result.extraction.requirements.special_requests == "No special requests"
    assert "special_requests" in result.missing_fields
    question = next(q for q in result.clarifying_questions if q.field == "special_requests")
    assert question.question == (
        "Please confirm that you have no special requirements or travel instructions."
    )


def test_description_is_generated_from_sufficient_structured_trip_details(
    patch_gateway,
    mock_gateway,
):
    mock_gateway.generate_structured_sync.return_value = EnquiryExtraction(
        destination="Goa",
        travelers=2,
    )

    result = svc.understand("Trip to Goa for 2 people")

    assert result.extraction.description == "Trip to Goa; 2 travellers"


def test_activity_destination_mismatch_is_flagged_without_removing_activity(
    patch_gateway,
    mock_gateway,
):
    mock_gateway.generate_structured_sync.return_value = EnquiryExtraction(
        destination="Rajasthan",
        requirements=EnquiryRequirements(activities=["Gulmarg Gondola"]),
    )

    result = svc.understand("Rajasthan trip with a Gulmarg Gondola ride")

    assert result.extraction.requirements.activities == ["Gulmarg Gondola"]
    assert "activity_destination_review" in result.extraction.flags


def test_equivalent_generic_sightseeing_labels_are_deduplicated(
    patch_gateway,
    mock_gateway,
):
    mock_gateway.generate_structured_sync.return_value = EnquiryExtraction(
        requirements=EnquiryRequirements(
            activities=["Backwater Cruise", "Sightseeing", "Local Sightseeing"],
        ),
    )

    result = svc.understand("We would like a backwater cruise and sightseeing")

    assert result.extraction.requirements.activities == [
        "Backwater Cruise",
        "Local Sightseeing",
    ]


def test_explicit_property_preferences_are_extracted_but_not_required(
    patch_gateway,
    mock_gateway,
):
    mock_gateway.generate_structured_sync.return_value = EnquiryExtraction()

    result = svc.understand(
        "Work trip with my dog; we need an entire apartment that is couple-friendly."
    )

    assert result.extraction.property_preferences == [
        "Travel for Work",
        "Travel with Pets",
        "Entire Apartment",
        "Couple Friendly",
    ]
    assert "property_preferences" not in result.missing_fields


def test_special_requests_deduplicate_pickups_and_sightseeing(
    patch_gateway,
    mock_gateway,
):
    mock_gateway.generate_structured_sync.return_value = EnquiryExtraction(
        destination="Kerala",
        requirements=EnquiryRequirements(
            special_requests=("pickup at Kochi airport; explore Munnar and Alleppey backwaters"),
        ),
    )

    result = svc.understand(
        "Arrange a private cab from Kochi airport and explore Munnar and Alleppey backwaters."
    )

    special = result.extraction.requirements.special_requests or ""
    assert special.casefold().count("kochi airport") == 1
    assert special.casefold().count("munnar and alleppey backwaters") == 1


def test_hinglish_sightseeing_stops_do_not_leak_into_english_fields(
    patch_gateway,
    mock_gateway,
):
    mock_gateway.generate_structured_sync.return_value = EnquiryExtraction(
        description=(
            "Planning a Rajasthan trip from 15 January to 21 January 2027, "
            "exploring Jaipur, Jodhpur and Udaipur, with 3 adults and private "
            "transportation for the entire trip"
        ),
        destination="Rajasthan",
        travel_date="15/01/27",
        travel_date_end="21/01/27",
        travelers=3,
        adults=3,
        requirements=EnquiryRequirements(
            transport="Private Transportation",
            special_requests="Sightseeing in Jaipur, Jodhpur and Udaipur",
        ),
    )

    result = svc.understand(
        "Hum 15 January se 21 January 2027 tak Rajasthan trip plan kar rahe "
        "hain. Hum total 3 adults hain. Hum Jaipur, Jodhpur aur Udaipur "
        "explore karna chahte hain, aur poori trip ke liye private "
        "transportation chahiye."
    )

    assert result.extraction.requirements.special_requests == (
        "Sightseeing in Jaipur, Jodhpur and Udaipur"
    )
    assert "karna chahte hain" not in (result.extraction.description or "")
    assert "poori trip ke liye" not in (result.extraction.description or "")


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

    result = svc.understand("We need two family rooms in a 4-star hotel with breakfast included.")

    assert result.extraction.requirements.special_requests == (
        "Two family rooms; activities suitable for children"
    )
