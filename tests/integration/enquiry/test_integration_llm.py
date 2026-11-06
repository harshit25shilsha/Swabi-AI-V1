"""Live tests against the real Groq API.

Run with:  pytest -m integration -v
Skip in CI: excluded by default via `-m 'not integration'`.
"""

from __future__ import annotations

import pytest

from app.capabilities.enquiry.schemas import EnquiryExtraction
from app.capabilities.enquiry.service import understand
from app.core.config import settings

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not settings.groq_api_key or settings.groq_api_key == "test-groq-key",
        reason="Requires a real GROQ_API_KEY in .env",
    ),
]


# ---------------------------------------------------------------------------
# Basic extraction
# ---------------------------------------------------------------------------


def test_extracts_destination_and_duration():
    r = understand("I want a 5-day trip to Manali in December, budget 60000 INR.")
    assert r.extraction.destination == "Manali"
    assert r.extraction.duration_days == 5


def test_extracts_family_with_children():
    r = understand("5-day family trip to Manali for 2 adults and two children aged 7 and 12.")
    e = r.extraction
    assert e.travelers == 4 or (e.adults == 2 and len(e.children) == 2)
    assert e.adults == 2
    assert [c.age for c in e.children] == [7, 12]
    assert e.trip_type == "family"


def test_extracts_accommodation_enum():
    r = understand("Need a 5-star hotel in Goa for 4 nights.")
    assert r.extraction.requirements.accommodation == "5-Star Hotel"


def test_extracts_multi_select_meals():
    r = understand("3-day trip to Jaipur, breakfast and dinner included, 4-star hotel.")
    meals = r.extraction.requirements.meals
    assert "Breakfast" in meals
    assert "Dinner" in meals


def test_extracts_canonical_activity_name():
    r = understand("Kashmir trip, we want a shikara ride and skiing.")
    activities = r.extraction.requirements.activities
    assert "Shikara Ride" in activities
    assert "Skiing" in activities


def test_extracts_custom_activity():
    r = understand("Goa trip, we want fly fishing.")
    activities = r.extraction.requirements.activities
    assert any("fishing" in a.lower() for a in activities)


def test_extracts_budget_with_k_suffix():
    r = understand("Trip to Kerala, budget around 60k INR.")
    assert r.extraction.budget is not None
    assert r.extraction.budget.amount == 60000
    assert r.extraction.budget.currency == "INR"


def test_extracts_budget_lakh():
    r = understand("Goa trip, budget 1.5 lakh total.")
    assert r.extraction.budget is not None
    assert r.extraction.budget.amount == 150000


def test_extracts_pets_flag():
    r = understand("Villa in Goa, 4 nights, we have a dog.")
    assert "pets" in r.extraction.flags


def test_extracts_special_requests():
    r = understand("Manali trip for senior citizens. Need wheelchair accessibility.")
    sr = r.extraction.requirements.special_requests
    assert sr is not None
    assert "wheelchair" in sr.lower()


# ---------------------------------------------------------------------------
# Date resolution — new rules from the updated prompt
# ---------------------------------------------------------------------------


def test_month_only_keeps_travel_date_null():
    """'in December' must NOT be resolved to a specific date."""
    r = understand("5-day trip to Manali in December.")
    assert r.extraction.travel_date is None


def test_explicit_date_is_resolved():
    r = understand("Trip to Goa on 15/12/26, 3 nights.")
    assert r.extraction.travel_date is not None
    assert "15" in r.extraction.travel_date
    assert "12" in r.extraction.travel_date


def test_date_range_produces_start_and_end():
    r = understand("Goa from 15/12/26 to 20/12/26.")
    assert r.extraction.travel_date is not None
    assert r.extraction.travel_date_end is not None


# ---------------------------------------------------------------------------
# Missing fields
# ---------------------------------------------------------------------------


def test_sparse_message_flags_missing_required():
    r = understand("I want to go somewhere.")
    assert "destination" in r.missing_fields
    assert r.completeness < 1.0


def test_full_message_has_high_completeness():
    r = understand(
        "5-day trip to Manali, 15/12/26, 4 travellers, "
        "budget 60000 INR, 5-star hotel, all meals, private SUV, "
        "English speaking guide, skiing and sightseeing, "
        "senior citizen friendly. We want a comfortable family trip."
    )
    assert r.completeness >= 0.8


# ---------------------------------------------------------------------------
# Response shape
# ---------------------------------------------------------------------------


def test_extraction_is_valid_pydantic():
    r = understand("2-day trip to Agra.")
    # Round-trip through the schema — if the LLM produced something
    # off-schema, the service would have raised AIValidationError.
    assert isinstance(r.extraction, EnquiryExtraction)


def test_prompt_version_returned():
    r = understand("Test trip.")
    assert r.prompt_version.startswith("enquiry_")
