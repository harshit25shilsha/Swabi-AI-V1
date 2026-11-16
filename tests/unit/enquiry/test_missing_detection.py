"""Tests for the deterministic missing-field detection logic."""

from __future__ import annotations

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
    OPTIONAL_FIELDS,
    RECOMMENDED_FIELDS,
    REQUIRED_FIELDS,
)
from app.capabilities.enquiry.schemas import (
    DateFlexibility,
    EnquiryBudget,
    EnquiryExtraction,
    EnquiryRequirements,
)
from app.capabilities.enquiry.service import (
    detect_missing_fields,
    generate_clarifying_questions,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _full_extraction() -> EnquiryExtraction:
    return EnquiryExtraction(
        title="Test",
        description="A brief description.",
        destination="Manali",
        travel_date="15/12/26",
        duration_days=5,
        duration_nights=4,
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


# ---------------------------------------------------------------------------
# Empty extraction
# ---------------------------------------------------------------------------


def test_empty_extraction_flags_required_and_recommended_only():
    missing = detect_missing_fields(EnquiryExtraction())
    for f in REQUIRED_FIELDS:
        assert f in missing, f"expected {f} in missing"
    for f in RECOMMENDED_FIELDS:
        if f == FIELD_TRAVEL_DATE:
            continue
        assert f in missing, f"expected {f} in missing"
    for f in OPTIONAL_FIELDS:
        assert f not in missing, f"optional field {f} should not be missing"


def test_missing_order_starts_with_required():
    missing = detect_missing_fields(EnquiryExtraction())
    # The first len(REQUIRED_FIELDS) items should be required fields,
    # in the REQUIRED_FIELDS order.
    assert missing[: len(REQUIRED_FIELDS)] == list(REQUIRED_FIELDS)


# ---------------------------------------------------------------------------
# Fully populated extraction
# ---------------------------------------------------------------------------


def test_full_extraction_has_no_missing():
    assert detect_missing_fields(_full_extraction()) == []


# ---------------------------------------------------------------------------
# "when" is satisfied by any of travel_date / duration_days / duration_nights
# ---------------------------------------------------------------------------


def test_when_satisfied_by_travel_date_only():
    e = EnquiryExtraction(travel_date="15/12/26")
    assert FIELD_WHEN not in detect_missing_fields(e)


def test_when_satisfied_by_duration_days_only():
    e = EnquiryExtraction(duration_days=5)
    assert FIELD_WHEN not in detect_missing_fields(e)


def test_when_satisfied_by_duration_nights_only():
    e = EnquiryExtraction(duration_nights=4)
    assert FIELD_WHEN not in detect_missing_fields(e)


def test_when_flagged_when_no_date_or_duration():
    e = EnquiryExtraction()
    assert FIELD_WHEN in detect_missing_fields(e)


# ---------------------------------------------------------------------------
# Askable vs required
# ---------------------------------------------------------------------------


def test_askable_duration_flagged_when_duration_missing():
    e = EnquiryExtraction(travel_date="15/12/26")
    missing = detect_missing_fields(e)
    assert FIELD_DURATION in missing
    assert FIELD_TRAVEL_DATE not in missing  # travel_date is present
    assert FIELD_WHEN not in missing  # when is satisfied


def test_askable_travel_date_flagged_when_date_missing():
    e = EnquiryExtraction(duration_days=5)
    missing = detect_missing_fields(e)
    assert FIELD_TRAVEL_DATE in missing
    assert FIELD_DURATION not in missing  # duration is present
    assert FIELD_WHEN not in missing  # when is satisfied


# ---------------------------------------------------------------------------
# Requirements-level fields
# ---------------------------------------------------------------------------


def test_requirements_fields_flagged_individually():
    e = EnquiryExtraction(
        destination="Goa",
        travel_date="01/01/26",
        travelers=2,
        description="Short",
        budget=EnquiryBudget(amount=10000),
        requirements=EnquiryRequirements(
            accommodation="Villa",
            # meals, transport, tour_guide, activities, special_requests missing
        ),
    )
    missing = detect_missing_fields(e)
    assert FIELD_MEALS not in missing
    assert FIELD_TRANSPORT in missing
    assert FIELD_TOUR_GUIDE not in missing
    assert FIELD_ACTIVITIES not in missing
    assert FIELD_SPECIAL_INSTRUCTIONS not in missing
    assert FIELD_ACCOMMODATION not in missing


def test_empty_description_is_optional():
    e = EnquiryExtraction(description="")
    assert FIELD_DESCRIPTION not in detect_missing_fields(e)


def test_empty_special_requests_are_optional():
    e = EnquiryExtraction(requirements=EnquiryRequirements(special_requests=""))
    assert FIELD_SPECIAL_INSTRUCTIONS not in detect_missing_fields(e)


def test_broad_date_window_satisfies_when_without_an_exact_date():
    e = EnquiryExtraction(
        destination="Kerala",
        duration_days=6,
        travelers=2,
        date_flexibility=DateFlexibility(
            is_flexible=True,
            note="last week of December",
        ),
    )
    missing = detect_missing_fields(e)
    assert FIELD_WHEN not in missing
    assert FIELD_TRAVEL_DATE not in missing
    assert FIELD_DURATION not in missing


def test_generic_flexibility_note_does_not_replace_a_travel_window():
    e = EnquiryExtraction(
        date_flexibility=DateFlexibility(is_flexible=True, note="Dates are flexible"),
    )
    assert FIELD_WHEN in detect_missing_fields(e)


# ---------------------------------------------------------------------------
# Clarifying questions
# ---------------------------------------------------------------------------


def test_clarifying_questions_for_missing_required():
    missing = [FIELD_DESTINATION, FIELD_TRAVELERS]
    qs = generate_clarifying_questions(missing)
    assert len(qs) == 2
    assert qs[0].field == FIELD_DESTINATION
    assert qs[1].field == FIELD_TRAVELERS
    assert qs[0].question


def test_clarifying_questions_capped_at_max():
    missing = list(REQUIRED_FIELDS) + list(RECOMMENDED_FIELDS)
    qs = generate_clarifying_questions(missing)
    assert len(qs) <= MAX_CLARIFYING_QUESTIONS


def test_clarifying_questions_skips_optional_and_unknown_fields():
    qs = generate_clarifying_questions(
        ["not_a_real_field", FIELD_TOUR_GUIDE, FIELD_ACTIVITIES]
    )
    assert qs == []


def test_clarifying_questions_defer_recommendations_until_core_is_complete():
    qs = generate_clarifying_questions(
        [FIELD_DESTINATION, FIELD_BUDGET, FIELD_ACCOMMODATION]
    )
    assert [question.field for question in qs] == [FIELD_DESTINATION]


def test_clarifying_questions_ask_recommendations_when_core_is_complete():
    qs = generate_clarifying_questions([FIELD_BUDGET, FIELD_ACCOMMODATION])
    assert [question.field for question in qs] == [FIELD_BUDGET, FIELD_ACCOMMODATION]


def test_clarifying_questions_empty_input():
    assert generate_clarifying_questions([]) == []
