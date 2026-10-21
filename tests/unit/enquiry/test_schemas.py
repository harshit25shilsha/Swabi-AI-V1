"""Validation tests for the enquiry Pydantic schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.capabilities.enquiry.schemas import (
    Child,
    ClarificationQuestion,
    DateFlexibility,
    EnquiryBudget,
    EnquiryExtraction,
    EnquiryExtractRequest,
    EnquiryRequirements,
    EnquiryUnderstandResponse,
)

# ---------------------------------------------------------------------------
# Child
# ---------------------------------------------------------------------------


def test_child_age_optional():
    c = Child()
    assert c.age is None


@pytest.mark.parametrize("age", [0, 7, 12, 17])
def test_child_age_in_range(age):
    assert Child(age=age).age == age


@pytest.mark.parametrize("age", [-1, 18, 100])
def test_child_age_out_of_range(age):
    with pytest.raises(ValidationError):
        Child(age=age)


# ---------------------------------------------------------------------------
# DateFlexibility
# ---------------------------------------------------------------------------


def test_date_flexibility_defaults():
    df = DateFlexibility()
    assert df.is_flexible is None
    assert df.window_days is None
    assert df.note is None


def test_date_flexibility_window_days_bounds():
    assert DateFlexibility(window_days=0).window_days == 0
    assert DateFlexibility(window_days=30).window_days == 30
    with pytest.raises(ValidationError):
        DateFlexibility(window_days=31)
    with pytest.raises(ValidationError):
        DateFlexibility(window_days=-1)


# ---------------------------------------------------------------------------
# EnquiryBudget
# ---------------------------------------------------------------------------


def test_budget_defaults():
    b = EnquiryBudget()
    assert b.amount is None
    assert b.currency is None
    assert b.is_per_person is None


def test_budget_amount_non_negative():
    assert EnquiryBudget(amount=0).amount == 0
    assert EnquiryBudget(amount=60000).amount == 60000
    with pytest.raises(ValidationError):
        EnquiryBudget(amount=-1)


# ---------------------------------------------------------------------------
# EnquiryRequirements
# ---------------------------------------------------------------------------


def test_requirements_defaults():
    r = EnquiryRequirements()
    assert r.accommodation is None
    assert r.meals == []
    assert r.transport is None
    assert r.activities == []
    assert r.tour_guide is None
    assert r.special_requests is None


def test_requirements_multi_select_lists():
    r = EnquiryRequirements(
        meals=["Breakfast", "Dinner"],
        activities=["Skiing", "Custom Activity"],
    )
    assert r.meals == ["Breakfast", "Dinner"]
    assert "Custom Activity" in r.activities


def test_requirements_accommodation_is_single_string():
    r = EnquiryRequirements(accommodation="5-Star Hotel")
    assert isinstance(r.accommodation, str)
    assert r.accommodation == "5-Star Hotel"


# ---------------------------------------------------------------------------
# EnquiryExtraction
# ---------------------------------------------------------------------------


def test_extraction_all_defaults():
    e = EnquiryExtraction()
    assert e.destination is None
    assert e.travel_date is None
    assert e.duration_days is None
    assert e.travelers is None
    assert e.children == []
    assert e.budget is None
    assert e.flags == []
    assert e.tags == []
    assert isinstance(e.requirements, EnquiryRequirements)


def test_extraction_duration_bounds():
    assert EnquiryExtraction(duration_days=0).duration_days == 0
    assert EnquiryExtraction(duration_days=365).duration_days == 365
    with pytest.raises(ValidationError):
        EnquiryExtraction(duration_days=366)


def test_extraction_travelers_bounds():
    assert EnquiryExtraction(travelers=1).travelers == 1
    assert EnquiryExtraction(travelers=50).travelers == 50
    with pytest.raises(ValidationError):
        EnquiryExtraction(travelers=0)


def test_extraction_accepts_children_with_ages():
    e = EnquiryExtraction(children=[{"age": 7}, {"age": 12}])
    assert len(e.children) == 2
    assert e.children[0].age == 7


def test_extraction_accepts_children_with_null_age():
    e = EnquiryExtraction(children=[{"age": None}])
    assert len(e.children) == 1
    assert e.children[0].age is None


# ---------------------------------------------------------------------------
# EnquiryExtractRequest
# ---------------------------------------------------------------------------


def test_request_requires_message():
    with pytest.raises(ValidationError):
        EnquiryExtractRequest()


def test_request_rejects_empty_message():
    with pytest.raises(ValidationError):
        EnquiryExtractRequest(message="")


def test_request_rejects_oversized_message():
    with pytest.raises(ValidationError):
        EnquiryExtractRequest(message="x" * 4001)


def test_request_accepts_max_length():
    EnquiryExtractRequest(message="x" * 4000)


def test_request_optional_fields():
    r = EnquiryExtractRequest(message="hi")
    assert r.user_id is None
    assert r.currency_hint is None


# ---------------------------------------------------------------------------
# EnquiryUnderstandResponse
# ---------------------------------------------------------------------------


def test_response_completeness_bounds():
    EnquiryUnderstandResponse(extraction=EnquiryExtraction(), completeness=0.0)
    EnquiryUnderstandResponse(extraction=EnquiryExtraction(), completeness=1.0)
    with pytest.raises(ValidationError):
        EnquiryUnderstandResponse(extraction=EnquiryExtraction(), completeness=1.5)
    with pytest.raises(ValidationError):
        EnquiryUnderstandResponse(extraction=EnquiryExtraction(), completeness=-0.1)


def test_response_metadata_defaults():
    r = EnquiryUnderstandResponse(extraction=EnquiryExtraction(), completeness=0.0)
    assert r.provider == ""
    assert r.prompt_version == ""
    assert r.request_id == ""
    assert r.missing_fields == []
    assert r.clarifying_questions == []


def test_clarification_question_shape():
    q = ClarificationQuestion(field="rooms", question="How many rooms?")
    assert q.field == "rooms"
    assert q.question == "How many rooms?"
