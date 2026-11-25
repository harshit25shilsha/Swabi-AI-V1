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
    r = svc.understand("test message")
    assert r.extraction.destination == "Manali"
    assert r.extraction.duration_days == 5
    assert r.completeness == 1.0
    assert r.missing_fields == []


def test_understand_passes_system_prompt(patch_gateway, mock_gateway):
    svc.understand("test message")
    kwargs = mock_gateway.generate_structured_sync.call_args.kwargs
    assert "system" in kwargs
    assert "structured-data extractor" in kwargs["system"]


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
    # Missing meals/transport/activities/tour_guide/special_requests
    assert "meals" in r.missing_fields
    assert "activities" in r.missing_fields


def test_completeness_score_partial(patch_gateway, mock_gateway):
    partial = EnquiryExtraction(destination="Goa")
    mock_gateway.generate_structured_sync.return_value = partial
    r = svc.understand("test")
    assert 0.0 < r.completeness < 1.0


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
