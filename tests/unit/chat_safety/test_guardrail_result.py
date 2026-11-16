"""Tests for the internal GuardrailResult shape.

Nothing consumes it yet — these tests establish the contract before step 9
starts using it.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from app.capabilities.chat_safety.schemas import (
    GuardrailDecision,
    GuardrailResult,
)

# ---------------------------------------------------------------------------
# GuardrailDecision
# ---------------------------------------------------------------------------


def test_guardrail_decision_values():
    assert GuardrailDecision.ALLOW == "ALLOW"
    assert GuardrailDecision.BLOCK == "BLOCK"
    assert GuardrailDecision.REVIEW == "REVIEW"
    assert GuardrailDecision.UNAVAILABLE == "UNAVAILABLE"


def test_guardrail_decision_is_str_enum():
    """String equality and JSON serialization behave like plain strings."""
    assert isinstance(GuardrailDecision.ALLOW, str)
    assert json.dumps({"d": GuardrailDecision.BLOCK}) == '{"d": "BLOCK"}'


def test_guardrail_decision_rejects_unknown_value():
    with pytest.raises(ValueError):
        GuardrailDecision("MAYBE")


# ---------------------------------------------------------------------------
# GuardrailResult — construction
# ---------------------------------------------------------------------------


def test_minimal_result():
    r = GuardrailResult(decision=GuardrailDecision.ALLOW)
    assert r.decision == GuardrailDecision.ALLOW
    assert r.reason_codes == []
    assert r.confidence is None
    assert r.source == "unknown"
    assert r.message is None


def test_full_result():
    r = GuardrailResult(
        decision=GuardrailDecision.BLOCK,
        reason_codes=["PHONE_NUMBER"],
        confidence=0.95,
        source="llm",
        message="Detected a disguised phone number",
    )
    assert r.decision == GuardrailDecision.BLOCK
    assert r.reason_codes == ["PHONE_NUMBER"]
    assert r.confidence == 0.95
    assert r.source == "llm"
    assert r.message == "Detected a disguised phone number"


def test_multiple_reason_codes():
    r = GuardrailResult(
        decision=GuardrailDecision.BLOCK,
        reason_codes=["PHONE_NUMBER", "OFF_PLATFORM_COMMUNICATION"],
    )
    assert len(r.reason_codes) == 2


# ---------------------------------------------------------------------------
# GuardrailResult — validation
# ---------------------------------------------------------------------------


def test_confidence_must_be_in_range():
    with pytest.raises(ValidationError):
        GuardrailResult(decision=GuardrailDecision.ALLOW, confidence=1.5)
    with pytest.raises(ValidationError):
        GuardrailResult(decision=GuardrailDecision.ALLOW, confidence=-0.1)


def test_confidence_accepts_boundary_values():
    r_low = GuardrailResult(decision=GuardrailDecision.ALLOW, confidence=0.0)
    r_high = GuardrailResult(decision=GuardrailDecision.BLOCK, confidence=1.0)
    assert r_low.confidence == 0.0
    assert r_high.confidence == 1.0


def test_decision_is_required():
    with pytest.raises(ValidationError):
        GuardrailResult()  # type: ignore[call-arg]


def test_decision_must_be_valid():
    with pytest.raises(ValidationError):
        GuardrailResult(decision="MAYBE")  # type: ignore[arg-type]


def test_result_is_json_serializable():
    r = GuardrailResult(
        decision=GuardrailDecision.BLOCK,
        reason_codes=["EMAIL"],
        confidence=0.9,
        source="deterministic",
    )
    dumped = r.model_dump_json()
    assert '"decision":"BLOCK"' in dumped
    assert '"EMAIL"' in dumped

    parsed = json.loads(dumped)
    assert parsed["decision"] == "BLOCK"
    assert parsed["reason_codes"] == ["EMAIL"]
    assert parsed["confidence"] == 0.9
    assert parsed["source"] == "deterministic"


# ---------------------------------------------------------------------------
# GuardrailResult is independent of ModerationResponse
# ---------------------------------------------------------------------------


def test_moderation_response_unchanged():
    """The public response shape must not have gained or lost fields."""
    from app.capabilities.chat_safety.schemas import ModerationResponse

    fields = set(ModerationResponse.model_fields.keys())
    expected = {
        "allowed",
        "action",
        "category",
        "confidence",
        "source",
        "request_id",
        "prompt_version",
        "provider",
    }
    assert fields == expected, (
        f"ModerationResponse fields drifted. "
        f"Extra: {fields - expected}. Missing: {expected - fields}."
    )
