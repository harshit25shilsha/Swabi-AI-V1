"""LLM intent classifier — gateway is mocked."""

from __future__ import annotations

from unittest.mock import MagicMock

from app.capabilities.support.intent_llm import (
    MIN_CONFIDENCE,
    LLMIntentClassifier,
    LLMIntentResult,
)
from app.capabilities.support.live_data_classifier import LiveDataIntent
from app.capabilities.support.schemas import ConversationTurn
from app.core.exceptions import AIProviderError


def _result(intent, confidence=0.9, entities=None):
    return LLMIntentResult(intent=intent, confidence=confidence, entities=entities or {})


def _classifier(response=None, raises=None):
    gw = MagicMock()
    if raises:
        gw.generate_structured_sync.side_effect = raises
    else:
        gw.generate_structured_sync.return_value = response
    return LLMIntentClassifier(gateway=gw)


def test_high_confidence_intent_returned():
    c = _classifier(
        response=LLMIntentResult(
            intent=LiveDataIntent.MY_BOOKINGS,
            requested_limit=2,
            confidence=0.95,
        )
    )
    out = c.classify("show me my last booking", "customer")
    assert out is not None
    assert out.intent == LiveDataIntent.MY_BOOKINGS
    assert out.requested_limit == 2


def test_classifier_receives_bounded_conversation_context():
    turns = [
        ConversationTurn(role="user", content="Show packages in Goa"),
        ConversationTurn(role="assistant", content="I found a few."),
    ]
    c = _classifier(response=_result(LiveDataIntent.PACKAGE_SEARCH, 0.95))

    c.classify("Only two, please", "customer", conversation_history=turns)

    prompt = c.gateway.generate_structured_sync.call_args.kwargs["prompt"]
    assert "RECENT CONVERSATION" in prompt
    assert "Show packages in Goa" in prompt
    assert "CURRENT QUESTION (untrusted): Only two, please" in prompt


def test_low_confidence_returns_none():
    c = _classifier(response=_result(LiveDataIntent.MY_BOOKINGS, 0.5))
    assert c.classify("something", "customer") is None


def test_none_intent_returned():
    c = _classifier(response=_result(LiveDataIntent.NONE, 0.95))
    out = c.classify("what is the cancellation policy", "customer")
    assert out is not None
    assert out.intent == LiveDataIntent.NONE


def test_provider_error_returns_none():
    c = _classifier(raises=AIProviderError("boom"))
    assert c.classify("anything", "customer") is None


def test_validation_error_returns_none():
    from app.core.exceptions import AIValidationError

    c = _classifier(raises=AIValidationError("bad json"))
    assert c.classify("anything", "customer") is None


def test_unexpected_exception_returns_none():
    c = _classifier(raises=RuntimeError("unexpected"))
    assert c.classify("anything", "customer") is None


def test_confidence_threshold_constant_exists():
    assert 0.0 < MIN_CONFIDENCE < 1.0
