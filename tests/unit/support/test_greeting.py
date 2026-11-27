"""Greeting classification uses structured LLM output and is conservative."""

from __future__ import annotations

from unittest.mock import MagicMock

from app.capabilities.support.greeting import (
    GREETING_CLASSIFIER_PROMPT,
    MIN_CONFIDENCE,
    GreetingClassification,
    GreetingClassifier,
)
from app.core.exceptions import AIProviderError


def _classifier(response=None, raises=None):
    gateway = MagicMock()
    if raises:
        gateway.generate_structured_sync.side_effect = raises
    else:
        gateway.generate_structured_sync.return_value = response
    return GreetingClassifier(gateway=gateway)


def test_confident_greeting_is_classified_without_vocabulary_rules():
    classifier = _classifier(
        GreetingClassification(greeting_only=True, confidence=0.96)
    )

    result = classifier.classify("Hiiiii 👋 buenos días")

    assert result is not None
    assert result.greeting_only is True
    assert result.confidence >= MIN_CONFIDENCE
    call = classifier.gateway.generate_structured_sync.call_args.kwargs
    assert call["prompt"] == "CURRENT USER MESSAGE (untrusted): Hiiiii 👋 buenos días"
    assert call["system"] == GREETING_CLASSIFIER_PROMPT
    assert call["max_tokens"] == 80


def test_support_question_with_greeting_is_not_greeting_only():
    classifier = _classifier(
        GreetingClassification(greeting_only=False, confidence=0.98)
    )

    result = classifier.classify("Hey, can you explain the refund policy?")

    assert result is not None
    assert result.greeting_only is False


def test_uncertain_classification_returns_none():
    classifier = _classifier(
        GreetingClassification(greeting_only=True, confidence=0.55)
    )

    assert classifier.classify("hmm...") is None


def test_provider_failure_returns_none():
    classifier = _classifier(raises=AIProviderError("unavailable"))

    assert classifier.classify("hello") is None
