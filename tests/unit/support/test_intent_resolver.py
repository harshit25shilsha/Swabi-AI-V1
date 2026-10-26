"""Resolver: deterministic hit, scope block, LLM fallback."""
from __future__ import annotations

from unittest.mock import MagicMock

from app.capabilities.support.intent_llm import LLMIntentResult
from app.capabilities.support.intent_resolver import IntentResolver
from app.capabilities.support.live_data_classifier import LiveDataIntent


def _resolver(llm_result=None):
    llm = MagicMock()
    llm.classify.return_value = llm_result
    return IntentResolver(llm_classifier=llm), llm


def test_deterministic_hit_skips_llm():
    r, llm = _resolver()
    out = r.resolve("What are my bookings?", "customer")
    assert out.intent == LiveDataIntent.MY_BOOKINGS
    assert out.source == "deterministic"
    llm.classify.assert_not_called()


def test_scope_block_skips_llm():
    r, llm = _resolver()
    out = r.resolve("What is my vendor wallet balance?", "customer")
    assert out.intent == LiveDataIntent.NONE
    assert "scope" in out.reason.lower()
    llm.classify.assert_not_called()


def test_no_match_consults_llm_and_uses_result():
    r, llm = _resolver(
        llm_result=LLMIntentResult(
            intent=LiveDataIntent.MY_BOOKINGS,
            confidence=0.9,
        )
    )
    # Phrasing that the deterministic classifier does not match.
    out = r.resolve("please surface the trips I recently confirmed", "customer")
    assert out.intent == LiveDataIntent.MY_BOOKINGS
    assert out.source == "llm"
    llm.classify.assert_called_once()


def test_llm_returns_none_stays_none():
    r, llm = _resolver(
        llm_result=LLMIntentResult(
            intent=LiveDataIntent.NONE,
            confidence=0.95,
        )
    )
    out = r.resolve("what is the weather on mars", "customer")
    assert out.intent == LiveDataIntent.NONE
    assert out.source == "llm"


def test_llm_unavailable_returns_none():
    llm = MagicMock()
    llm.classify.return_value = None
    r = IntentResolver(llm_classifier=llm)
    out = r.resolve("please surface my recent confirmations", "customer")
    assert out.intent == LiveDataIntent.NONE
    assert out.source == "none"
