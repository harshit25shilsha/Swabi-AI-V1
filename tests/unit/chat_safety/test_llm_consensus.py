"""Unit tests for the LLM consensus orchestrator in app.llm.moderator.

These patch _classify_with_fallback directly, so they exercise the
orchestrator logic without touching Groq or reading GROQ_API_KEY.
"""
from unittest.mock import patch

import pytest

from app.capabilities.chat_safety.moderator import moderate_with_llm
from app.capabilities.chat_safety.schemas import Action, LLMResult

# imports
from app.core.config import settings

# patch target (was "app.llm.moderator._classify_with_fallback")
PATCH_TARGET = "app.capabilities.chat_safety.moderator._classify_with_fallback"

def _allow(conf: float = 0.9) -> LLMResult:
    return LLMResult(decision=Action.ALLOW, category=None, confidence=conf)


def _block(cat: str = "PHONE_NUMBER", conf: float = 0.9) -> LLMResult:
    return LLMResult(decision=Action.BLOCK, category=cat, confidence=conf)


def _ok(result: LLMResult, provider: str = "groq"):
    """Wrap a result as _classify_with_fallback's return value."""
    return (result, provider)



@pytest.fixture(autouse=True)
def _two_attempts(monkeypatch):
    monkeypatch.setattr(settings, "llm_consensus_attempts", 2)

def test_allow_then_block_returns_block():
    with patch(
        PATCH_TARGET,
        side_effect=[_ok(_allow()), _ok(_block())],
    ) as m:
        result, provider = moderate_with_llm("test")
    assert result.decision == Action.BLOCK
    assert result.category == "PHONE_NUMBER"
    assert provider == "groq"
    assert m.call_count == 2


def test_block_short_circuits_on_first_attempt():
    with patch(
        PATCH_TARGET,
        side_effect=[_ok(_block()), _ok(_allow())],
    ) as m:
        result, provider = moderate_with_llm("test")
    assert result.decision == Action.BLOCK
    assert provider == "groq"
    assert m.call_count == 1  # second call skipped


def test_both_allow_returns_highest_confidence():
    with patch(
        PATCH_TARGET,
        side_effect=[_ok(_allow(0.80)), _ok(_allow(0.95), "gemini")],
    ) as m:
        result, provider = moderate_with_llm("test")
    assert result.decision == Action.ALLOW
    assert result.confidence == 0.95
    assert provider == "gemini"     # provider of the winning result
    assert m.call_count == 2


def test_first_raises_second_blocks():
    with patch(
        PATCH_TARGET,
        side_effect=[RuntimeError("boom"), _ok(_block())],
    ) as m:
        result, provider = moderate_with_llm("test")
    assert result.decision == Action.BLOCK
    assert provider == "groq"
    assert m.call_count == 2


def test_first_raises_second_allows_returns_allow():
    with patch(
        PATCH_TARGET,
        side_effect=[RuntimeError("boom"), _ok(_allow(0.9))],
    ):
        result, provider = moderate_with_llm("test")
    assert result.decision == Action.ALLOW
    assert provider == "groq"


def test_all_attempts_raise_reraises_last_exception():
    with patch(PATCH_TARGET, side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError, match="boom"):
            moderate_with_llm("test")



def test_single_attempt_config_makes_one_call(monkeypatch):
    monkeypatch.setattr(settings, "llm_consensus_attempts", 1)
    with patch(
        PATCH_TARGET,
        side_effect=[_ok(_allow())],
    ) as m:
        result, provider = moderate_with_llm("test")
    assert result.decision == Action.ALLOW
    assert provider == "groq"
    assert m.call_count == 1


def test_multiple_blocks_returns_first_block(monkeypatch):
    monkeypatch.setattr(settings, "llm_consensus_attempts", 3)
    with patch(
        PATCH_TARGET,
        side_effect=[
            _ok(_allow()),
            _ok(_block(conf=0.6)),
            _ok(_block(conf=0.95)),
        ],
    ):
        result, provider = moderate_with_llm("test")
    assert result.decision == Action.BLOCK
    assert result.confidence == 0.6
    assert provider == "groq"
