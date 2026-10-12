"""Chat Safety + circuit breaker integration tests.

The state-machine tests live in tests/unit/test_circuit_breaker.py.
This file covers the interaction between the moderator's provider ladder
and the shared circuit registry.
"""
from __future__ import annotations

import pytest

from app.capabilities.chat_safety import moderator
from app.core.config import settings


@pytest.fixture(autouse=True)
def _reset_circuits():
    """Every test starts with a clean provider circuit registry."""
    moderator._provider_circuits.clear()
    yield
    moderator._provider_circuits.clear()


def test_moderate_with_llm_raises_when_all_circuits_open():
    """Open every provider circuit; moderate_with_llm must raise.

    TODO(step 9): switch to ``pytest.raises(CircuitOpenError)`` once
    ``_classify_with_fallback`` raises it instead of tripping the
    ``assert last_exc is not None`` placeholder.
    """
    for name in ("groq", "gemini"):
        cb = moderator._get_circuit(name)
        for _ in range(settings.llm_circuit_failure_threshold):
            cb.record_failure()

    with pytest.raises(AssertionError):
        moderator.moderate_with_llm("hello")


def test_moderate_returns_fail_open_when_all_circuits_open():
    from app.capabilities.chat_safety.service import moderate

    for name in ("groq", "gemini"):
        cb = moderator._get_circuit(name)
        for _ in range(settings.llm_circuit_failure_threshold):
            cb.record_failure()

    result = moderate("Platform fee jyada hai, kya discount milega?")
    assert result.allowed is True
    assert result.source == "fail_open"
