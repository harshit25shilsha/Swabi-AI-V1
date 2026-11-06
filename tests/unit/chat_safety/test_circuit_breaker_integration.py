"""Chat Safety + circuit breaker integration tests.

The state-machine tests live in tests/unit/test_circuit_breaker.py.
This file covers the interaction between the moderator's provider ladder
and the shared circuit registry.
"""

from __future__ import annotations

import pytest

from app.capabilities.chat_safety import moderator
from app.core.config import settings
from app.core.exceptions import CircuitOpenError


@pytest.fixture(autouse=True)
def _reset_circuits():
    """Every test starts with a clean provider circuit registry."""
    moderator._provider_circuits.clear()
    yield
    moderator._provider_circuits.clear()


def _open_all_circuits() -> None:
    for name in ("groq", "gemini"):
        cb = moderator._get_circuit(name)
        for _ in range(settings.llm_circuit_failure_threshold):
            cb.record_failure()


def test_moderate_with_llm_raises_when_all_circuits_open():
    """All circuits open-> moderate_with_llm re-raises CircuitOpenError."""
    _open_all_circuits()

    with pytest.raises(CircuitOpenError):
        moderator.moderate_with_llm("hello")


def test_moderate_returns_fail_open_when_all_circuits_open():
    from app.capabilities.chat_safety.service import moderate

    _open_all_circuits()

    result = moderate("Platform fee jyada hai, kya discount milega?")
    assert result.allowed is True
    assert result.source == "fail_open"
