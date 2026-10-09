"""Tests for the extended exception hierarchy."""

from __future__ import annotations

import pytest

from app.core.exceptions import (
    AIProviderError,
    AITimeoutError,
    AIValidationError,
    CircuitOpenError,
    ConfigurationError,
    GuardrailError,
    GuardrailUnavailableError,
    GuardrailValidationError,
    SwabiAIError,
    SwabiAPIError,
    SwabiAuthenticationError,
    ToolExecutionError,
)

# ---------------------------------------------------------------------------
# Root of the hierarchy
# ---------------------------------------------------------------------------


def test_every_exception_derives_from_swabi_ai_error():
    """Single catch-all works for every error in the codebase."""
    for cls in (
        ConfigurationError,
        AIProviderError,
        AIValidationError,
        AITimeoutError,
        SwabiAPIError,
        SwabiAuthenticationError,
        ToolExecutionError,
        GuardrailError,
        GuardrailUnavailableError,
        GuardrailValidationError,
        CircuitOpenError,
    ):
        assert issubclass(cls, SwabiAIError), cls.__name__


# ---------------------------------------------------------------------------
# Foundation
# ---------------------------------------------------------------------------


def test_swabi_auth_error_is_a_swabi_api_error():
    assert issubclass(SwabiAuthenticationError, SwabiAPIError)
    assert issubclass(SwabiAuthenticationError, SwabiAIError)


def test_foundation_types_are_independent_siblings():
    """No accidental cross-hierarchy between foundation errors."""
    assert not issubclass(AIProviderError, SwabiAPIError)
    assert not issubclass(AITimeoutError, AIProviderError)
    assert not issubclass(AIValidationError, AIProviderError)
    assert not issubclass(ToolExecutionError, AIProviderError)


# ---------------------------------------------------------------------------
# Guardrail
# ---------------------------------------------------------------------------


def test_guardrail_errors_derive_from_guardrail_error():
    assert issubclass(GuardrailUnavailableError, GuardrailError)
    assert issubclass(GuardrailValidationError, GuardrailError)


def test_circuit_open_is_unavailable():
    """The specific signal is caught by the general handler."""
    assert issubclass(CircuitOpenError, GuardrailUnavailableError)
    assert issubclass(CircuitOpenError, GuardrailError)
    assert issubclass(CircuitOpenError, SwabiAIError)


def test_circuit_open_caught_by_unavailable_clause():
    with pytest.raises(GuardrailUnavailableError):
        raise CircuitOpenError("all circuits open")


def test_circuit_open_caught_by_specific_clause():
    with pytest.raises(CircuitOpenError):
        raise CircuitOpenError("all circuits open")


def test_unavailable_not_caught_by_specific_circuit_open_clause():
    """The reverse direction must not hold: CircuitOpenError is narrower."""
    with pytest.raises(GuardrailUnavailableError):
        try:
            raise GuardrailUnavailableError("providers exhausted")
        except CircuitOpenError:  # pragma: no cover - should not match
            pytest.fail("GuardrailUnavailableError should not match CircuitOpenError")


# ---------------------------------------------------------------------------
# Guardrail errors do not cross into foundation handlers
# ---------------------------------------------------------------------------


def test_guardrail_errors_are_not_foundation_errors():
    assert not issubclass(GuardrailError, AIProviderError)
    assert not issubclass(GuardrailUnavailableError, AIProviderError)
    assert not issubclass(GuardrailValidationError, SwabiAPIError)
    assert not issubclass(CircuitOpenError, AITimeoutError)


# ---------------------------------------------------------------------------
# Message propagation
# ---------------------------------------------------------------------------


def test_message_is_preserved():
    err = CircuitOpenError("all circuits open")
    assert str(err) == "all circuits open"

    err2 = GuardrailValidationError("bad payload")
    assert str(err2) == "bad payload"


def test_can_be_raised_and_caught_cleanly():
    """A real raise/catch cycle through the base class works as expected."""
    caught: list[str] = []
    try:
        raise CircuitOpenError("circuit says no")
    except SwabiAIError as exc:
        caught.append(str(exc))
    assert caught == ["circuit says no"]
