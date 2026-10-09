"""Domain Exceptions used across the AI service."""

from __future__ import annotations


# Foundation
class SwabiAIError(Exception):
    """Base class for all AI service exceptions."""


class ConfigurationError(SwabiAIError):
    """Invalid or missing configuration settings."""


class AIProviderError(SwabiAIError):
    """LLM provider returned an error response."""


class AIValidationError(SwabiAIError):
    """LLM structured output failed validation."""


class AITimeoutError(SwabiAIError):
    """LLM provider request timed out."""


class SwabiAPIError(SwabiAIError):
    """Swabi backend returned an error response."""


class SwabiAuthenticationError(SwabiAPIError):
    """Swabi backend authentication failed."""


class ToolExecutionError(SwabiAIError):
    """A Tool failed during execution."""


# Guardrail


class GuardrailError(SwabiAIError):
    """Base class for Guardrail-related exceptions."""


class GuardrailUnavailableError(GuardrailError):
    """The Guardrail could not produce a decision."""


class GuardrailValidationError(GuardrailError):
    """The guardrail received an invalid request payload."""


class CircuitOpenError(GuardrailUnavailableError):
    """All provider circuits are open; no call was attempted."""
