"""Tests for the provider ladder and circuit interaction."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.capabilities.chat_safety import moderator
from app.capabilities.chat_safety.llm_adapter import ChatSafetyLLMAdapter
from app.core.config import settings
from app.core.exceptions import CircuitOpenError


@pytest.fixture(autouse=True)
def _reset_circuits():
    moderator._provider_circuits.clear()
    yield
    moderator._provider_circuits.clear()


@pytest.fixture
def mock_adapter():
    adapter = MagicMock(spec=ChatSafetyLLMAdapter)
    adapter.available_providers.return_value = ["groq", "gemini"]
    return adapter


def _patch_adapter(adapter):
    return patch(
        "app.capabilities.chat_safety.moderator.get_adapter",
        return_value=adapter,
    )


def test_falls_back_to_gemini_when_groq_fails(mock_adapter):
    def classify(name, system, user):
        if name == "groq":
            raise RuntimeError("groq down")
        return '{"decision":"BLOCK","category":"PHONE_NUMBER","confidence":0.9}'

    mock_adapter.classify.side_effect = classify

    with _patch_adapter(mock_adapter):
        result, provider = moderator._classify_with_fallback("test")

    assert provider == "gemini"
    assert result.decision == "BLOCK"


def test_raises_when_all_providers_fail(mock_adapter):
    mock_adapter.classify.side_effect = RuntimeError("all down")

    with _patch_adapter(mock_adapter):
        with pytest.raises(RuntimeError):
            moderator._classify_with_fallback("test")


def test_skips_provider_with_open_circuit(mock_adapter):
    circuit = moderator._get_circuit("groq")
    for _ in range(settings.llm_circuit_failure_threshold):
        circuit.record_failure()

    mock_adapter.classify.return_value = '{"decision":"ALLOW","category":null,"confidence":0.9}'

    with _patch_adapter(mock_adapter):
        result, provider = moderator._classify_with_fallback("test")

    assert provider == "gemini"
    called_names = [c.args[0] for c in mock_adapter.classify.call_args_list]
    assert "groq" not in called_names


def test_raises_circuit_open_when_all_circuits_open(mock_adapter):
    for name in ("groq", "gemini"):
        circuit = moderator._get_circuit(name)
        for _ in range(settings.llm_circuit_failure_threshold):
            circuit.record_failure()

    with _patch_adapter(mock_adapter):
        with pytest.raises(CircuitOpenError):
            moderator._classify_with_fallback("test")

    mock_adapter.classify.assert_not_called()


def test_raises_runtime_error_when_no_providers(mock_adapter):
    mock_adapter.available_providers.return_value = []

    with _patch_adapter(mock_adapter):
        with pytest.raises(RuntimeError, match="no LLM providers"):
            moderator._classify_with_fallback("test")


def test_increments_provider_counter_on_success(mock_adapter):
    from app.observability import counters

    counters.reset()
    mock_adapter.available_providers.return_value = ["gemini"]
    mock_adapter.classify.return_value = '{"decision":"ALLOW","category":null,"confidence":0.9}'

    with _patch_adapter(mock_adapter):
        moderator._classify_with_fallback("test")

    snap = counters.snapshot()
    assert snap["llm_calls_by_provider"].get("gemini") == 1
    assert snap["llm_calls_total"] == 1


def test_increments_failure_counter_on_error(mock_adapter):
    from app.observability import counters

    counters.reset()
    mock_adapter.available_providers.return_value = ["groq"]
    mock_adapter.classify.side_effect = RuntimeError("boom")

    with _patch_adapter(mock_adapter):
        with pytest.raises(RuntimeError):
            moderator._classify_with_fallback("test")

    snap = counters.snapshot()
    assert snap["llm_failures_by_provider"].get("groq") == 1
    assert snap["llm_failures_total"] == 1
