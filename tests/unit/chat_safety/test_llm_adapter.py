"""Tests for the Chat Safety LLM adapter."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.capabilities.chat_safety import llm_adapter
from app.capabilities.chat_safety.llm_adapter import (
    ChatSafetyLLMAdapter,
    get_adapter,
)
from app.core.config import settings
from app.llm.schemas import LLMResponse, TokenUsage


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Ensure each test starts with a fresh module-level adapter."""
    llm_adapter._adapter = None
    yield
    llm_adapter._adapter = None


def _response(content: str) -> LLMResponse:
    return LLMResponse(
        content=content,
        model="test-model",
        usage=TokenUsage(total_tokens=1),
        latency_ms=1.0,
    )


# ---------------------------------------------------------------------------
# available_providers
# ---------------------------------------------------------------------------


def test_available_providers_groq_only_when_fallback_disabled(monkeypatch):
    monkeypatch.setattr(settings, "llm_fallback_enabled", False)
    monkeypatch.setattr(settings, "gemini_api_key", "some-key")
    a = ChatSafetyLLMAdapter()
    assert a.available_providers() == ["groq"]


def test_available_providers_groq_only_when_no_gemini_key(monkeypatch):
    monkeypatch.setattr(settings, "llm_fallback_enabled", True)
    monkeypatch.setattr(settings, "gemini_api_key", "")
    a = ChatSafetyLLMAdapter()
    assert a.available_providers() == ["groq"]


def test_available_providers_includes_gemini(monkeypatch):
    monkeypatch.setattr(settings, "llm_fallback_enabled", True)
    monkeypatch.setattr(settings, "gemini_api_key", "some-key")
    a = ChatSafetyLLMAdapter()
    assert a.available_providers() == ["groq", "gemini"]


# ---------------------------------------------------------------------------
# classify dispatch
# ---------------------------------------------------------------------------


def test_classify_unknown_provider_raises():
    a = ChatSafetyLLMAdapter()
    with pytest.raises(ValueError, match="unknown provider"):
        a.classify("openai", "sys", "user")


def test_classify_groq_calls_gateway(monkeypatch):
    gateway = MagicMock()
    gateway.generate_sync.return_value = _response('{"decision":"ALLOW"}')
    a = ChatSafetyLLMAdapter(gateway=gateway)

    result = a.classify("groq", "system prompt", "hello")

    gateway.generate_sync.assert_called_once()
    call_kwargs = gateway.generate_sync.call_args.kwargs
    assert call_kwargs["system"] == "system prompt"
    assert call_kwargs["json_mode"] is True
    assert result == '{"decision":"ALLOW"}'


def test_classify_groq_propagates_gateway_errors(monkeypatch):
    from app.core.exceptions import AIProviderError

    gateway = MagicMock()
    gateway.generate_sync.side_effect = AIProviderError("boom")
    a = ChatSafetyLLMAdapter(gateway=gateway)

    with pytest.raises(AIProviderError):
        a.classify("groq", "sys", "user")


def test_classify_gemini_calls_sdk(monkeypatch):
    monkeypatch.setattr(settings, "gemini_model", "gemini-test")
    monkeypatch.setattr(settings, "gemini_api_key", "fake-key")

    fake_response = MagicMock()
    fake_response.text = '{"decision":"BLOCK","category":"PHONE_NUMBER"}'
    fake_client = MagicMock()
    fake_client.models.generate_content.return_value = fake_response

    a = ChatSafetyLLMAdapter()
    a._gemini_client = fake_client  # pre-seed to bypass real SDK construction

    result = a.classify("gemini", "sys", "user")

    assert result == '{"decision":"BLOCK","category":"PHONE_NUMBER"}'
    fake_client.models.generate_content.assert_called_once()
    assert fake_client.models.generate_content.call_args.kwargs["model"] == "gemini-test"


def test_classify_gemini_empty_response_returns_empty_json(monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "fake-key")

    fake_response = MagicMock()
    fake_response.text = None
    fake_client = MagicMock()
    fake_client.models.generate_content.return_value = fake_response

    a = ChatSafetyLLMAdapter()
    a._gemini_client = fake_client

    assert a.classify("gemini", "sys", "user") == "{}"


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------


def test_get_adapter_returns_same_instance():
    a1 = get_adapter()
    a2 = get_adapter()
    assert a1 is a2


def test_get_adapter_constructs_lazily():
    assert llm_adapter._adapter is None
    a = get_adapter()
    assert llm_adapter._adapter is a
