from unittest.mock import MagicMock, patch

import pytest

from app.capabilities.chat_safety import moderator
from app.capabilities.chat_safety.providers import GeminiProvider, GroqProvider
from app.core.config import settings


@pytest.fixture(autouse=True)
def _reset_circuits():
    moderator._provider_circuits.clear()
    yield
    moderator._provider_circuits.clear()


def test_falls_back_to_gemini_when_groq_fails():
    groq = MagicMock(spec=GroqProvider)
    groq.name = "groq"
    groq.is_available.return_value = True
    groq.classify.side_effect = RuntimeError("groq down")

    gemini = MagicMock(spec=GeminiProvider)
    gemini.name = "gemini"
    gemini.is_available.return_value = True
    gemini.classify.return_value = '{"decision":"BLOCK","category":"PHONE_NUMBER","confidence":0.9}'

    with patch(
        "app.capabilities.chat_safety.moderator.get_providers",
        return_value=[groq, gemini],
    ):
        result, provider = moderator._classify_with_fallback("test")

    assert provider == "gemini"
    assert result.decision == "BLOCK"


def test_raises_when_all_providers_fail():
    groq = MagicMock(spec=GroqProvider)
    groq.name = "groq"
    groq.is_available.return_value = True
    groq.classify.side_effect = RuntimeError("groq down")

    gemini = MagicMock(spec=GeminiProvider)
    gemini.name = "gemini"
    gemini.is_available.return_value = True
    gemini.classify.side_effect = RuntimeError("gemini down")

    with patch(
        "app.capabilities.chat_safety.moderator.get_providers",
        return_value=[groq, gemini],
    ):
        with pytest.raises(RuntimeError):
            moderator._classify_with_fallback("test")


def test_skips_provider_with_open_circuit():
    groq = MagicMock(spec=GroqProvider)
    groq.name = "groq"
    groq.is_available.return_value = True

    gemini = MagicMock(spec=GeminiProvider)
    gemini.name = "gemini"
    gemini.is_available.return_value = True
    gemini.classify.return_value = '{"decision":"ALLOW","category":null,"confidence":0.9}'

    circuit = moderator._get_circuit("groq")
    for _ in range(settings.llm_circuit_failure_threshold):
        circuit.record_failure()

    with patch(
        "app.capabilities.chat_safety.moderator.get_providers",
        return_value=[groq, gemini],
    ):
        result, provider = moderator._classify_with_fallback("test")

    assert provider == "gemini"
    groq.classify.assert_not_called()


def test_increments_provider_counter_on_success():
    from app.observability import counters

    counters.reset()

    gemini = MagicMock(spec=GeminiProvider)
    gemini.name = "gemini"
    gemini.classify.return_value = '{"decision":"ALLOW","category":null,"confidence":0.9}'

    with patch(
        "app.capabilities.chat_safety.moderator.get_providers",
        return_value=[gemini],
    ):
        moderator._classify_with_fallback("test")

    snap = counters.snapshot()
    assert snap["llm_calls_by_provider"].get("gemini") == 1
    assert snap["llm_calls_total"] == 1
