from unittest.mock import patch

from app.capabilities.chat_safety.schemas import LLMResult
from app.capabilities.chat_safety.service import moderate


def test_phone_blocks_without_llm():
    with patch("app.capabilities.chat_safety.service.moderate_with_llm") as mock_llm:
        result = moderate("Call me at 9876543210")
        assert result.action == "BLOCK"
        assert result.category == "PHONE_NUMBER"
        mock_llm.assert_not_called()


def test_llm_block_off_platform():
    with patch("app.capabilities.chat_safety.service.moderate_with_llm") as mock_llm:
        mock_llm.return_value = (
            LLMResult(
                decision="BLOCK",
                category="OFF_PLATFORM_COMMUNICATION",
                confidence=0.95,
            ),
            "groq",
        )
        result = moderate("WhatsApp pe baat karte hain.")
        assert result.action == "BLOCK"
        assert result.category == "OFF_PLATFORM_COMMUNICATION"
        assert result.provider == "groq"


def test_llm_allow():
    with patch("app.capabilities.chat_safety.service.moderate_with_llm") as mock_llm:
        mock_llm.return_value = (
            LLMResult(decision="ALLOW", category=None, confidence=0.98),
            "groq",
        )
        result = moderate("Platform fee jyada hai, kya discount milega?")
        assert result.action == "ALLOW"


def test_llm_failure_fails_open():
    with patch(
        "app.capabilities.chat_safety.service.moderate_with_llm",
        side_effect=RuntimeError("groq down"),
    ):
        result = moderate("Is the villa available?")
        assert result.action == "ALLOW"
        assert result.confidence == 0.0
        assert result.source == "fail_open"


def test_encoded_digits_blocked_without_llm():
    with patch("app.capabilities.chat_safety.service.moderate_with_llm") as mock_llm:
        result = moderate("start with the number after five, then two less than ten, then twelve")
        assert result.action == "BLOCK"
        assert result.category == "PHONE_NUMBER"
        mock_llm.assert_not_called()
