"""
Integration tests that hit the real Groq API.
Run with:  pytest -m integration -v
Skip in CI: pytest -m "not integration"
"""

import time

import pytest
from groq import BadRequestError, RateLimitError

from app.capabilities.chat_safety.service import moderate
from app.core.config import settings

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not settings.GROQ_API_KEY,
        reason="Requires GROQ_API_KEY in .env",
    ),
]

# openai/gpt-oss-120b is a reasoning model and burns a lot of tokens per call.
# A short gap between calls keeps us under free-tier TPM limits.
_INTER_CALL_DELAY_SECONDS = 4


@pytest.fixture(autouse=True)
def _pace_requests():
    yield
    time.sleep(_INTER_CALL_DELAY_SECONDS)


def _call_with_retry(text: str, attempts: int = 3):
    last = None
    for i in range(attempts):
        try:
            return moderate(text)
        except (RateLimitError, BadRequestError) as e:
            last = e
            backoff = 15 * (i + 1)
            print(f"\n[transient] {type(e).__name__}, retry {i + 1}/{attempts} after {backoff}s")
            time.sleep(backoff)
    raise last


# No deterministic rule covers these yet and the LLM misses them intermittently.
# KNOWN_LLM_MISS = pytest.mark.xfail(
#     reason="relational encoding not covered by rules", strict=False
# )

DISGUISED_MUST_BLOCK = [
    "aath saat teen chaar, phir gyarah terah satrah",
    "pehle paanch, phir do do, uske baad nau aur chhe",
    "zero se shuru karo, teen baar chaar, phir saat",
    "ek kam das, do zyada paanch, phir aath",
    "fourteen ke baad sixteen, phir nineteen aur twenty one",
    "do sau ke baad teen, phir paanch aur saat mila dena",
    "mera number wali sequence: teen, double six, zero, nine",
    "paanch ko do baar bolo, uske baad ek, phir chaar chhe",
    "first one is seven, next is double two, then nine",
    "aath ka aadha nahi, seedha aath; phir teen teen aur ek",
    "teen score ke baad paanch, phir do aur nau",
    "the digits are hidden as words: ek do teen, phir chhe saat aath",
    "do plus do, phir teen minus one, aur end mein nine",
    "kal jo sequence batayi thi—pehle chaar, phir eleven, thirteen, seventeen",
    "start with the number after five, then two less than ten, then twelve",
    "ek room chahiye, do bed ke saath, teen raat ke liye, chaar log hain, "
    "paach bje phuchenge, das bje khayenge, gyarah bje niklenge, "
    "bara bje lunch karenge, pandrah minute late honge."
]

NORMAL_MUST_ALLOW = [
    "Platform fee jyada hai, kya discount milega?",
    "What payment methods does the platform support?",
    "Can I receive an email confirmation?",
    "Is the villa available from 10th to 12th October?",
    "What time is check-in?",
    "Can I cancel my booking?",
    "do din ka rent kitna hai?",
    "teen raat ke liye booking karni hai",
    "teen sau ya paanch sau rupaye mein ho jayega?",
    "No, I need one room with two beds for three nights",
    "ek room chahiye, do bed ke saath, teen raat ke liye",
]


@pytest.mark.parametrize("text", DISGUISED_MUST_BLOCK)
def test_llm_blocks_disguised_numbers(text):
    result = _call_with_retry(text)
    assert result.action == "BLOCK", f"Missed: {text!r} -> {result}"
    assert result.category == "PHONE_NUMBER", f"Wrong category for {text!r}: {result}"


@pytest.mark.parametrize("text", NORMAL_MUST_ALLOW)
def test_llm_allows_normal_chat(text):
    result = _call_with_retry(text)
    assert result.action == "ALLOW", f"False positive: {text!r} -> {result}"
    assert result.confidence > 0, f"LLM failed and pipeline failed open: {text!r}"
