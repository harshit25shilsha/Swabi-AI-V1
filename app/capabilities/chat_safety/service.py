"""Chat Safety orchestrator.

Runs the two-layer pipeline:

    1. Deterministic detectors — cheap, no LLM, catches obvious patterns.
    2. LLM classifier — for context-dependent intent, via the provider
       ladder in ``.moderator``.

Then applies the policy layer to produce the final ALLOW/BLOCK decision.

Returns the public ``ModerationResponse``. The public contract is
preserved byte-for-byte; step 10 serializes it unchanged.
"""

from __future__ import annotations

import logging

from app.capabilities.chat_safety.detectors.arithmetic_digits import (
    detect_arithmetic_digits,
)
from app.capabilities.chat_safety.detectors.digit_density import (
    detect_high_digit_density,
)
from app.capabilities.chat_safety.detectors.email import detect_email
from app.capabilities.chat_safety.detectors.encoded_digits import (
    detect_encoded_digits,
)
from app.capabilities.chat_safety.detectors.normalize import normalize
from app.capabilities.chat_safety.detectors.number_words import (
    detect_number_word_sequence,
)
from app.capabilities.chat_safety.detectors.payment import detect_payment
from app.capabilities.chat_safety.detectors.phone import detect_phone
from app.capabilities.chat_safety.detectors.url import detect_url
from app.capabilities.chat_safety.moderator import moderate_with_llm
from app.capabilities.chat_safety.policies.policy import apply_policy
from app.capabilities.chat_safety.schemas import (
    Action,
    ModerationCategory,
    ModerationResponse,
)
from app.core.exceptions import CircuitOpenError
from app.observability import counters

logger = logging.getLogger(__name__)

# Order matters. Email before URL so "@" forms are classified as EMAIL.
# (Detector fn, category, confidence)
_DETERMINISTIC_CHECKS = [
    (detect_phone, ModerationCategory.PHONE_NUMBER, 0.99),
    (detect_number_word_sequence, ModerationCategory.PHONE_NUMBER, 0.80),
    (detect_encoded_digits, ModerationCategory.PHONE_NUMBER, 0.80),
    (detect_high_digit_density, ModerationCategory.PHONE_NUMBER, 0.75),
    (detect_arithmetic_digits, ModerationCategory.PHONE_NUMBER, 0.85),
    (detect_email, ModerationCategory.EMAIL, 0.99),
    (detect_payment, ModerationCategory.PAYMENT_INFORMATION, 0.97),
    (detect_url, ModerationCategory.EXTERNAL_URL, 0.95),
]


def moderate(raw_message: str) -> ModerationResponse:
    text = normalize(raw_message)

    # 1) Deterministic layer — skip the LLM when an obvious violation is found.
    for detector, category, confidence in _DETERMINISTIC_CHECKS:
        try:
            if detector(text):
                return ModerationResponse(
                    allowed=False,
                    action=Action.BLOCK,
                    category=category,
                    confidence=confidence,
                    source="deterministic",
                    provider="",
                )
        except Exception:
            # A broken detector must not take the whole request down.
            logger.exception("Detector %s raised", detector.__name__)

    # 2) LLM layer.
    try:
        llm_result, provider_name = moderate_with_llm(text)
    except CircuitOpenError:
        # Expected during a provider outage - no traceback needed.
        logger.warning("LLM circuit open; failing open")
        counters.increment("fail_open_total")
        return ModerationResponse(
            allowed=True,
            action=Action.ALLOW,
            category=None,
            confidence=0.0,
            source="fail_open",
            provider="",
        )
    except Exception:
        logger.exception("LLM moderation failed; failing open")
        counters.increment("fail_open_total")
        return ModerationResponse(
            allowed=True,
            action=Action.ALLOW,
            category=None,
            confidence=0.0,
            source="fail_open",
            provider="",
        )

    # 3) Policy layer — final decision.
    action, category, confidence = apply_policy(llm_result)
    return ModerationResponse(
        allowed=(action == Action.ALLOW),
        action=action,
        category=category,
        confidence=confidence,
        source="llm",
        provider=provider_name,
    )
