"""Chat Safety orchestrator.

Two-layer decision with escalation:

    1. Hard detectors (bool -> BLOCK on True). Fast, deterministic.
    2. Soft detectors (ALLOW / REVIEW / BLOCK). Aggregate signals.
    3. LLM classifier.
    4. Policy + escalation:
         - hard BLOCK -> BLOCK (LLM skipped)
         - soft BLOCK -> BLOCK
         - LLM BLOCK -> BLOCK
         - LLM ALLOW + soft REVIEW -> REVIEW  (do not drop the signal)
         - LLM ALLOW + soft ALLOW -> ALLOW
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
from app.capabilities.chat_safety.detectors.scattered_numbers import (
    detect_scattered_number_sequence,
)
from app.capabilities.chat_safety.detectors.url import detect_url
from app.capabilities.chat_safety.moderator import moderate_with_llm
from app.capabilities.chat_safety.policies.policy import apply_policy
from app.capabilities.chat_safety.schemas import (
    Action,
    GuardrailDecision,
    ModerationCategory,
    ModerationResponse,
)
from app.core.exceptions import CircuitOpenError
from app.observability import counters

logger = logging.getLogger(__name__)


# Hard detectors — True means BLOCK, LLM is skipped.

_HARD_CHECKS = [
    (detect_phone,                 ModerationCategory.PHONE_NUMBER,        0.99),
    (detect_number_word_sequence,  ModerationCategory.PHONE_NUMBER,        0.80),
    (detect_encoded_digits,        ModerationCategory.PHONE_NUMBER,        0.80),
    (detect_high_digit_density,    ModerationCategory.PHONE_NUMBER,        0.75),
    (detect_arithmetic_digits,     ModerationCategory.PHONE_NUMBER,        0.85),
    (detect_email,                 ModerationCategory.EMAIL,               0.99),
    (detect_payment,               ModerationCategory.PAYMENT_INFORMATION, 0.97),
    (detect_url,                   ModerationCategory.EXTERNAL_URL,        0.95),
]


# Soft detectors — return GuardrailDecision (ALLOW / REVIEW / BLOCK).

_SOFT_CHECKS = [
    (detect_scattered_number_sequence, ModerationCategory.PHONE_NUMBER),
]


# Response helpers

def _block(category, confidence, *, source, provider=""):
    return ModerationResponse(
        allowed=False,
        action=Action.BLOCK,
        category=category,
        confidence=confidence,
        source=source,
        provider=provider,
    )


def _allow(category, confidence, *, source, provider=""):
    return ModerationResponse(
        allowed=True,
        action=Action.ALLOW,
        category=category,
        confidence=confidence,
        source=source,
        provider=provider,
    )


def _review(category, confidence, *, provider=""):
    """Map internal REVIEW to the public contract.

    Public contract is ALLOW / BLOCK only. We surface REVIEW as a BLOCK
    with `source="review"` so the caller can distinguish a hard block
    from a soft one. If the public contract is later extended to include
    a REVIEW action, change this helper only.
    """
    return ModerationResponse(
        allowed=False,
        action=Action.BLOCK,
        category=category,
        confidence=confidence,
        source="review",
        provider=provider,
    )


# Orchestrator

def moderate(raw_message: str) -> ModerationResponse:
    text = normalize(raw_message)

    # 1. Hard deterministic layer
    for detector, category, confidence in _HARD_CHECKS:
        try:
            if detector(text):
                return _block(category, confidence, source="deterministic")
        except Exception:
            logger.exception("Detector %s raised", detector.__name__)

    # 2. Soft deterministic layer
    soft_decision = GuardrailDecision.ALLOW
    soft_category: ModerationCategory | None = None
    for detector, category in _SOFT_CHECKS:
        try:
            decision = detector(text)
        except Exception:
            logger.exception("Detector %s raised", detector.__name__)
            continue

        # Take the worst (BLOCK > REVIEW > ALLOW).
        rank = {
            GuardrailDecision.ALLOW: 0,
            GuardrailDecision.REVIEW: 1,
            GuardrailDecision.BLOCK: 2,
        }
        if rank[decision] > rank[soft_decision]:
            soft_decision = decision
            soft_category = category

    if soft_decision == GuardrailDecision.BLOCK:
        return _block(soft_category, 0.85, source="deterministic")

    # 3. LLM layer
    try:
        llm_result, provider_name = moderate_with_llm(text)
    except CircuitOpenError:
        logger.warning("LLM circuit open; evaluating fallback")
        counters.increment("fail_open_total")
        if soft_decision == GuardrailDecision.REVIEW:
            return _review(soft_category, 0.6)
        return _allow(
            None, 0.0, source="fail_open",
        )
    except Exception:
        logger.exception("LLM moderation failed; evaluating fallback")
        counters.increment("fail_open_total")
        if soft_decision == GuardrailDecision.REVIEW:
            return _review(soft_category, 0.6)
        return _allow(
            None, 0.0, source="fail_open",
        )

    # 4. Policy layer
    action, category, confidence = apply_policy(llm_result)
    if action == Action.BLOCK:
        return _block(category, confidence, source="llm", provider=provider_name)
    if action == Action.REVIEW:
        fallback_category = ModerationCategory.OTHER_PROHIBITED_CONTENT
        return _review(
            category or fallback_category,
            confidence,
            provider=provider_name,
        )
    if soft_decision == GuardrailDecision.REVIEW:
        return _review(soft_category, 0.65, provider=provider_name)
    return _allow(category, confidence, source="llm", provider=provider_name)
