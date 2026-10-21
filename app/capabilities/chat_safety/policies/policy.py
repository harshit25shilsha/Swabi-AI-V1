"""Chat Safety policy layer.

Separates "what was detected" from "how we respond". The LLM and the
detectors report evidence; this module applies the fixed policy that
converts evidence into an ALLOW/BLOCK decision.
"""

from __future__ import annotations

from app.capabilities.chat_safety.schemas import (
    Action,
    ModerationCategory,
)

BLOCKED_CATEGORIES: set[ModerationCategory] = {
    ModerationCategory.PHONE_NUMBER,
    ModerationCategory.EMAIL,
    ModerationCategory.SOCIAL_CONTACT,
    ModerationCategory.EXTERNAL_URL,
    ModerationCategory.PAYMENT_INFORMATION,
    ModerationCategory.OFF_PLATFORM_COMMUNICATION,
    ModerationCategory.OFF_PLATFORM_TRANSACTION,
    ModerationCategory.OTHER_PROHIBITED_CONTENT,
}


def apply_policy(llm_result):
    if llm_result.decision == Action.BLOCK and llm_result.category in BLOCKED_CATEGORIES:
        return Action.BLOCK, llm_result.category, llm_result.confidence
    if llm_result.decision == Action.REVIEW:
        return Action.REVIEW, llm_result.category, llm_result.confidence
    return Action.ALLOW, None, llm_result.confidence
