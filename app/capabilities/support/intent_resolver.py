"""Combine deterministic and LLM intent classification.

Order:
  1. Deterministic classifier runs first. Fast, cheap, deterministic.
     - Hit -> use it, skip the LLM.
     - Scope mismatch -> preserve the reason so the service can produce
       the right message. Do NOT consult the LLM.
     - No match -> consult the LLM.
  2. LLM classifier runs only on "no match". Returns a structured intent
     with confidence. On LLM failure or low confidence, fall through to
     NONE (RAG path).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from app.capabilities.support.intent_llm import LLMIntentClassifier
from app.capabilities.support.live_data_classifier import (
    LiveDataIntent,
)
from app.capabilities.support.live_data_classifier import (
    classify as classify_deterministic,
)

log = logging.getLogger(__name__)


@dataclass
class ResolvedIntent:
    intent: LiveDataIntent
    entities: dict[str, Any] = field(default_factory=dict)
    source: str = "deterministic"  # "deterministic" | "llm" | "none"
    reason: str = ""


class IntentResolver:
    def __init__(self, llm_classifier: LLMIntentClassifier | None = None) -> None:
        self.llm = llm_classifier or LLMIntentClassifier()

    def resolve(self, question: str, audience: str) -> ResolvedIntent:
        det = classify_deterministic(question, audience)

        if det.intent != LiveDataIntent.NONE:
            return ResolvedIntent(
                intent=det.intent,
                entities=det.entities,
                source="deterministic",
                reason=det.reason,
            )

        # Deterministic returned NONE. Was it a scope block?
        if det.reason and "scope" in det.reason.lower():
            log.info(
                "intent_resolver_scope_block",
                extra={"audience": audience, "reason": det.reason},
            )
            return ResolvedIntent(
                intent=LiveDataIntent.NONE,
                source="none",
                reason=det.reason,  # preserve for the service
            )

        # No pattern matched -> ask the LLM.
        llm_result = self.llm.classify(question, audience)
        if llm_result is None or llm_result.intent == LiveDataIntent.NONE:
            return ResolvedIntent(
                intent=LiveDataIntent.NONE,
                source="llm" if llm_result else "none",
                reason="llm_none" if llm_result else "llm_unavailable",
            )

        log.info(
            "intent_resolver_llm_hit",
            extra={
                "audience": audience,
                "intent": llm_result.intent.value,
                "confidence": llm_result.confidence,
            },
        )
        return ResolvedIntent(
            intent=llm_result.intent,
            entities=llm_result.entities,
            source="llm",
            reason=f"llm_conf={llm_result.confidence:.2f}",
        )
