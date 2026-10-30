"""Combine deterministic and LLM intent classification.

Order:
  1. Deterministic classifier runs first. Fast, cheap, deterministic.
     - Hit -> use it, skip the LLM.
     - Scope mismatch -> preserve the reason so the service can produce
       the right message. Do NOT consult the LLM.
     - No match -> consult the LLM only for likely account/live-data or
       public-catalogue requests; ordinary knowledge questions go to RAG.
  2. LLM classifier runs only on "no match". Returns a structured intent
     with confidence. On LLM failure or low confidence, fall through to
     NONE (RAG path).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from app.capabilities.support.intent_llm import LLMIntentClassifier
from app.capabilities.support.live_data_check import (
    is_general_process_question,
    is_live_data_question,
)
from app.capabilities.support.live_data_classifier import (
    LiveDataIntent,
)
from app.capabilities.support.live_data_classifier import (
    classify as classify_deterministic,
)

log = logging.getLogger(__name__)

_PERSONAL_DATA_SUBJECT = re.compile(
    r"\b(bookings?|trips?|payments?|bids?|enquir(?:y|ies)|"
    r"coins?|wallet|withdrawals?|subscription|plan)\b",
    re.IGNORECASE,
)
_PERSONAL_REFERENCE = re.compile(r"\b(my|mine|i|me)\b", re.IGNORECASE)
_CATALOGUE_REQUEST = re.compile(
    r"\b(search|browse|find|show|list|explore|recommend|get|provide|want|have|what|which|"
    r"any|available|popular|trending)\b"
    r".{0,60}\b(packages?|activities|tours?|destinations?)\b"
    r"|\b(packages?|activities|tours?|destinations?)\b"
    r".{0,60}\b(search|browse|find|show|list|explore|recommend|get|want|have|"
    r"provide|available|popular|trending)\b"
    r"|\b(packages?|tours?)\b.{0,80}\b"
    r"(?:for|in|during|around|near|under|between|winter|summer|spring|autumn|"
    r"fall|january|february|march|april|may|june|july|august|september|"
    r"october|november|december|budget|price|family|children|travelers?)\b",
    re.IGNORECASE,
)
_POPULAR_DESTINATIONS = re.compile(
    r"\b(popular|trending)\s+destinations?\b", re.IGNORECASE
)
_DIRECT_PACKAGE_SEARCH = re.compile(
    r"\b(?:show|list|provide|find|search|browse|get|recommend|need|want)\b"
    r".{0,80}\b(?:packages?|tours?)\b"
    r"|\b(?:packages?|tours?)\b.{0,80}\b"
    r"(?:available|bookable|show|list|find|search|browse)\b"
    r"|\b(?:packages?|tours?)\b.{0,80}\b"
    r"(?:for|in|during|around|near|under|between|winter|summer|spring|"
    r"autumn|fall|january|february|march|april|may|june|july|august|"
    r"september|october|november|december|budget|price|family|children|"
    r"travelers?)\b",
    re.IGNORECASE,
)
_PACKAGE_DESTINATION_HINT = re.compile(
    r"\b(?:in|to|near|around)\s+([a-z][a-z -]{1,35})",
    re.IGNORECASE,
)
_NON_DESTINATION_HINTS = {
    "winter", "summer", "spring", "autumn", "fall",
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    "the", "swabi",
}


def _may_need_live_intent_resolution(question: str) -> bool:
    """Avoid LLM classification for questions clearly handled by RAG."""
    return (
        is_live_data_question(question)
        or (
            _PERSONAL_REFERENCE.search(question) is not None
            and _PERSONAL_DATA_SUBJECT.search(question) is not None
        )
        or _CATALOGUE_REQUEST.search(question) is not None
        or _POPULAR_DESTINATIONS.search(question) is not None
    )


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

        if (
            is_general_process_question(question)
            and not is_live_data_question(question)
        ):
            return ResolvedIntent(
                intent=LiveDataIntent.NONE,
                source="none",
                reason="general_process_question",
            )

        if det.intent != LiveDataIntent.NONE:
            return ResolvedIntent(
                intent=det.intent,
                entities=det.entities,
                source="deterministic",
                reason=det.reason,
            )

        direct_package_search = (
            _DIRECT_PACKAGE_SEARCH.search(question) is not None
        )
        needs_entity_extraction = any(
            match.group(1).strip().lower().split()[0]
            not in _NON_DESTINATION_HINTS
            and not match.group(1).strip().lower().startswith("swabi platform")
            for match in _PACKAGE_DESTINATION_HINT.finditer(question)
        )

        if direct_package_search and not needs_entity_extraction:
            return ResolvedIntent(
                intent=LiveDataIntent.PACKAGE_SEARCH,
                source="deterministic",
                reason="public_package_search",
            )

        if not _may_need_live_intent_resolution(question) and not direct_package_search:
            return ResolvedIntent(
                intent=LiveDataIntent.NONE,
                source="none",
                reason="not_live_intent_candidate",
            )

        # A plausible live-data or catalogue request did not match a
        # deterministic pattern; use the LLM to resolve the ambiguity.
        llm_result = self.llm.classify(question, audience)
        if llm_result is not None and llm_result.intent != LiveDataIntent.NONE:
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

        if direct_package_search:
            return ResolvedIntent(
                intent=LiveDataIntent.PACKAGE_SEARCH,
                source="deterministic",
                reason="public_package_search",
            )

        return ResolvedIntent(
            intent=LiveDataIntent.NONE,
            source="llm" if llm_result else "none",
            reason="llm_none" if llm_result else "llm_unavailable",
        )
