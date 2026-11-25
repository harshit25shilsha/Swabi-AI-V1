"""Resolve support intent semantically while keeping deterministic scope guards."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from app.capabilities.support.intent_llm import LLMIntentClassifier
from app.capabilities.support.live_data_classifier import (
    LiveDataIntent,
)
from app.capabilities.support.live_data_classifier import (
    classify as classify_deterministic,
)
from app.capabilities.support.schemas import (
    MAX_SUPPORT_LIST_ITEMS,
    ConversationTurn,
)

log = logging.getLogger(__name__)

_PACKAGE_LOCATION_HINT = re.compile(
    r"\b(?:in|to|near|around)\s+([a-z][a-z -]{1,35})",
    re.IGNORECASE,
)
_PACKAGE_BASED_LOCATION = re.compile(
    r"\b(?:packages?|tours?)\s+([a-z][a-z -]{1,35}?)\s+based\b",
    re.IGNORECASE,
)
_LOCATION_BASED_PACKAGE_SEARCH = re.compile(
    r"\b(?:provide|show|list|find|search|browse|get|recommend|need|want)\s+"
    r"(?:me\s+)?(?:only\s+)?(?:\d+\s+)?"
    r"(?P<location>[a-z][a-z -]{1,35}?)\s+based\s+(?:packages?|tours?)\b",
    re.IGNORECASE,
)
_PACKAGE_LOCATION_END = re.compile(
    r"\b(?:based|during|under|below|between|winter|summer|spring|autumn|fall|"
    r"january|february|march|april|may|june|july|august|september|october|"
    r"november|december|budget|price|family|children|travelers?|travellers?)\b",
    re.IGNORECASE,
)
_NON_LOCATION_TERMS = {"the", "this", "that", "swabi", "platform"}

_COUNT_WORDS = {
    "one": 1,
    "a": 1,
    "an": 1,
    "two": 2,
    "couple": 2,
    "pair": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}
_RESULT_NOUNS = re.compile(
    r"\b(packages?|tours?|payments?|bookings?|trips?|bids?|transactions?|"
    r"activities?|destinations?|enquir(?:y|ies)|withdrawals?|results?|options?|items?)\b",
    re.IGNORECASE,
)
_COUNT_TOKEN = re.compile(r"\b(?P<count>\d{1,2}|[a-z]+)\b", re.IGNORECASE)


def _extract_requested_limit(question: str) -> int | None:
    """Extract a plainly stated count as a safe fallback if the LLM is unavailable."""
    for match in _COUNT_TOKEN.finditer(question):
        prefix = question[max(0, match.start() - 24) : match.start()]
        suffix = question[match.end() : match.end() + 40]
        if re.search(r"\b(?:under|below|less than|budget|price|cost)\s*$", prefix, re.I):
            continue
        if re.match(r"\s+(?:couple|pair)\b", suffix, re.I):
            continue
        if re.match(
            r"\s*(?:people|persons|travelers?|travellers?|adults?|children|days?|nights?)\b",
            suffix,
            re.I,
        ):
            continue
        if not _RESULT_NOUNS.search(suffix):
            continue
        token = match.group("count").lower()
        try:
            count = int(token)
        except ValueError:
            count = _COUNT_WORDS.get(token, 0)
        if 1 <= count <= MAX_SUPPORT_LIST_ITEMS:
            return count
    return None


def _requested_limit_from_result(
    result: Any,
    question: str,
) -> int | None:
    limit = getattr(result, "requested_limit", None)
    if limit is None:
        limit = result.entities.get("limit")
    if isinstance(limit, int) and not isinstance(limit, bool):
        if 1 <= limit <= MAX_SUPPORT_LIST_ITEMS:
            return limit
    elif isinstance(limit, str) and limit.isdecimal():
        parsed = int(limit)
        if 1 <= parsed <= MAX_SUPPORT_LIST_ITEMS:
            return parsed
    return _extract_requested_limit(question)


def _extract_package_location(question: str) -> str | None:
    """Extract an explicitly stated package destination without LLM guessing."""
    location_before_packages = _LOCATION_BASED_PACKAGE_SEARCH.search(question)
    if location_before_packages:
        return location_before_packages.group("location").strip()

    based_match = _PACKAGE_BASED_LOCATION.search(question)
    if based_match:
        return based_match.group(1).strip()

    for match in _PACKAGE_LOCATION_HINT.finditer(question):
        candidate = _PACKAGE_LOCATION_END.split(match.group(1), maxsplit=1)[0].strip()
        if not candidate:
            continue
        first_term = candidate.casefold().split()[0]
        if first_term not in _NON_LOCATION_TERMS and first_term not in {
            "a",
            "an",
            "all",
            "any",
            "available",
        }:
            return candidate
    return None


@dataclass
class ResolvedIntent:
    intent: LiveDataIntent
    entities: dict[str, Any] = field(default_factory=dict)
    source: str = "deterministic"  # "deterministic" | "llm" | "none"
    reason: str = ""
    use_rag: bool = False
    requested_limit: int | None = None
    knowledge_query: str | None = None


class IntentResolver:
    def __init__(self, llm_classifier: LLMIntentClassifier | None = None) -> None:
        self.llm = llm_classifier or LLMIntentClassifier()

    def resolve(
        self,
        question: str,
        audience: str,
        conversation_history: list[ConversationTurn] | None = None,
    ) -> ResolvedIntent:
        det = classify_deterministic(question, audience)

        # Enforce deterministic scope blocks before semantic planning.
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

        package_location = _extract_package_location(question)
        llm_result = self.llm.classify(
            question,
            audience,
            conversation_history=conversation_history,
        )
        if llm_result is None:
            return ResolvedIntent(
                intent=LiveDataIntent.NONE,
                source="none",
                reason="llm_unavailable",
                use_rag=True,
            )

        knowledge_query = (
            llm_result.knowledge_query.strip()
            if llm_result.requires_knowledge and llm_result.knowledge_query
            else None
        )
        if llm_result.intent == LiveDataIntent.NONE:
            return ResolvedIntent(
                intent=LiveDataIntent.NONE,
                source="llm",
                reason="llm_none",
                use_rag=llm_result.requires_knowledge,
                knowledge_query=knowledge_query,
            )

        entities = dict(llm_result.entities)
        if llm_result.intent == LiveDataIntent.PACKAGE_SEARCH and package_location:
            entities["query"] = package_location

        log.info(
            "intent_resolver_llm_hit",
            extra={
                "audience": audience,
                "intent": llm_result.intent.value,
                "confidence": llm_result.confidence,
                "requires_knowledge": llm_result.requires_knowledge,
            },
        )
        return ResolvedIntent(
            intent=llm_result.intent,
            entities=entities,
            source="llm",
            reason=f"llm_conf={llm_result.confidence:.2f}",
            use_rag=llm_result.requires_knowledge,
            requested_limit=_requested_limit_from_result(
                llm_result,
                question,
            ),
            knowledge_query=knowledge_query,
        )
