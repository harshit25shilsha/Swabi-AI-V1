"""Resolve support intent semantically while keeping deterministic scope guards."""

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
from app.capabilities.support.schemas import (
    MAX_SUPPORT_LIST_ITEMS,
    ConversationTurn,
)

log = logging.getLogger(__name__)

_PERSONAL_DATA_SUBJECT = re.compile(
    r"\b(bookings?|trips?|payments?|transactions?|bids?|enquir(?:y|ies)|"
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
_DIRECT_PACKAGE_SEARCH = re.compile(
    r"\b(?:show|list|provide|find|search|browse|get|recommend|need|want)\b"
    r".{0,80}\b(?:packages?|tours?)\b"
    r"|\b(?:packages?|tours?)\b.{0,80}\b"
    r"(?:available|bookable|show|list|find|search|browse)\b",
    re.IGNORECASE,
)
_PACKAGE_FILTER_HINT = re.compile(
    r"\b(?:for|during|around|near|under|between|winter|summer|spring|autumn|"
    r"fall|january|february|march|april|may|june|july|august|september|"
    r"october|november|december|budget|price|family|children|travelers?)\b",
    re.IGNORECASE,
)
_PACKAGE_LOCATION_HINT = re.compile(
    r"\b(?:in|to|near|around|for)\s+([a-z][a-z -]{1,35})",
    re.IGNORECASE,
)
_NON_LOCATION_TERMS = {"the", "this", "that", "swabi", "platform"}
_POPULAR_DESTINATIONS = re.compile(r"\b(popular|trending)\s+destinations?\b", re.IGNORECASE)


_LIST_INTENTS = {
    LiveDataIntent.MY_BOOKINGS,
    LiveDataIntent.MY_PAYMENTS,
    LiveDataIntent.MY_ENQUIRY_BIDS,
    LiveDataIntent.MY_COIN_HISTORY,
    LiveDataIntent.MY_ENQUIRIES,
    LiveDataIntent.PACKAGE_SEARCH,
    LiveDataIntent.ACTIVITY_SEARCH,
    LiveDataIntent.POPULAR_DESTINATIONS,
    LiveDataIntent.VENDOR_TRANSACTIONS,
    LiveDataIntent.VENDOR_WITHDRAWALS,
    LiveDataIntent.VENDOR_INQUIRIES,
    LiveDataIntent.VENDOR_BOOKINGS,
    LiveDataIntent.VENDOR_BIDS,
}

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


@dataclass
class ResolvedIntent:
    intent: LiveDataIntent
    entities: dict[str, Any] = field(default_factory=dict)
    source: str = "deterministic"  # "deterministic" | "llm" | "none"
    reason: str = ""
    use_rag: bool = False
    requested_limit: int | None = None


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
        knowledge_requested = is_general_process_question(question)
        live_data_requested = is_live_data_question(question)

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

        if knowledge_requested and not live_data_requested:
            return ResolvedIntent(
                intent=LiveDataIntent.NONE,
                source="none",
                reason="general_process_question",
            )

        # Exact detail lookups are a safe fast path; list/search requests are
        # classified semantically so limits and contextual filters are retained.
        if det.intent not in _LIST_INTENTS and det.intent != LiveDataIntent.NONE:
            return ResolvedIntent(
                intent=det.intent,
                entities=det.entities,
                source="deterministic",
                reason=det.reason,
                use_rag=knowledge_requested,
            )

        if (
            det.intent == LiveDataIntent.MY_PAYMENTS
            and det.reason == "customer transaction history"
            and not conversation_history
        ):
            return ResolvedIntent(
                intent=det.intent,
                entities=det.entities,
                source="deterministic",
                reason=det.reason,
                use_rag=knowledge_requested,
                requested_limit=_extract_requested_limit(question),
            )

        location_matches = [
            match.group(1).strip().lower().split()[0]
            for match in _PACKAGE_LOCATION_HINT.finditer(question)
        ]
        has_unresolved_location = any(
            location not in _NON_LOCATION_TERMS for location in location_matches
        )
        if (
            det.intent == LiveDataIntent.NONE
            and not conversation_history
            and not knowledge_requested
            and _DIRECT_PACKAGE_SEARCH.search(question)
            and not has_unresolved_location
            and not _PACKAGE_FILTER_HINT.search(question)
        ):
            return ResolvedIntent(
                intent=LiveDataIntent.PACKAGE_SEARCH,
                entities={"query": ""},
                source="deterministic",
                reason="unfiltered_package_search_fast_path",
                requested_limit=_extract_requested_limit(question),
            )

        candidate = (
            det.intent in _LIST_INTENTS
            or _may_need_live_intent_resolution(question)
            or bool(conversation_history)
        )
        if not candidate:
            return ResolvedIntent(
                intent=LiveDataIntent.NONE,
                source="none",
                reason="not_live_intent_candidate",
            )

        llm_result = self.llm.classify(
            question,
            audience,
            conversation_history=conversation_history,
        )
        if llm_result is not None:
            if llm_result.intent == LiveDataIntent.NONE:
                return ResolvedIntent(
                    intent=LiveDataIntent.NONE,
                    source="llm",
                    reason="llm_none",
                    use_rag=knowledge_requested,
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
                use_rag=knowledge_requested,
                requested_limit=_requested_limit_from_result(
                    llm_result,
                    question,
                ),
            )

        if det.intent != LiveDataIntent.NONE:
            return ResolvedIntent(
                intent=det.intent,
                entities=det.entities,
                source="deterministic",
                reason=det.reason or "deterministic_fallback",
                use_rag=knowledge_requested,
                requested_limit=_extract_requested_limit(question),
            )

        if _DIRECT_PACKAGE_SEARCH.search(question) and not has_unresolved_location:
            return ResolvedIntent(
                intent=LiveDataIntent.PACKAGE_SEARCH,
                source="deterministic",
                reason="catalogue_search_fallback",
                requested_limit=_extract_requested_limit(question),
            )

        return ResolvedIntent(
            intent=LiveDataIntent.NONE,
            source="none",
            reason="llm_unavailable",
        )
