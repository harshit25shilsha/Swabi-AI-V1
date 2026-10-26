"""LLM-based intent classifier for support questions.

Used as a fallback when the deterministic classifier does not match a
question. Returns a structured intent + entities, or None on any failure.

The LLM classifier does NOT enforce audience scope. It returns the intent
that best matches the question, even if the audience is wrong. The service
layer enforces scope: customer-only intents requested by a vendor (or vice
versa) return a scope-mismatch message instead of executing the fetch.

This split is deliberate. The LLM's only job is "what does the user want?"
The service's job is "are they allowed to have it?"
"""
from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from app.capabilities.support.live_data_classifier import LiveDataIntent
from app.core.exceptions import AIProviderError, AIValidationError
from app.llm.gateway import LLMGateway

log = logging.getLogger(__name__)

MIN_CONFIDENCE = 0.7

_SYSTEM_PROMPT = """You classify customer or vendor support questions about Swabi, a travel bidding marketplace, into live-data intents.

Return ONLY a JSON object with three fields:
- "intent": one of the intent names below, or "none"
- "entities": an object with any IDs, destinations, or names you extracted (may be empty)
- "confidence": a number between 0.0 and 1.0

Do NOT answer the question. Do NOT explain. Return JSON only.

CUSTOMER INTENTS (only meaningful when audience is "customer"):
- my_bookings: the user wants to see their own bookings or trips
- my_payments: the user wants their payment history
- my_coins: the user wants their Swabi coin balance
- my_coin_history: the user wants their Swabi coin transaction history
- my_enquiries: the user wants their posted enquiries or trip requests

VENDOR INTENTS (only meaningful when audience is "vendor"):
- vendor_wallet: the user wants their vendor wallet balance
- vendor_transactions: the user wants their vendor wallet transaction ledger
- vendor_withdrawals: the user wants their withdrawal requests
- vendor_inquiries: the user wants enquiries they can bid on
- vendor_bookings: the user wants their vendor-side bookings
- vendor_bids: the user wants the bids they placed
- vendor_subscription: the user wants their subscription / plan status

Use intent "none" when:
- The question is about platform policy, cancellation, refunds, escrow, commissions, or any procedural knowledge.
- The question is general travel knowledge (destinations, activities, tips).
- The question is clearly out of scope for Swabi.
- The question does not need user-specific or current data.

The "audience" is provided. Use it to inform which intents to consider, but if the question is clearly the wrong scope (a customer asking about vendor wallets), still return the matching intent. The service layer handles the scope mismatch.

Examples:

Audience: customer
Question: "What is the status of my last booked trip?"
Output: {"intent":"my_bookings","entities":{},"confidence":0.95}

Audience: vendor
Question: "How much can I withdraw from my wallet?"
Output: {"intent":"vendor_wallet","entities":{},"confidence":0.9}

Audience: customer
Question: "What is the cancellation policy for October?"
Output: {"intent":"none","entities":{},"confidence":0.95}

Audience: customer
Question: "Show me my vendor wallet balance"
Output: {"intent":"vendor_wallet","entities":{},"confidence":0.85}

Audience: customer
Question: "Which packages have I purchased recently?"
Output: {"intent":"my_bookings","entities":{},"confidence":0.92}

Audience: vendor
Question: "What are the new enquiries in Kashmir?"
Output: {"intent":"vendor_inquiries","entities":{"destination":"Kashmir"},"confidence":0.88}
"""


class LLMIntentResult(BaseModel):
    intent: LiveDataIntent
    entities: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0)


class LLMIntentClassifier:
    """Structured-output wrapper around LLMGateway.

    Never raises on provider failure: returns None so the caller can fall
    back to the deterministic result (NONE -> RAG path).
    """

    def __init__(self, gateway: LLMGateway | None = None) -> None:
        self.gateway = gateway or LLMGateway()

    def classify(
        self,
        question: str,
        audience: str,
    ) -> LLMIntentResult | None:
        user_message = f"Audience: {audience}\nQuestion: {question}"
        try:
            result = self.gateway.generate_structured_sync(
                prompt=user_message,
                response_model=LLMIntentResult,
                system=_SYSTEM_PROMPT,
                temperature=0.0,
                max_tokens=500,
            )
        except (AIProviderError, AIValidationError) as exc:
            log.warning("intent_llm_failed: %s", exc)
            return None
        except Exception as exc:  # noqa: BLE001
            log.exception("intent_llm_unexpected: %s", exc)
            return None

        if result.confidence < MIN_CONFIDENCE:
            log.info(
                "intent_llm_low_confidence",
                extra={
                    "intent": result.intent.value,
                    "confidence": result.confidence,
                    "threshold": MIN_CONFIDENCE,
                },
            )
            return None

        return result
