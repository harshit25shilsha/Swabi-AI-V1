"""LLM-based intent classifier for support questions.

Used as a fallback when the deterministic classifier does not match a
question. Returns a structured intent and extracted entities, or None
when classification fails or confidence is too low.

The classifier identifies the user's goal; it does not authorize access
or call APIs. The support service checks audience and authentication,
then dispatches only to explicitly implemented fetchers.
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

_SYSTEM_PROMPT = """You classify customer or vendor support questions about Swabi,
a travel marketplace, into one supported live-data intent.

Return ONLY a JSON object with exactly these fields:
- "intent": one of the intent names listed below, or "none"
- "entities": an object containing only values explicitly stated or clearly
  implied by the question; it may be empty
- "confidence": a number between 0.0 and 1.0

Do NOT answer the question. Do NOT explain. Return JSON only.
Never invent IDs, destinations, dates, filters, or other entity values.
Never create an intent that is not listed below.

When a LIVE DATA block is present and contains the requested information,
answer from it directly. Never return INSUFFICIENT_CONTEXT in that case.
Prefer plain text over pipe-separated fields; do not rewrite delimiters
into other punctuation.

CUSTOMER INTENTS (only meaningful when audience is "customer"):
- my_bookings: list or filter the user's own bookings or trips.
  Normalize cancel/canceled/cancelled to "cancelled", complete/completed
  to "completed", and extract pending or upcoming as those canonical
  status values when the user asks for that booking status.
- my_booking_detail: details or status for one specific booking.
  Extract its UUID as "booking_id" when present. Do not use a booking code
  as a UUID.
- my_payments: the user's payment history.
  Normalize cancel/canceled/cancelled to "cancelled" and
  complete/completed to "completed". Extract pending or failed as those
  status values when requested.
- my_enquiry_bids: bids or proposals received on one of the user's enquiries.
  Extract the enquiry UUID as "enquiry_id" when present. If no enquiry is
  specified, return this intent with empty entities; the service will fetch
  bids for the user's most recently created enquiry only.
- my_coins: the user's Swabi coin balance.
- my_coin_history: the user's Swabi coin transaction history.
- my_enquiries: the user's posted enquiries or trip requests.

PUBLIC CATALOGUE INTENTS (no authentication required, any audience):
- package_search: browse the public package catalogue.
  Extract "destination" when the user names a place (e.g. "Goa", "Kashmir").
- package_detail: details of one specific package.
  Extract its UUID as "package_id" only when the UUID is explicitly in the question.
- activity_search: browse public activities.
  Extract "query" for free-text (destination, name) and/or "category"
  (one of adventure, heritage, sightseeing, cultural, wildlife).
- popular_destinations: trending/popular travel destinations.

VENDOR INTENTS (only meaningful when audience is "vendor"):
- vendor_wallet: the user's vendor wallet balance.
- vendor_transactions: the user's vendor wallet transaction ledger.
- vendor_withdrawals: the user's withdrawal requests.
- vendor_inquiries: enquiries the vendor can bid on.
  Extract "destination" when the user specifies one.
- vendor_bookings: the vendor's bookings or trips they are serving.
- vendor_bids: bids the vendor placed.
- vendor_subscription: the vendor's subscription or plan status.

Use intent "none" when:
- The question is about platform policy, cancellation, refunds, escrow,
  commissions, or procedural knowledge that does not need current account data.
- The question is general travel knowledge, such as destinations, activities,
  or travel tips.
- The question is clearly outside Swabi and travel support.
- The question does not require user-specific or current marketplace data.

AUDIENCE AND SCOPE:
The audience is provided in the user message as "customer" or "vendor".
Use it to interpret the request, but if the question clearly asks for data
from the other role, return that matching role's intent. The service layer
will enforce audience scope and must not execute a cross-role request.
The package_search intent is public catalogue data, not customer- or
vendor-private data.

ENTITY RULES:
- For booking detail, use {"booking_id": "<UUID>"} only when a UUID is present.
- For enquiry bids, use {"enquiry_id": "<UUID>"} only when a UUID is present.
- For list filters, use {"status": "<supported status>"} only when the user
  asks for that status.
- For package_search, use {"query": "<search phrase>"}; use an empty string
  for an unfiltered catalogue request.
- Do not put a whole sentence or the user's question into an entity value.

Examples:

Audience: customer
Question: "What is the status of my last booked trip?"
Output: {"intent":"my_bookings","entities":{},"confidence":0.95}

Audience: customer
Question: "Show my pending packages"
Output: {"intent":"my_bookings","entities":{"status":"pending"},"confidence":0.92}

Audience: customer
Question: "Show my cancel bookings"
Output: {"intent":"my_bookings","entities":{"status":"cancelled"},"confidence":0.92}

Audience: customer
Question: "Show my complete payments"
Output: {"intent":"my_payments","entities":{"status":"completed"},"confidence":0.92}

Audience: customer
Question: "What is the status of booking 46f46148-e88d-42b6-bbd4-d87f14009f65?"
Output: {"intent":"my_booking_detail",
         "entities":{"booking_id":"46f46148-e88d-42b6-bbd4-d87f14009f65"},
         "confidence":0.97}

Audience: customer
Question: "Who bid on my enquiry 46f46148-e88d-42b6-bbd4-d87f14009f65?"
Output: {"intent":"my_enquiry_bids",
         "entities":{"enquiry_id":"46f46148-e88d-42b6-bbd4-d87f14009f65"},
         "confidence":0.95}

Audience: customer
Question: "Are there new packages available?"
Output: {"intent":"package_search","entities":{"query":""},"confidence":0.90}

Audience: customer
Question: "Show me packages in Goa"
Output: {"intent":"package_search","entities":{"query":"Goa"},"confidence":0.94}

Audience: customer
Question: "What is the cancellation policy?"
Output: {"intent":"none","entities":{},"confidence":0.95}

Audience: customer
Question: "Show me my vendor wallet balance"
Output: {"intent":"vendor_wallet","entities":{},"confidence":0.85}

Audience: vendor
Question: "How much can I withdraw from my wallet?"
Output: {"intent":"vendor_wallet","entities":{},"confidence":0.90}

Audience: vendor
Question: "What are the new enquiries in Kashmir?"
Output: {"intent":"vendor_inquiries","entities":{"destination":"Kashmir"},"confidence":0.88}

Audience: customer
Question: "Tell me about the Rishikesh Tours package 17da203f-942d-4650-b9bb-b84a0d928aca"
Output: {"intent":"package_detail",
         "entities":{"package_id":"17da203f-942d-4650-b9bb-b84a0d928aca"},
         "confidence":0.96}

Audience: customer
Question: "Show me adventure activities in India"
Output: {"intent":"activity_search",
         "entities":{"query":"India","category":"adventure"},
         "confidence":0.92}

Audience: customer
Question: "What are the trending destinations right now?"
Output: {"intent":"popular_destinations","entities":{},"confidence":0.95}

"""


class LLMIntentResult(BaseModel):
    intent: LiveDataIntent
    entities: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0)


class LLMIntentClassifier:
    """Structured-output wrapper around LLMGateway.

    Returns None on provider/validation failure or low confidence so the
    resolver can continue with its safe fallback behavior.
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
