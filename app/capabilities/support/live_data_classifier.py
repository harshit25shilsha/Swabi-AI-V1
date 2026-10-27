"""Deterministic classifier: question + audience -> live-data intent.

No LLM. Every pattern is a curated match on common phrasings. If a
question requires live data and the classifier returns NONE, the service
falls back to RAG (which will correctly say "I don't have that" for
user-private questions).

The classifier does NOT authorise. It only proposes an intent and, when
extractable, the parameters the fetcher needs (booking id, enquiry id,
status filter). The service enforces audience match and JWT presence
before any API call.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class LiveDataIntent(StrEnum):
    NONE = "none"
    # Customer
    MY_BOOKINGS = "my_bookings"
    MY_BOOKING_DETAIL = "my_booking_detail"
    MY_PAYMENTS = "my_payments"
    MY_ENQUIRY_BIDS = "my_enquiry_bids"
    MY_COINS = "my_coins"
    MY_COIN_HISTORY = "my_coin_history"
    MY_ENQUIRIES = "my_enquiries"
    # Public catalogue intent; API integration is not wired yet.
    PACKAGE_SEARCH = "package_search"
    # Vendor
    VENDOR_WALLET = "vendor_wallet"
    VENDOR_TRANSACTIONS = "vendor_transactions"
    VENDOR_WITHDRAWALS = "vendor_withdrawals"
    VENDOR_INQUIRIES = "vendor_inquiries"
    VENDOR_BOOKINGS = "vendor_bookings"
    VENDOR_BIDS = "vendor_bids"
    VENDOR_SUBSCRIPTION = "vendor_subscription"


CUSTOMER_ONLY: set[LiveDataIntent] = {
    LiveDataIntent.MY_BOOKINGS,
    LiveDataIntent.MY_BOOKING_DETAIL,
    LiveDataIntent.MY_PAYMENTS,
    LiveDataIntent.MY_ENQUIRY_BIDS,
    LiveDataIntent.MY_COINS,
    LiveDataIntent.MY_COIN_HISTORY,
    LiveDataIntent.MY_ENQUIRIES,
}

VENDOR_ONLY: set[LiveDataIntent] = {
    LiveDataIntent.VENDOR_WALLET,
    LiveDataIntent.VENDOR_TRANSACTIONS,
    LiveDataIntent.VENDOR_WITHDRAWALS,
    LiveDataIntent.VENDOR_INQUIRIES,
    LiveDataIntent.VENDOR_BOOKINGS,
    LiveDataIntent.VENDOR_BIDS,
    LiveDataIntent.VENDOR_SUBSCRIPTION,
}


@dataclass
class ClassifiedIntent:
    intent: LiveDataIntent = LiveDataIntent.NONE
    entities: dict[str, Any] = field(default_factory=dict)
    reason: str = ""


# ---------------------------------------------------------------------------
# Entity extraction — IDs and status filters the fetcher consumes.
# ---------------------------------------------------------------------------

_UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}"
    r"-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
_BOOKING_CODE_RE = re.compile(r"\bBK-\d+\b", re.I)


def _extract_uuid(q: str) -> str | None:
    m = _UUID_RE.search(q)
    return m.group(0) if m else None


def _extract_booking_id(q: str) -> str | None:
    """Return a UUID or booking code (``BK-XXXX``) if present.

    The fetcher will use the UUID directly; a booking code triggers the
    fetcher's graceful fall-back to ``MY_BOOKINGS`` (the LLM sees the
    full list and answers about the specific one).
    """
    uuid = _extract_uuid(q)
    if uuid:
        return uuid
    m = _BOOKING_CODE_RE.search(q)
    return m.group(0).upper() if m else None


# Supported by the backend: bookings = all|cancelled|pending|upcoming
_BOOKING_STATUS_MAP: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bcancell?ed\b", re.I), "cancelled"),
    (re.compile(r"\bpending\b", re.I), "pending"),
    (re.compile(r"\bupcoming\b", re.I), "upcoming"),
]

# Supported by the backend: payments = pending|completed|cancelled.
# "failed" is mapped to "cancelled" because the backend has no failed state.
_PAYMENT_STATUS_MAP: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bcancell?ed\b", re.I), "cancelled"),
    (re.compile(r"\bfailed\b", re.I), "cancelled"),
    (re.compile(r"\bcompleted?\b", re.I), "completed"),
    (re.compile(r"\bpending\b", re.I), "pending"),
]


def _extract_status(
    q: str, table: list[tuple[re.Pattern[str], str]]
) -> str | None:
    for pattern, value in table:
        if pattern.search(q):
            return value
    return None


# ---------------------------------------------------------------------------
# Scope guards — specific phrases that unambiguously belong to one audience.
# ---------------------------------------------------------------------------

_VENDOR_CONTEXT_PATTERNS = [
    re.compile(r"\bvendor\s+(wallet|bookings?|inquir|transactions?)\b", re.I),
    re.compile(r"\bmy\s+withdrawals?\b", re.I),
    re.compile(r"\bpending\s+withdrawals?\b", re.I),
    re.compile(r"\bwithdrawal\s+requests?\b", re.I),
    re.compile(r"\benquir(y|ies)\s+(to\s+bid|i\s+can\s+bid)\b", re.I),
    re.compile(r"\bbids?\s+i\s+placed\b", re.I),
    re.compile(r"\b(accepted|rejected|withdrawn)\s+bids?\b", re.I),
    re.compile(r"\bmy\s+subscription\b", re.I),
    re.compile(r"\bwhich\s+plan\s+am\s+i\s+on\b", re.I),
    re.compile(r"\btrips?\s+i('| a)m\s+(serving|running|operating)\b", re.I),
]

_CUSTOMER_CONTEXT_PATTERNS = [
    re.compile(r"\bmy\s+payments?\b", re.I),
    re.compile(r"\bpayment\s+history\b", re.I),
    re.compile(r"\bhow\s+much\s+did\s+i\s+pay\b", re.I),
    re.compile(r"\b(swabi\s+)?coins?\s+(balance|history)\b", re.I),
    re.compile(r"\bhow\s+many\s+(swabi\s+)?coins?\b", re.I),
    re.compile(r"\bmy\s+posted\s+(trips?|travel\s+requests?)\b", re.I),
]


# ---------------------------------------------------------------------------
# Context patterns for the ID-required intents. The ID must be extractable
# for these to fire; otherwise they fall through to the list form.
# ---------------------------------------------------------------------------

_MY_BOOKING_DETAIL_CONTEXT = [
    re.compile(r"\bbooking\b", re.I),
    re.compile(r"\bbk-\d+\b", re.I),
]

_MY_ENQUIRY_BIDS_CONTEXT = [
    re.compile(r"\bbids?\s+(on|for)\s+(my\s+|the\s+)?enquir", re.I),
    re.compile(r"\bwho\s+bid\s+(on|for)\s+(my\s+|the\s+)?enquir", re.I),
    re.compile(r"\bproposals?\s+(on|for)\s+(my\s+|the\s+)?enquir", re.I),
    re.compile(r"\benquir(y|ies)\s+bids?\b", re.I),
]


# ---------------------------------------------------------------------------
# Per-intent patterns
# ---------------------------------------------------------------------------

_MY_BOOKINGS = [
    re.compile(r"\bmy\s+bookings?\b", re.I),
    re.compile(r"\bbooking\s+status\b", re.I),
    re.compile(r"\bstatus\s+of\s+my\s+booking\b", re.I),
    re.compile(r"\bshow\s+(me\s+)?my\s+(bookings?|trips?)\b", re.I),
    re.compile(
        r"\bmy\s+(upcoming|pending|cancelled|past)\s+(bookings?|trips?)\b",
        re.I,
    ),
    re.compile(
        r"\bshow\s+(me\s+)?my\s+(upcoming|pending|cancelled|past)\s+"
        r"packages?\b",
        re.I,
    ),
    re.compile(
        r"\bmy\s+(last|latest|recent(?:ly)?|past|previous)\s+"
        r"(booked\s+|purchased\s+)?(packages?|bookings?|trips?)\b",
        re.I,
    ),
    re.compile(
        r"\bmy\s+(booked|purchased)\s+(packages?|bookings?|trips?)\b",
        re.I,
    ),
    re.compile(
        r"\b(what|which)\s+(packages?\s+|trips?\s+)?(have\s+)?i\s+"
        r"(booked|book)\b",
        re.I,
    ),
]

_MY_PAYMENTS = [
    re.compile(r"\bmy\s+payments?\b", re.I),
    re.compile(r"\bhow\s+much\s+did\s+i\s+pay\b", re.I),
    re.compile(r"\bpayment\s+history\b", re.I),
    re.compile(
        r"\bmy\s+(pending|completed|cancelled|failed)\s+payments?\b", re.I
    ),
]

_MY_COINS = [
    re.compile(r"\bmy\s+(swabi\s+)?coins?\b", re.I),
    re.compile(r"\bcoins?\s+balance\b", re.I),
    re.compile(r"\bhow\s+many\s+(swabi\s+)?coins?\b", re.I),
    re.compile(r"\bmy\s+wallet\s+balance\b", re.I),
]

_MY_COIN_HISTORY = [
    re.compile(r"\b(my\s+)?coins?\s+(history|transactions?)\b", re.I),
    re.compile(r"\b(my\s+)?wallet\s+(history|transactions?|ledger)\b", re.I),
    re.compile(r"\bwhere\s+did\s+(my\s+)?coins?\s+go\b", re.I),
]

_MY_ENQUIRIES = [
    re.compile(r"\bmy\s+enquir(y|ies)\b", re.I),
    re.compile(r"\bmy\s+(posted\s+)?(trip|travel)\s+requests?\b", re.I),
    re.compile(r"\b(my\s+)?enquir(y|ies)\s+(status|list)\b", re.I),
]

_VENDOR_WALLET = [
    re.compile(r"\bvendor\s+wallet\s+balance\b", re.I),
    # Added — plain "my wallet balance" from a vendor is unambiguously
    # vendor-scope (customers' equivalent is "my coins balance").
    re.compile(r"\bmy\s+wallet\s+balance\b", re.I),
    re.compile(r"\bmy\s+(withdrawable|available)\s+balance\b", re.I),
    re.compile(
        r"\bhow\s+much\s+(can\s+i\s+withdraw|is\s+in\s+my\s+wallet)\b",
        re.I,
    ),
]

_VENDOR_TRANSACTIONS = [
    re.compile(
        r"\b(my\s+)?(vendor\s+)?wallet\s+(transactions?|ledger|history)\b",
        re.I,
    ),
    re.compile(r"\b(bid|topup|top-up)\s+(history|transactions?)\b", re.I),
]

_VENDOR_WITHDRAWALS = [
    re.compile(r"\bmy\s+withdrawals?\b", re.I),
    re.compile(r"\bwithdrawal\s+(requests?|status|history)\b", re.I),
    re.compile(r"\bpending\s+withdrawals?\b", re.I),
]

_VENDOR_INQUIRIES = [
    re.compile(
        r"\b(new\s+)?enquir(y|ies)\s+(i\s+can\s+bid|to\s+bid)\b", re.I
    ),
    re.compile(r"\bnew\s+enquir(y|ies)\b", re.I),
    re.compile(r"\binquir(y|ies)\s+to\s+bid\b", re.I),
    re.compile(r"\bmy\s+vendor\s+inquir(y|ies)\b", re.I),
]

_VENDOR_BOOKINGS = [
    re.compile(r"\bmy\s+vendor\s+bookings?\b", re.I),
    re.compile(
        r"\bbookings?\s+(assigned\s+to\s+me|for\s+my\s+trips?)\b", re.I
    ),
    re.compile(r"\btrips?\s+i('| a)m\s+(serving|running|operating)\b", re.I),
]

_VENDOR_BIDS = [
    re.compile(r"\bmy\s+(active\s+)?bids?\b", re.I),
    re.compile(r"\bbids?\s+i\s+(placed|submitted)\b", re.I),
    re.compile(r"\bmy\s+(accepted|rejected|withdrawn)\s+bids?\b", re.I),
]

_VENDOR_SUBSCRIPTION = [
    re.compile(r"\bmy\s+subscription\b", re.I),
    re.compile(r"\bmy\s+(vendor\s+)?plan\b", re.I),
    re.compile(r"\bwhich\s+plan\s+am\s+i\s+on\b", re.I),
    re.compile(r"\bam\s+i\s+subscribed\b", re.I),
]


def _matches(patterns: list[re.Pattern[str]], text: str) -> bool:
    return any(p.search(text) for p in patterns)


def classify(question: str, audience: str) -> ClassifiedIntent:
    """Return the live-data intent for this question + audience.

    Order:
      1. Empty question -> NONE.
      2. Scope guard: if the phrasing is unambiguously the *other*
         audience, return NONE (the service turns this into a
         needs_live_data with a scope-mismatch message).
      3. Customer ID-required intents first — they are strict supersets
         of the list intents and require an extracted ID to fire.
      4. Plain intent match within the caller's audience.
    """
    q = question.strip()
    if not q:
        return ClassifiedIntent(reason="empty question")

    # --- Scope guards ---------------------------------------------------
    if audience == "customer":
        if _matches(_VENDOR_CONTEXT_PATTERNS, q):
            return ClassifiedIntent(
                intent=LiveDataIntent.NONE,
                reason="vendor-scope phrasing; customer audience",
            )
    elif audience == "vendor":
        if _matches(_CUSTOMER_CONTEXT_PATTERNS, q):
            return ClassifiedIntent(
                intent=LiveDataIntent.NONE,
                reason="customer-scope phrasing; vendor audience",
            )

    # --- Customer intents ----------------------------------------------
    if audience == "customer":
        # MY_BOOKING_DETAIL requires both booking context and an ID.
        if _matches(_MY_BOOKING_DETAIL_CONTEXT, q):
            booking_id = _extract_booking_id(q)
            if booking_id:
                return ClassifiedIntent(
                    LiveDataIntent.MY_BOOKING_DETAIL,
                    {"booking_id": booking_id},
                    reason="booking detail",
                )
            # No ID found — fall through to MY_BOOKINGS.

        # MY_ENQUIRY_BIDS requires both bids context and an enquiry ID.
        if _matches(_MY_ENQUIRY_BIDS_CONTEXT, q):
            enquiry_id = _extract_uuid(q)
            if enquiry_id:
                return ClassifiedIntent(
                    LiveDataIntent.MY_ENQUIRY_BIDS,
                    {"enquiry_id": enquiry_id},
                    reason="enquiry bids",
                )
            # No ID found — fall through to MY_ENQUIRIES.

        if _matches(_MY_COIN_HISTORY, q):
            return ClassifiedIntent(
                LiveDataIntent.MY_COIN_HISTORY, reason="coin history"
            )
        if _matches(_MY_COINS, q):
            return ClassifiedIntent(
                LiveDataIntent.MY_COINS, reason="coin balance"
            )
        if _matches(_MY_PAYMENTS, q):
            status = _extract_status(q, _PAYMENT_STATUS_MAP)
            entities = {"status": status} if status else {}
            return ClassifiedIntent(
                LiveDataIntent.MY_PAYMENTS, entities, reason="my payments"
            )
        if _matches(_MY_BOOKINGS, q):
            status = _extract_status(q, _BOOKING_STATUS_MAP)
            entities = {"status": status} if status else {}
            return ClassifiedIntent(
                LiveDataIntent.MY_BOOKINGS, entities, reason="my bookings"
            )
        if _matches(_MY_ENQUIRIES, q):
            return ClassifiedIntent(
                LiveDataIntent.MY_ENQUIRIES, reason="my enquiries"
            )

    # --- Vendor intents -------------------------------------------------
    elif audience == "vendor":
        if _matches(_VENDOR_SUBSCRIPTION, q):
            return ClassifiedIntent(
                LiveDataIntent.VENDOR_SUBSCRIPTION, reason="subscription"
            )
        if _matches(_VENDOR_WITHDRAWALS, q):
            return ClassifiedIntent(
                LiveDataIntent.VENDOR_WITHDRAWALS, reason="withdrawals"
            )
        if _matches(_VENDOR_TRANSACTIONS, q):
            return ClassifiedIntent(
                LiveDataIntent.VENDOR_TRANSACTIONS,
                reason="wallet transactions",
            )
        if _matches(_VENDOR_WALLET, q):
            return ClassifiedIntent(
                LiveDataIntent.VENDOR_WALLET, reason="wallet balance"
            )
        if _matches(_VENDOR_INQUIRIES, q):
            return ClassifiedIntent(
                LiveDataIntent.VENDOR_INQUIRIES, reason="inquiries to bid"
            )
        if _matches(_VENDOR_BIDS, q):
            return ClassifiedIntent(
                LiveDataIntent.VENDOR_BIDS, reason="my bids"
            )
        if _matches(_VENDOR_BOOKINGS, q):
            return ClassifiedIntent(
                LiveDataIntent.VENDOR_BOOKINGS, reason="vendor bookings"
            )

    return ClassifiedIntent(reason="no match")
