"""Intent classification: question + audience -> live-data intent."""
from __future__ import annotations

import pytest

from app.capabilities.support.live_data_classifier import (
    LiveDataIntent,
    classify,
)


# ---------------------------------------------------------------------------
# Customer
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("q,expected", [
    # Original
    ("What are my bookings?", LiveDataIntent.MY_BOOKINGS),
    ("What is the status of my booking?", LiveDataIntent.MY_BOOKINGS),
    ("Show me my upcoming bookings", LiveDataIntent.MY_BOOKINGS),
    ("my upcoming trips", LiveDataIntent.MY_BOOKINGS),
    ("What are my payments?", LiveDataIntent.MY_PAYMENTS),
    ("How much did I pay last month?", LiveDataIntent.MY_PAYMENTS),
    ("payment history", LiveDataIntent.MY_PAYMENTS),
    ("How many Swabi coins do I have?", LiveDataIntent.MY_COINS),
    ("my coins balance", LiveDataIntent.MY_COINS),
    ("my wallet balance", LiveDataIntent.MY_COINS),
    ("coin history", LiveDataIntent.MY_COIN_HISTORY),
    ("wallet transactions", LiveDataIntent.MY_COIN_HISTORY),
    ("My enquiries", LiveDataIntent.MY_ENQUIRIES),
    ("List my posted travel requests", LiveDataIntent.MY_ENQUIRIES),
    (
        "Show me the bids on my any one single enquiry only",
        LiveDataIntent.MY_ENQUIRY_BIDS,
    ),
    # New — "last booked packages" variants
    ("Show me my last booked packages", LiveDataIntent.MY_BOOKINGS),
    ("my last booking", LiveDataIntent.MY_BOOKINGS),
    ("my recent trips", LiveDataIntent.MY_BOOKINGS),
    ("my latest booked packages", LiveDataIntent.MY_BOOKINGS),
    ("my booked packages", LiveDataIntent.MY_BOOKINGS),
    ("my purchased trips", LiveDataIntent.MY_BOOKINGS),
    ("What have I booked?", LiveDataIntent.MY_BOOKINGS),
    ("Which packages have I booked?", LiveDataIntent.MY_BOOKINGS),
])
def test_customer_intents(q, expected):
    result = classify(q, "customer")
    assert result.intent == expected


def test_pending_packages_resolve_to_pending_customer_bookings():
    result = classify("Show me my pending packages", "customer")

    assert result.intent == LiveDataIntent.MY_BOOKINGS
    assert result.entities == {"status": "pending"}


@pytest.mark.parametrize(
    "question,expected_status",
    [
        ("Show my cancel bookings", "cancelled"),
        ("Show my canceled bookings", "cancelled"),
        ("Show my cancelled bookings", "cancelled"),
        ("Show my bookings that are cancel", "cancelled"),
        ("Show my complete bookings", "completed"),
        ("Show my completed bookings", "completed"),
        ("Show my upcoming bookings", "upcoming"),
        ("Show my pending bookings", "pending"),
    ],
)
def test_booking_status_variants_are_normalized(question, expected_status):
    result = classify(question, "customer")

    assert result.intent == LiveDataIntent.MY_BOOKINGS
    assert result.entities == {"status": expected_status}


@pytest.mark.parametrize(
    "question,expected_status",
    [
        ("Show my cancel payments", "cancelled"),
        ("Show my canceled payments", "cancelled"),
        ("Show my cancelled payments", "cancelled"),
        ("Show my complete payments", "completed"),
        ("Show my completed payments", "completed"),
    ],
)
def test_payment_status_variants_are_normalized(question, expected_status):
    result = classify(question, "customer")

    assert result.intent == LiveDataIntent.MY_PAYMENTS
    assert result.entities == {"status": expected_status}


# ---------------------------------------------------------------------------
# Vendor
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("q,expected", [
    ("What is my vendor wallet balance?", LiveDataIntent.VENDOR_WALLET),
    ("my withdrawable balance", LiveDataIntent.VENDOR_WALLET),
    ("How much can I withdraw?", LiveDataIntent.VENDOR_WALLET),
    ("wallet transactions", LiveDataIntent.VENDOR_TRANSACTIONS),
    ("my wallet ledger", LiveDataIntent.VENDOR_TRANSACTIONS),
    ("bid history", LiveDataIntent.VENDOR_TRANSACTIONS),
    ("my withdrawals", LiveDataIntent.VENDOR_WITHDRAWALS),
    ("pending withdrawals", LiveDataIntent.VENDOR_WITHDRAWALS),
    ("withdrawal requests", LiveDataIntent.VENDOR_WITHDRAWALS),
    ("new enquiries to bid on", LiveDataIntent.VENDOR_INQUIRIES),
    ("enquiries I can bid", LiveDataIntent.VENDOR_INQUIRIES),
    ("my vendor inquiries", LiveDataIntent.VENDOR_INQUIRIES),
    ("my vendor bookings", LiveDataIntent.VENDOR_BOOKINGS),
    ("trips I am serving", LiveDataIntent.VENDOR_BOOKINGS),
    ("my bids", LiveDataIntent.VENDOR_BIDS),
    ("bids I placed", LiveDataIntent.VENDOR_BIDS),
    ("my accepted bids", LiveDataIntent.VENDOR_BIDS),
    ("my subscription", LiveDataIntent.VENDOR_SUBSCRIPTION),
    ("which plan am I on", LiveDataIntent.VENDOR_SUBSCRIPTION),
])
def test_vendor_intents(q, expected):
    result = classify(q, "vendor")
    assert result.intent == expected


# ---------------------------------------------------------------------------
# Scope gates: cross-audience phrasing must yield NONE
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("q", [
    "What is my vendor wallet balance?",
    "my withdrawals",
    "my vendor bookings",
    "enquiries to bid",
    "my subscription plan",
    "my bids",
])
def test_customer_asking_vendor_scope_none(q):
    assert classify(q, "customer").intent == LiveDataIntent.NONE


@pytest.mark.parametrize("q", [
    "What are my payments?",
    "How much did I pay?",
    "payment history",
])
def test_vendor_asking_customer_scope_none(q):
    assert classify(q, "vendor").intent == LiveDataIntent.NONE


# ---------------------------------------------------------------------------
# Ambiguous / non-live questions -> NONE
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("q", [
    "What is the cancellation policy?",
    "How does escrow work?",
    "Tell me about Swabi Plus",
    "What activities are in Kashmir?",
    "How do I place a bid?",
    "",
    # New negative cases — must NOT trigger live-data intent
    "How do I book a package?",
    "What should I do if I booked the wrong dates?",
    "I booked a trip yesterday and I have a question",
    "Tell me about package booking",
])
def test_policy_and_travel_questions_none(q):
    assert classify(q, "customer").intent == LiveDataIntent.NONE
    assert classify(q, "vendor").intent == LiveDataIntent.NONE


@pytest.mark.parametrize(
    ("question", "expected_entity"),
    [
        (
            "Tell me the details for package PKG-922",
            {"package_code": "PKG-922"},
        ),
        (
            "Tell me about package 17da203f-942d-4650-b9bb-b84a0d928aca",
            {"package_id": "17da203f-942d-4650-b9bb-b84a0d928aca"},
        ),
    ],
)
def test_public_package_detail_resolves_code_or_uuid(question, expected_entity):
    result = classify(question, "customer")

    assert result.intent == LiveDataIntent.PACKAGE_DETAIL
    assert result.entities == expected_entity
