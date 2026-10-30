"""Formatter tests: compact, capped, no raw JSON leaks."""
from __future__ import annotations

from app.capabilities.support.live_data_classifier import LiveDataIntent
from app.capabilities.support.live_data_format import format_for_intent


def test_my_bookings_empty():
    assert format_for_intent(
        LiveDataIntent.MY_BOOKINGS, {"bookings": [], "total": 0}
    ) == "You have no bookings."


def test_my_bookings_lists_status_and_dates():
    payload = {
        "bookings": [{
            "bookingCode": "BK-1",
            "metadata": {"destination": "Goa"},
            "startDate": "2026-11-01T00:00:00Z",
            "endDate": "2026-11-03T00:00:00Z",
            "amount": "8000.00",
            "currency": "INR",
            "status": "upcoming",
            "paymentStatus": "paid",
        }],
        "total": 1,
    }
    text = format_for_intent(LiveDataIntent.MY_BOOKINGS, payload)
    assert "BK-1" in text
    assert "Goa" in text
    assert "upcoming" in text
    assert "paid" in text


def test_my_bookings_capped_at_5():
    bookings = [
        {
            "bookingCode": f"BK-{i}",
            "metadata": {"destination": "Goa"},
            "startDate": "2026-11-01T00:00:00Z",
            "endDate": "2026-11-03T00:00:00Z",
            "amount": "100",
            "currency": "INR",
            "status": "upcoming",
            "paymentStatus": "paid",
        }
        for i in range(20)
    ]
    text = format_for_intent(
        LiveDataIntent.MY_BOOKINGS, {"bookings": bookings, "total": 20}
    )
    assert text.count("BK-") == 5
    assert "of 20" in text


def test_package_search_hides_negative_duration():
    text = format_for_intent(
        LiveDataIntent.PACKAGE_SEARCH,
        {
            "packages": [{
                "packageCode": "PKG-920",
                "title": "Italy",
                "destination": "Italy",
                "duration": "-15 Days",
                "numberOfDays": -15,
                "price": 599,
            }],
            "total": 1,
        },
    )

    assert "duration unavailable" in text
    assert "-15 Days" not in text


def test_package_search_uses_conversational_chat_format():
    text = format_for_intent(
        LiveDataIntent.PACKAGE_SEARCH,
        {
            "packages": [
                {
                    "packageCode": "PKG-922",
                    "title": "Rishikesh Tours",
                    "destination": "India",
                    "numberOfDays": 2,
                    "price": 2000,
                    "currency": "INR",
                    "agent": {"companyName": "Karan Tours"},
                },
                {
                    "packageCode": "PKG-921",
                    "title": "Uttar Pradesh Tour",
                    "destination": "India",
                    "duration": "2 Days",
                    "price": 4000,
                    "currency": "INR",
                    "agent": {"companyName": "Navneet Tours"},
                },
            ],
            "total": 13,
        },
    )

    assert "I found 13 packages in the catalogue" in text
    assert "Here are 2 to get you started" in text
    assert "**Rishikesh Tours** (PKG-922) - India, 2 days, INR 2,000.00" in text
    assert "offered by Karan Tours" in text
    assert "Would you like details about any of these packages?" in text
    assert "PUBLIC PACKAGES" not in text
    assert " | " not in text


def test_package_search_uses_valid_day_count_over_bad_duration_label():
    text = format_for_intent(
        LiveDataIntent.PACKAGE_SEARCH,
        {
            "packages": [{
                "packageCode": "PKG-920",
                "title": "Italy",
                "destination": "Italy",
                "duration": "-15 Days",
                "numberOfDays": 15,
                "price": 599,
            }],
            "total": 1,
        },
    )

    assert "15 days" in text
    assert "-15 Days" not in text


def test_my_coins_shows_all_fields():
    text = format_for_intent(LiveDataIntent.MY_COINS, {
        "coinBalance": 100, "totalCoinsBalance": 250,
        "pendingCoins": 150, "lifetimeSavings": 500, "creditBalance": 0,
    })
    assert "100" in text
    assert "250" in text
    assert "150" in text


def test_vendor_wallet_shows_balances():
    text = format_for_intent(LiveDataIntent.VENDOR_WALLET, {
        "escrowBalance": 500, "topupBalance": 1000,
        "lockedBalance": 0, "withdrawable": 500,
        "availableForBids": 1000,
        "lifetimeEarned": 2000, "lifetimeWithdrawn": 500,
        "lifetimeBidSpend": 300, "lifetimeTopup": 1500,
    })
    assert "INR 500.00" in text
    assert "INR 1,000.00" in text
    assert "INR 2,000.00" in text
    assert "INR 0.00" in text


def test_money_formats_negative_with_sign_before_currency():
    from app.capabilities.support.live_data_format import _money

    assert _money("-140.00", "INR") == "-INR 140.00"
    assert _money("1696.88", "INR") == "INR 1,696.88"
    assert _money(0, "INR") == "INR 0.00"
    assert _money(None, "INR") == "—"


def test_money_without_currency_has_no_prefix():
    from app.capabilities.support.live_data_format import _money

    assert _money(500) == "500.00"
    assert _money("1234.56") == "1,234.56"


def test_vendor_inquiries_includes_stats():
    text = format_for_intent(LiveDataIntent.VENDOR_INQUIRIES, {
        "inquiries": [{
            "inquiryCode": "ENQ-1", "destination": "Shimla",
            "travelDates": {"from": "2026-10-17", "to": "2026-10-17"},
            "persons": 2, "budget": "9000", "status": "new",
        }],
        "total": 1,
        "stats": {"new": 11, "bid_placed": 4, "won": 0, "lost": 2},
    })
    assert "ENQ-1" in text
    assert "new=11" in text


def test_vendor_bids_capped():
    bids = [
        {
            "amount": f"{1000 + i}",
            "enquiry": {"title": "T", "destination": "Goa"},
            "status": "active",
            "negotiationStatus": "none",
            "createdAt": "2026-10-01T00:00:00Z",
        }
        for i in range(20)
    ]
    text = format_for_intent(
        LiveDataIntent.VENDOR_BIDS,
        {"bids": bids, "total": 20, "stats": {}},
    )
    assert text.count("- ") == 5


def test_vendor_subscription_none():
    assert format_for_intent(
        LiveDataIntent.VENDOR_SUBSCRIPTION, None
    ) == "You have no active subscription plan."


def test_vendor_subscription_present():
    text = format_for_intent(LiveDataIntent.VENDOR_SUBSCRIPTION, {
        "plan": "Professional", "status": "active",
        "expiresAt": "2027-01-01T00:00:00Z",
    })
    assert "Professional" in text
    assert "active" in text
