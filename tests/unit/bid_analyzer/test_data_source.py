from __future__ import annotations

from decimal import Decimal
from datetime import UTC, datetime

import pytest

from app.capabilities.bid_analyzer import data_source as ds
from app.capabilities.bid_analyzer.data_source import (
    EnquiryAccessError,
    fetch_analysis_input,
    normalize_bid,
)

ENQ = "11111111-1111-4111-8111-111111111111"

REAL_BID = {
    "id": "3fb02ae1-a401-4ff8-85cb-0cdc29922e01",
    "amount": "900000.00",
    "originalAmount": "900000.00",
    "inclusions": ["tets"],
    "exclusions": ["tets"],
    "status": "active",
    "termsAndConditions": ["ssss"],
    "negotiationStatus": "customer_countered",
    "negotiationAmount": "90000.00",
    "negotiationHistory": [
        {"amount": 80000, "status": "pending", "createdAt": "2026-10-09T05:33:36.472Z",
         "proposedBy": "customer", "proposedById": "secret-customer-id"},
        {"amount": 90000, "status": "pending", "createdAt": "2026-10-09T05:34:20.815Z",
         "proposedBy": "customer", "proposedById": "secret-customer-id"},
    ],
    "proposalItinerary": "Day 1: test",
    "validUntil": "2026-10-10T00:00:00.000Z",
    "pdfItineraryUrl": None,
    "agent": {"id": "209d4826-0f79-4f85-9159-ce6f8051119a",
              "companyName": "Akshay Tours", "rating": "0.00", "averageResponseTime": 0},
}

REAL_ENQUIRY = {
    "id": ENQ,
    "title": "Goa Trip",
    "destination": "Goa",
    "duration": "2 Nights / 3 Days",
    "travelDate": "2026-10-17",
    "travelers": 2,
    "budget": "92000.00",
    "status": "live",
    "requirements": {
        "meals": ["Breakfast"],
        "transport": "Tempo Traveller (12-Seater)",
        "activities": ["Scuba Diving"],
        "accommodation": "3-Star Hotel",
        "specialRequests": "Guide: English Speaking Guide. aaaa",
    },
    "user": {"passwordHash": "$2b$LEAK", "emailOtp": "123456", "verificationToken": "LEAK"},
    "bids": [REAL_BID],
}


def _fake_get(routes):
    def fake(path, *, user_jwt, params=None):
        value = routes[path]
        return value(params) if callable(value) else value

    return fake


def _enquiry():
    return {
        "id": ENQ,
        "title": "Goa Trip",
        "destination": "Goa",
        "travelers": 2,
        "budget": "80000",
        "requirements": {"accommodation": "4-star Hotel", "meals": ["Breakfast"]},
        "travelerBreakdown": {"adults": 2, "children": 0, "seniors": 0, "rooms": 1},
    }


def test_foreign_enquiry_is_rejected(monkeypatch):
    monkeypatch.setattr(ds, "get_json", _fake_get({"/enquiries/my": {"data": []}}))
    with pytest.raises(EnquiryAccessError):
        fetch_analysis_input("jwt", ENQ)


def test_invalid_id_is_rejected_without_network(monkeypatch):
    monkeypatch.setattr(ds, "get_json", lambda *a, **k: pytest.fail("no call expected"))
    with pytest.raises(EnquiryAccessError):
        fetch_analysis_input("jwt", "ENQ-12345")


def test_happy_path_filters_and_flags(monkeypatch):
    bids = [
        {"id": "b1", "status": "active", "amount": "75000", "currency": "INR",
         "inclusions": ["Hotel", "hotel"], "exclusions": ["Airfare"],
         "agent": {"companyName": "A Tours"}, "pdfItineraryUrl": "https://x/y.pdf"},
        {"id": "b2", "status": "withdrawn", "amount": "1"},
        {"id": "b3", "status": "pending", "amount": None},
        {"id": "b4", "status": "pending", "amount": 85000},
    ]
    monkeypatch.setattr(
        ds,
        "get_json",
        _fake_get(
            {
                "/enquiries/my": {"data": [{"id": ENQ}]},
                f"/enquiries/{ENQ}": {"data": _enquiry()},
                f"/bids/enquiry/{ENQ}": {"data": {"bids": bids, "total": 4}},
            }
        ),
    )
    out = fetch_analysis_input("jwt", ENQ)

    assert [b.id for b in out.bids] == ["b1", "b4"]
    assert out.excluded_bids == {"b2": "status=withdrawn", "b3": "missing or invalid amount"}
    assert out.bids[0].inclusions == ["Hotel"]
    assert out.bids[0].has_pdf_itinerary is True
    assert out.bids[1].currency_assumed is True
    assert out.enquiry.budget == Decimal("80000")
    assert out.enquiry.budget_currency == "INR"
    assert out.enquiry.budget_currency_assumed is True
    assert any("b4" in w and "INR" in w for w in out.warnings)


def test_pagination_collects_all_pages(monkeypatch):
    def bids_page(params):
        start = params["offset"]
        n = min(50, 60 - start)
        return {"data": {"bids": [
            {"id": f"b{start + i}", "status": "active", "amount": 1000 + i} for i in range(n)
        ], "total": 60}}

    monkeypatch.setattr(
        ds,
        "get_json",
        _fake_get(
            {
                "/enquiries/my": {"data": [{"id": ENQ}]},
                f"/enquiries/{ENQ}": {"data": _enquiry()},
                f"/bids/enquiry/{ENQ}": bids_page,
            }
        ),
    )
    assert len(fetch_analysis_input("jwt", ENQ).bids) == 60


def test_unknown_amount_is_never_defaulted_to_zero():
    bid = normalize_bid({"id": "x", "amount": "abc"})
    assert bid is not None and bid.amount is None

def _routes(bid):
    return {
        "/enquiries/my": {"data": [{"id": ENQ}]},
        f"/enquiries/{ENQ}": {"data": REAL_ENQUIRY},
        f"/bids/enquiry/{ENQ}": {"data": {"bids": [bid], "total": 1}},
    }


def test_real_bid_fields_are_normalized():
    bid = normalize_bid(REAL_BID)
    assert bid.amount == Decimal("900000.00")
    assert bid.original_amount == Decimal("900000.00")
    assert bid.negotiation_amount == Decimal("90000.00")
    assert bid.vendor.rating == 0.0
    assert bid.vendor.company_name == "Akshay Tours"
    assert bid.has_negotiation is True
    assert [e.proposed_by for e in bid.negotiation_history] == ["customer", "customer"]
    assert "secret-customer-id" not in bid.model_dump_json()


def test_expired_bid_is_excluded_by_default(monkeypatch):
    monkeypatch.setattr(ds, "get_json", _fake_get(_routes(REAL_BID)))
    out = fetch_analysis_input("jwt", ENQ, now=datetime(2026, 11, 24, tzinfo=UTC))
    assert out.bids == []
    assert out.excluded_bids[REAL_BID["id"]] == "validity ended 2026-10-10"


def test_expired_bid_can_be_included_and_negotiation_is_warned(monkeypatch):
    monkeypatch.setattr(ds, "get_json", _fake_get(_routes(REAL_BID)))
    out = fetch_analysis_input(
        "jwt", ENQ, include_expired=True, now=datetime(2026, 11, 24, tzinfo=UTC)
    )
    assert out.bids[0].is_expired is True
    assert any("pending counter-offer" in w for w in out.warnings)


def test_bid_valid_through_its_last_day(monkeypatch):
    monkeypatch.setattr(ds, "get_json", _fake_get(_routes(REAL_BID)))
    out = fetch_analysis_input("jwt", ENQ, now=datetime(2026, 10, 10, 23, 0, tzinfo=UTC))
    assert len(out.bids) == 1


def test_sensitive_enquiry_fields_are_never_retained():
    dumped = ds.normalize_enquiry(REAL_ENQUIRY).model_dump_json()
    for leaked in ("passwordHash", "$2b$LEAK", "emailOtp", "verificationToken", "LEAK"):
        assert leaked not in dumped
    assert ds.normalize_enquiry(REAL_ENQUIRY).budget == Decimal("92000.00")
    

@pytest.mark.parametrize("status", ["withdrawn", "rejected", "expired"])
def test_only_withdrawn_rejected_expired_are_excluded(monkeypatch, status):
    bid = {**REAL_BID, "status": status, "validUntil": "2027-01-01T00:00:00.000Z"}
    monkeypatch.setattr(ds, "get_json", _fake_get(_routes(bid)))
    out = fetch_analysis_input("jwt", ENQ, now=datetime(2026, 11, 24, tzinfo=UTC))
    assert out.bids == []
    assert out.excluded_bids[bid["id"]] == f"status={status}"


@pytest.mark.parametrize("status", ["active", "accepted"])
def test_active_and_accepted_are_shown_without_status_warning(monkeypatch, status):
    bid = {**REAL_BID, "status": status, "validUntil": "2027-01-01T00:00:00.000Z"}
    monkeypatch.setattr(ds, "get_json", _fake_get(_routes(bid)))
    out = fetch_analysis_input("jwt", ENQ, now=datetime(2026, 11, 24, tzinfo=UTC))
    assert [b.status for b in out.bids] == [status]
    assert not any("unrecognized status" in w for w in out.warnings)


@pytest.mark.parametrize("status", ["pending", "cancelled"])
def test_other_statuses_are_kept_with_a_warning(monkeypatch, status):
    bid = {**REAL_BID, "status": status, "validUntil": "2027-01-01T00:00:00.000Z"}
    monkeypatch.setattr(ds, "get_json", _fake_get(_routes(bid)))
    out = fetch_analysis_input("jwt", ENQ, now=datetime(2026, 11, 24, tzinfo=UTC))
    assert len(out.bids) == 1
    assert any(f"unrecognized status '{status}'" in w for w in out.warnings)


def test_accepted_bid_is_not_dropped_for_a_past_validity_date(monkeypatch):
    bid = {**REAL_BID, "status": "accepted"}  # validUntil 2026-10-10, already passed
    monkeypatch.setattr(ds, "get_json", _fake_get(_routes(bid)))
    out = fetch_analysis_input("jwt", ENQ, now=datetime(2026, 11, 24, tzinfo=UTC))
    assert [b.status for b in out.bids] == ["accepted"]
    assert out.bids[0].is_expired is True