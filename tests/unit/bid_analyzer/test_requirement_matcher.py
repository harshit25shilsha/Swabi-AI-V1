from __future__ import annotations

import pytest

from app.capabilities.bid_analyzer import requirement_matcher as rm
from app.capabilities.bid_analyzer.matching_schemas import MatchResult as M
from app.capabilities.bid_analyzer.offer_schemas import (
    AccommodationItem,
    ActivityItem,
    ExtractedOffer,
    ExtractionStatus,
    NormalizedOffer,
    ServiceItem,
    TransportItem,
)
from app.capabilities.bid_analyzer.requirements import parse_special_requests
from app.capabilities.bid_analyzer.schemas import EnquiryRequirementsSnapshot, EnquirySnapshot


def _acc(stars=None, ptype=None, status="included", detail=None):
    return ExtractedOffer(
        accommodation=AccommodationItem(status=status, star_rating=stars, property_type=ptype, detail=detail, evidence="x")
    )


def _transport(vehicle=None, private=None, status="included"):
    return ExtractedOffer(transport=TransportItem(status=status, vehicle=vehicle, is_private=private, evidence="x"))


# ------------------------------------------------------------ accommodation


@pytest.mark.parametrize(
    "asked,offer,expected",
    [
        ("3-Star Hotel", _acc(4, "hotel"), M.MATCH),
        ("3-Star Hotel", _acc(2, "hotel"), M.PARTIAL),
        ("3-Star Hotel", _acc(1, "hotel"), M.MISMATCH),
        ("4-star Resort", _acc(4, "hotel"), M.PARTIAL),
        ("Houseboat", _acc(None, "hotel"), M.MISMATCH),
        ("4-star Hotel or 5-star Resort", _acc(4, "resort"), M.MATCH),
        ("3-Star Hotel", _acc(None, "hotel"), M.PARTIAL),  # type ok, stars not stated
        ("3-Star Hotel", _acc(status="unknown"), M.UNKNOWN),
        ("3-Star Hotel", _acc(status="excluded"), M.MISMATCH),
    ],
)
def test_accommodation(asked, offer, expected):
    assert rm.match_accommodation(asked, offer).result == expected


def test_unstated_stars_mark_the_match_as_unverified():
    match = rm.match_accommodation("3-Star Hotel", _acc(None, "hotel"))
    assert match.unverified is True


def test_no_accommodation_requirement_is_skipped():
    assert rm.match_accommodation(None, _acc(4, "hotel")) is None


# ---------------------------------------------------------------- transport


@pytest.mark.parametrize(
    "asked,offer,expected",
    [
        ("Tempo Traveller (12-Seater)", _transport("AC sedan", True), M.MISMATCH),
        ("Private SUV", _transport("sedan", True), M.PARTIAL),
        ("Private Cab (Sedan)", _transport("sedan", True), M.MATCH),
        ("Private Cab", _transport("Volvo bus", False), M.MISMATCH),
        ("Shared Coach", _transport("SUV", True), M.MATCH),
        ("Private Cab (Sedan)", _transport(None, None), M.PARTIAL),
        ("Local Transportation", _transport("sedan"), M.MATCH),
        ("Private Cab", _transport(status="unknown"), M.UNKNOWN),
        ("Private Cab", _transport(status="excluded"), M.MISMATCH),
    ],
)
def test_transport(asked, offer, expected):
    assert rm.match_transport(asked, offer).result == expected


# -------------------------------------------------------------------- meals


def _meals(b="unknown", lunch="unknown", d="unknown"):
    offer = ExtractedOffer()
    offer.meals.breakfast = ServiceItem(status=b, evidence="x")
    offer.meals.lunch = ServiceItem(status=lunch, evidence="x")
    offer.meals.dinner = ServiceItem(status=d, evidence="x")
    return offer


@pytest.mark.parametrize(
    "asked,offer,expected",
    [
        (["Breakfast"], _meals(b="included"), M.MATCH),
        (["All Meals"], _meals("included", "included", "included"), M.MATCH),
        (["All Meals"], _meals("included", "excluded", "unknown"), M.PARTIAL),
        (["Breakfast", "Dinner"], _meals(), M.UNKNOWN),
        (["Breakfast"], _meals(b="excluded"), M.MISMATCH),
        (["Breakfast"], _meals(b="included", d="excluded"), M.MATCH),  # dinner not requested
    ],
)
def test_meals(asked, offer, expected):
    assert rm.match_meals(asked, offer).result == expected


def test_no_meals_requested_is_skipped():
    assert rm.match_meals([], _meals(b="included")) is None


# --------------------------------------------------------------- activities


def test_known_activity_matches_by_keyword():
    offer = ExtractedOffer(activities=[ActivityItem(name="Scuba session", status="included", evidence="Scuba session")])
    assert rm.match_activities(["Scuba Diving"], offer).result == M.MATCH


def test_excluded_activity_is_a_miss():
    offer = ExtractedOffer(activities=[ActivityItem(name="White water rafting", status="excluded", evidence="rafting excluded")])
    assert rm.match_activities(["River Rafting"], offer).result == M.MISMATCH


def test_silent_vendor_is_unknown_not_mismatch():
    assert rm.match_activities(["Local Sightseeing"], ExtractedOffer()).result == M.UNKNOWN


def test_custom_activity_matches_other_inclusions():
    offer = ExtractedOffer(other_inclusions=["Candlelight dinner on the beach"])
    assert rm.match_activities(["Candlelight Dinner"], offer).result == M.MATCH


def test_activity_partial_when_some_found():
    offer = ExtractedOffer(activities=[ActivityItem(name="Trekking", status="included", evidence="Trekking")])
    match = rm.match_activities(["Trekking", "Paragliding"], offer)
    assert match.result == M.PARTIAL and match.unverified is True


# -------------------------------------------------------------------- guide


def test_guide_language_match_and_mismatch():
    parsed = parse_special_requests("Guide: English Speaking Guide.")
    ok = ExtractedOffer(guide=ServiceItem(status="included", detail="English-speaking guide", evidence="x"))
    other = ExtractedOffer(guide=ServiceItem(status="included", detail="Local language guide", evidence="x"))
    vague = ExtractedOffer(guide=ServiceItem(status="included", detail="Tour guide", evidence="x"))
    assert rm.match_guide(parsed, ok).result == M.MATCH
    assert rm.match_guide(parsed, other).result == M.PARTIAL
    vague_match = rm.match_guide(parsed, vague)
    assert vague_match.result == M.PARTIAL and vague_match.unverified is True
    assert rm.match_guide(parsed, ExtractedOffer()).result == M.UNKNOWN


def test_optional_guide_is_not_a_requirement():
    assert rm.match_guide(parse_special_requests("Guide: Optional"), ExtractedOffer()) is None


# --------------------------------------------------------------- end to end


def test_match_offer_against_the_real_enquiry_shape():
    enquiry = EnquirySnapshot(
        id="e1",
        requirements=EnquiryRequirementsSnapshot(
            accommodation="3-Star Hotel",
            meals=["Breakfast"],
            transport="Tempo Traveller (12-Seater)",
            activities=["Scuba Diving"],
            special_requests="Guide: English Speaking Guide. aaaaaaaaa",
        ),
    )
    offer = _acc(4, "hotel")
    offer.transport = TransportItem(status="included", vehicle="AC sedan", is_private=True, evidence="x")
    offer.meals.breakfast = ServiceItem(status="included", evidence="x")
    offer.guide = ServiceItem(status="included", detail="English-speaking guide", evidence="x")
    normalized = NormalizedOffer(bid_id="b1", vendor_name="Coastal Trails", status=ExtractionStatus.OK, offer=offer)

    report = rm.match_offer(enquiry, normalized)
    by_req = {m.requirement: m.result for m in report.matches}
    assert by_req == {
        "accommodation": M.MATCH,
        "transport": M.MISMATCH,
        "meals": M.MATCH,
        "activities": M.UNKNOWN,
        "guide": M.MATCH,
    }
    assert report.not_evaluated == ["Special requests not checked automatically: aaaaaaaaa"]


def test_failed_extraction_yields_unknowns_and_a_note():
    enquiry = EnquirySnapshot(id="e1", requirements=EnquiryRequirementsSnapshot(transport="Private Cab"))
    normalized = NormalizedOffer(bid_id="b1", status=ExtractionStatus.FAILED)
    report = rm.match_offer(enquiry, normalized)
    assert report.matches[0].result == M.UNKNOWN
    assert any("extraction failed" in n for n in report.not_evaluated)