"""Run the bid-offer extractor against Groq with representative vendor text.

Usage:
    python -m scripts.try_offer_extraction
    python -m scripts.try_offer_extraction --case all

Requires a valid ``GROQ_API_KEY`` in the repository's .env file.  This is a
live smoke test: it makes one provider request per selected case.
"""

from __future__ import annotations

import argparse
import json

from app.capabilities.bid_analyzer.offer_extractor import extract_offer
from app.capabilities.bid_analyzer.schemas import BidSnapshot, VendorSnapshot

CASES: dict[str, BidSnapshot] = {
    "goa": BidSnapshot(
        id="live-smoke-goa",
        vendor=VendorSnapshot(company_name="Coastal Trails Travel"),
        inclusions=[
            "3 nights in a 4-star hotel in North Goa on double occupancy.",
            "Daily breakfast at the hotel.",
            "Private AC sedan for airport transfers and local sightseeing.",
            "English-speaking guide for the Old Goa heritage tour.",
        ],
        exclusions=[
            "Airfare is excluded from this package.",
            "Lunch and dinner are not included.",
            "Personal expenses and travel insurance are excluded.",
        ],
        terms=[
            "Package price is per person, based on two adults sharing a room.",
            "50% advance payment is required to confirm the booking.",
            "Cancellation within 7 days of arrival is non-refundable.",
        ],
        itinerary_text=(
            "Day 1: Airport pickup and hotel check-in. Day 2: Old Goa heritage tour. "
            "Day 3: Free day at leisure. Day 4: Hotel checkout and airport drop."
        ),
    ),
    "manali": BidSnapshot(
        id="live-smoke-manali",
        vendor=VendorSnapshot(company_name="Himalayan Route Co."),
        inclusions=[
            "2 nights in a deluxe hotel with breakfast and dinner.",
            "Shared Volvo bus from Delhi to Manali and return.",
            "Solang Valley sightseeing, subject to weather conditions.",
        ],
        exclusions=[
            "Lunch is not included.",
            "Rohtang Pass permit and adventure activities are excluded.",
        ],
        terms=["All applicable hotel taxes are included in the quoted price."],
        itinerary_text=(
            "Day 1: Overnight Volvo from Delhi. Day 2: Check in at Manali hotel. "
            "Day 3: Solang Valley excursion. Day 4: Return to Delhi by Volvo."
        ),
    ),
    "kerala": BidSnapshot(
        id="live-smoke-kerala",
        vendor=VendorSnapshot(company_name="Backwater Holidays"),
        inclusions=[
            "One night on a private houseboat with all meals.",
            "Munnar tea-garden visit and Alleppey backwater cruise.",
        ],
        exclusions=["Flights are not part of this offer."],
        terms=["Balance payment is due at hotel check-in."],
        itinerary_text=(
            "Day 1: Arrive in Kochi. Day 2: Drive to Munnar. "
            "Day 3: Transfer to Alleppey houseboat. Day 4: Departure from Kochi."
        ),
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=[*CASES, "all"], default="goa")
    args = parser.parse_args()

    selected = CASES.values() if args.case == "all" else [CASES[args.case]]
    for bid in selected:
        result = extract_offer(bid)
        print(f"\n--- {bid.id} ({bid.vendor.company_name}) ---")
        print(json.dumps(result.model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    main()
