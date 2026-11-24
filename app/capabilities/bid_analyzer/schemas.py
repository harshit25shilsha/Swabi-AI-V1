"""Normalized source models for the Bid Analyzer.

These describe verified data read from NestJS. Unknown values are None or
empty; nothing is defaulted into a factual claim.
"""

from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel, Field


class EnquiryRequirementsSnapshot(BaseModel):
    accommodation: str | None = None
    meals: list[str] = Field(default_factory=list)
    transport: str | None = None
    activities: list[str] = Field(default_factory=list)
    # NOTE: the guide preference is embedded here as free text
    # ("Guide: English Speaking Guide. ...") - extracted in step 3.2/3.3.
    special_requests: str | None = None


class EnquirySnapshot(BaseModel):
    id: str
    title: str | None = None
    destination: str | None = None
    duration: str | None = None  # e.g. "2 Nights / 3 Days" (parsed in 3.4)
    travel_date: str | None = None
    travel_date_end: str | None = None
    status: str | None = None
    travelers: int | None = None
    adults: int | None = None
    children: int | None = None
    seniors: int | None = None
    rooms: int | None = None
    budget: Decimal | None = None
    budget_currency: str | None = None  # "INR" whenever a budget is present
    budget_currency_assumed: bool = True  # swagger doesn't state it reliably
    budget_min: Decimal | None = None
    budget_max: Decimal | None = None
    requirements: EnquiryRequirementsSnapshot = Field(
        default_factory=EnquiryRequirementsSnapshot
    )
    tags: list[str] = Field(default_factory=list)
    additional_notes: str | None = None
    description: str | None = None


class VendorSnapshot(BaseModel):
    vendor_id: str | None = None
    company_name: str | None = None
    is_verified: bool | None = None  # not returned by the bids endpoint
    rating: float | None = None


class NegotiationEvent(BaseModel):
    proposed_by: str | None = None  # "customer" | "vendor"
    amount: Decimal | None = None
    status: str | None = None
    created_at: str | None = None


class BidSnapshot(BaseModel):
    id: str
    status: str | None = None
    negotiation_status: str | None = None
    # `amount` is the vendor's offered price. A pending counter-offer is NOT
    # an agreed price and is never used for comparison.
    amount: Decimal | None = None
    original_amount: Decimal | None = None
    negotiation_amount: Decimal | None = None
    negotiation_history: list[NegotiationEvent] = Field(default_factory=list)
    has_negotiation: bool = False
    currency: str | None = None
    currency_assumed: bool = False
    inclusions: list[str] = Field(default_factory=list)
    exclusions: list[str] = Field(default_factory=list)
    itinerary_text: str | None = None
    terms: list[str] = Field(default_factory=list)
    valid_until: str | None = None
    is_expired: bool = False
    has_pdf_itinerary: bool = False  # PDF content is NOT analyzed
    vendor: VendorSnapshot = Field(default_factory=VendorSnapshot)
    created_at: str | None = None


class AnalysisInput(BaseModel):
    enquiry: EnquirySnapshot
    bids: list[BidSnapshot]
    excluded_bids: dict[str, str] = Field(default_factory=dict)  # bid_id -> reason
    total_bids_reported: int = 0
    warnings: list[str] = Field(default_factory=list)