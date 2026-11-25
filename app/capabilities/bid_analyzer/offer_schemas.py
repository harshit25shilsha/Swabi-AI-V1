"""Structured offer models (step 3.2).

`ExtractedOffer` is the LLM output shape; `NormalizedOffer` is what later
steps consume. Absence of information is always `unknown`.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class Availability(StrEnum):
    INCLUDED = "included"
    EXCLUDED = "excluded"
    UNKNOWN = "unknown"


class PriceScope(StrEnum):
    PER_PERSON = "per_person"
    TOTAL_GROUP = "total_group"
    UNKNOWN = "unknown"


class ServiceItem(BaseModel):
    status: Availability = Availability.UNKNOWN
    detail: str | None = Field(default=None, max_length=150)
    evidence: str | None = Field(default=None, max_length=200)


class AccommodationItem(ServiceItem):
    star_rating: int | None = Field(default=None, ge=1, le=5)
    property_type: str | None = Field(default=None, max_length=40)


class TransportItem(ServiceItem):
    is_private: bool | None = None
    vehicle: str | None = Field(default=None, max_length=60)


class MealsItem(BaseModel):
    breakfast: ServiceItem = Field(default_factory=ServiceItem)
    lunch: ServiceItem = Field(default_factory=ServiceItem)
    dinner: ServiceItem = Field(default_factory=ServiceItem)


class ActivityItem(ServiceItem):
    name: str = Field(default="", max_length=60)


class OfferConditions(BaseModel):
    price_scope: PriceScope = PriceScope.UNKNOWN
    taxes: ServiceItem = Field(default_factory=ServiceItem)
    advance_payment_percent: float | None = Field(default=None, ge=0, le=100)
    cancellation_terms: str | None = Field(default=None, max_length=200)
    payment_terms: str | None = Field(default=None, max_length=200)


class ExtractedOffer(BaseModel):
    accommodation: AccommodationItem = Field(default_factory=AccommodationItem)
    transport: TransportItem = Field(default_factory=TransportItem)
    meals: MealsItem = Field(default_factory=MealsItem)
    guide: ServiceItem = Field(default_factory=ServiceItem)
    flights: ServiceItem = Field(default_factory=ServiceItem)
    activities: list[ActivityItem] = Field(default_factory=list, max_length=15)
    other_inclusions: list[str] = Field(default_factory=list, max_length=10)
    other_exclusions: list[str] = Field(default_factory=list, max_length=10)
    conditions: OfferConditions = Field(default_factory=OfferConditions)
    unclear_points: list[str] = Field(default_factory=list, max_length=8)


class ExtractionStatus(StrEnum):
    OK = "ok"
    NO_CONTENT = "no_content"  # vendor gave no text; LLM not called
    FAILED = "failed"  # LLM/validation failure; everything unknown


class NormalizedOffer(BaseModel):
    bid_id: str
    vendor_name: str | None = None
    status: ExtractionStatus
    offer: ExtractedOffer = Field(default_factory=ExtractedOffer)
    notes: list[str] = Field(default_factory=list)
    has_pdf_itinerary: bool = False
    prompt_version: str = ""