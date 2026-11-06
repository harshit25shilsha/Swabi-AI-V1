"""Request, response, and extraction schemas for Enquiry Understanding.

The extraction shape is a superset of:
    * the live Enquiry form (13 fields, source of truth for the UI)
    * the CreateEnquiryDto (NestJS Swagger contract)
    * the Swabi BRD CUST-ENQ-* fields (for future UI additions)

Every field is optional. Missing fields are ``None`` or empty lists,
never fabricated defaults.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# Sub-objects


class DateFlexibility(BaseModel):
    """Mirrors CreateEnquiryDto.dateFlexibility"""

    is_flexible: bool | None = None
    window_days: int | None = Field(default=None, ge=0, le=30)
    note: str | None = None


class EnquiryBudget(BaseModel):
    """Mirrors the DTO's numeric ``budget`` plus a currency.

    The live form labels budget as per-person; the DTO stores a single
    number. We extract the raw amount and a per-person flag so the
    frontend can compute either.
    """

    amount: float | None = Field(default=None, ge=0)
    currency: str | None = None  # ISO 4217 (e.g. "INR")
    is_per_person: bool | None = None


class Child(BaseModel):
    """BRD CUST-ENQ-11. UI does not collect this today."""

    age: int | None = Field(default=None, ge=0, le=17)


class EnquiryRequirements(BaseModel):
    """Mirrors CreateEnquiryDto.requirements."""

    accommodation: str | None = None  # DTO: single string
    meals: list[str] = Field(default_factory=list)
    transport: str | None = None  # DTO: single string
    activities: list[str] = Field(default_factory=list)
    tour_guide: str | None = None  # no DTO slot; frontend maps
    special_requests: str | None = None


# Extraction - internal, structured LLM output


class EnquiryExtraction(BaseModel):
    # Identifying
    title: str | None = None  # DTO requires; derive if absent
    description: str | None = None  # UI: Trip Description (min 10)
    destination: str | None = None

    # When
    travel_date: str | None = None  # format: dd/mm/yy
    travel_date_end: str | None = None  # format: dd/mm/yy
    date_flexibility: DateFlexibility | None = None
    duration_days: int | None = Field(default=None, ge=0, le=365)
    duration_nights: int | None = Field(default=None, ge=0, le=365)

    # Who / how many
    travelers: int | None = Field(default=None, ge=1, le=50)
    adults: int | None = Field(default=None, ge=0, le=50)  # BRD
    children: list[Child] = Field(default_factory=list)  # BRD

    # Money
    budget: EnquiryBudget | None = None
    budget_range_min: float | None = Field(default=None, ge=0)  # DTO
    budget_range_max: float | None = Field(default=None, ge=0)  # DTO

    # Requirements (mapped to the DTO's nested object)
    requirements: EnquiryRequirements = Field(default_factory=EnquiryRequirements)

    # BRD superset (dormant until the UI exposes these fields)
    rooms: int | None = Field(default=None, ge=0, le=50)
    property_type: str | None = None
    trip_type: str | None = None
    flags: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


# API - request


class EnquiryExtractRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)

    # Optional hints from NestJS. Not required for extraction.
    user_id: str | None = None
    currency_hint: str | None = None


# API - Response


class ClarificationQuestion(BaseModel):
    field: str
    question: str


class EnquiryUnderstandResponse(BaseModel):
    extraction: EnquiryExtraction
    missing_fields: list[str] = Field(default_factory=list)
    clarifying_questions: list[ClarificationQuestion] = Field(default_factory=list)

    # 0.0 = nothing extracted, 1.0 = all required fields present.
    completeness: float = Field(..., ge=0.0, le=1.0)

    # Metadata, same shape as Chat Safety.
    provider: str = ""
    prompt_version: str = ""
    request_id: str = ""
