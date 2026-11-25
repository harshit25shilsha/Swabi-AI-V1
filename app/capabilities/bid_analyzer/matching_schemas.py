"""Requirement-matching result models (step 3.3)."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from app.capabilities.bid_analyzer.offer_schemas import ExtractionStatus


class MatchResult(StrEnum):
    MATCH = "match"
    PARTIAL = "partial"
    MISMATCH = "mismatch"
    UNKNOWN = "unknown"


class RequirementMatch(BaseModel):
    requirement: str  # accommodation | transport | meals | activities | guide
    customer_asked: str
    vendor_offers: str | None = None
    result: MatchResult
    # True when some part could not be verified because the vendor was silent.
    # Ranking should lower confidence for this, not treat it as a failure.
    unverified: bool = False
    reason: str
    evidence: list[str] = Field(default_factory=list)


class OfferMatchReport(BaseModel):
    bid_id: str
    vendor_name: str | None = None
    extraction_status: ExtractionStatus
    matches: list[RequirementMatch] = Field(default_factory=list)
    not_evaluated: list[str] = Field(default_factory=list)