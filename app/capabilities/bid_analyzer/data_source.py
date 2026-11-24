"""Authorized live-data retrieval + defensive normalization for bid analysis.

Ownership rule: /api/enquiries/{id} is also readable by other roles, so the
enquiry must appear in the caller's own /api/enquiries/my list before any
bid data is fetched.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any
from datetime import datetime, UTC

from app.capabilities.bid_analyzer.schemas import (
    AnalysisInput,
    BidSnapshot,
    EnquiryRequirementsSnapshot,
    EnquirySnapshot,
    NegotiationEvent,
    VendorSnapshot,
)
from app.core.exceptions import SwabiAIError
from app.core.logging import get_logger
from app.swabi.read_transport import SwabiNotFoundError, get_json

log = get_logger(__name__)

_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
_BID_PAGE_SIZE = 50
_BID_MAX_PAGES = 4  # hard cap: 200 bids


# Only these are excluded. Active and accepted bids are shown; any other
# status (pending, countered, ...) is kept with a warning.

INELIGIBLE_BID_STATUSES = frozenset({"withdrawn", "rejected", "expired"})
_KNOWN_BID_STATUSES = INELIGIBLE_BID_STATUSES | {"active", "accepted"}

class EnquiryAccessError(SwabiAIError):
    """Enquiry does not exist or does not belong to the caller."""


class BidDataError(SwabiAIError):
    """Bid or enquiry data was missing or structurally unusable."""


# helpers


def _dec(value: Any) -> Decimal | None:
    if value is None or isinstance(value, bool) or value == "":
        return None
    try:
        out = Decimal(str(value))
    except InvalidOperation:
        return None
    return out if out.is_finite() and out >= 0 else None


def _int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _str_list(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        text = _str(item)
        if text and text.casefold() not in {o.casefold() for o in out}:
            out.append(text)
    return out


def _data(body: dict[str, Any]) -> Any:
    return body.get("data", body)


# normalizers


def normalize_enquiry(raw: dict[str, Any]) -> EnquirySnapshot:
    if isinstance(raw.get("enquiry"), dict):
        raw = raw["enquiry"]
    enquiry_id = _str(raw.get("id"))
    if not enquiry_id:
        raise BidDataError("enquiry response has no id")

    req = raw.get("requirements") if isinstance(raw.get("requirements"), dict) else {}
    breakdown = (
        raw.get("travelerBreakdown") if isinstance(raw.get("travelerBreakdown"), dict) else {}
    )
    budget_range = raw.get("budgetRange") if isinstance(raw.get("budgetRange"), dict) else {}
    budget = _dec(raw.get("budget"))
    return EnquirySnapshot(
        id=enquiry_id,
        title=_str(raw.get("title")),
        destination=_str(raw.get("destination")),
        duration=_str(raw.get("duration")),
        travel_date=_str(raw.get("travelDate")),
        travel_date_end=_str(raw.get("travelDateEnd")),
        status=_str(raw.get("status")),
        travelers=_int(raw.get("travelers")),
        adults=_int(breakdown.get("adults")),
        children=_int(breakdown.get("children")),
        seniors=_int(breakdown.get("seniors")),
        rooms=_int(breakdown.get("rooms")),
        budget=budget,
        budget_currency="INR" if budget is not None else None,
        budget_currency_assumed=True,
        budget_min=_dec(budget_range.get("min")),
        budget_max=_dec(budget_range.get("max")),
        requirements=EnquiryRequirementsSnapshot(
            accommodation=_str(req.get("accommodation")),
            meals=_str_list(req.get("meals")),
            transport=_str(req.get("transport")),
            activities=_str_list(req.get("activities")),
            special_requests=_str(req.get("specialRequests")),
        ),
        tags=_str_list(raw.get("tags")),
        additional_notes=_str(raw.get("additionalNotes")),
        description=_str(raw.get("description")),
    )


def _negotiation_history(value: Any) -> list[NegotiationEvent]:
    if not isinstance(value, list):
        return []
    # proposedById is deliberately dropped (internal identifier).
    return [
        NegotiationEvent(
            proposed_by=_str(item.get("proposedBy")),
            amount=_dec(item.get("amount")),
            status=_str(item.get("status")),
            created_at=_str(item.get("createdAt")),
        )
        for item in value
        if isinstance(item, dict)
    ]


def _validity_ended(valid_until: str | None, now: datetime) -> bool:
    """A bid is valid through the whole UTC day of validUntil."""
    if not valid_until:
        return False
    try:
        end = datetime.fromisoformat(valid_until.replace("Z", "+00:00"))
    except ValueError:
        return False
    return now.date() > end.date()


def normalize_bid(raw: dict[str, Any]) -> BidSnapshot | None:
    bid_id = _str(raw.get("id"))
    if not bid_id:
        return None
    agent = raw.get("agent") if isinstance(raw.get("agent"), dict) else {}
    currency = _str(raw.get("currency"))
    rating = _dec(agent.get("rating"))  # API returns a string such as "0.00"
    history = _negotiation_history(raw.get("negotiationHistory"))
    negotiation_status = _str(raw.get("negotiationStatus"))
    return BidSnapshot(
        id=bid_id,
        status=(_str(raw.get("status")) or "").casefold() or None,
        negotiation_status=negotiation_status,
        amount=_dec(raw.get("amount")),
        original_amount=_dec(raw.get("originalAmount")),
        negotiation_amount=_dec(raw.get("negotiationAmount")),
        negotiation_history=history,
        has_negotiation=bool(history)
        or (negotiation_status or "none").casefold() != "none",
        currency=(currency or "INR").upper(),
        currency_assumed=currency is None,  # CreateBidDto documents INR
        inclusions=_str_list(raw.get("inclusions")),
        exclusions=_str_list(raw.get("exclusions")),
        itinerary_text=_str(raw.get("proposalItinerary")),
        terms=_str_list(raw.get("termsAndConditions")),
        valid_until=_str(raw.get("validUntil")),
        has_pdf_itinerary=bool(_str(raw.get("pdfItineraryUrl"))),
        vendor=VendorSnapshot(
            vendor_id=_str(agent.get("id")),
            company_name=_str(agent.get("companyName")),
            is_verified=agent.get("isVerified")
            if isinstance(agent.get("isVerified"), bool)
            else None,
            rating=float(rating) if rating is not None else None,
        ),
        created_at=_str(raw.get("createdAt")),
    )


#  retrieval


def _verify_ownership(user_jwt: str, enquiry_id: str) -> None:
    data = _data(get_json("/enquiries/my", user_jwt=user_jwt))
    if isinstance(data, dict):
        data = data.get("enquiries", [])
    if not isinstance(data, list):
        raise BidDataError("/enquiries/my returned an unexpected shape")
    for item in data:
        if (
            isinstance(item, dict)
            and str(item.get("id", "")).casefold() == enquiry_id.casefold()
            and not item.get("isDeleted", False)
        ):
            return
    raise EnquiryAccessError("enquiry not found for this user")


def _fetch_bid_pages(user_jwt: str, enquiry_id: str) -> tuple[list[dict[str, Any]], int, list[str]]:
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    total = 0
    for page in range(_BID_MAX_PAGES):
        data = _data(
            get_json(
                f"/bids/enquiry/{enquiry_id}",
                user_jwt=user_jwt,
                params={"limit": _BID_PAGE_SIZE, "offset": page * _BID_PAGE_SIZE},
            )
        )
        if not isinstance(data, dict):
            raise BidDataError("bids response has an unexpected shape")
        batch = data.get("bids", [])
        if not isinstance(batch, list):
            raise BidDataError("bids response has no bids list")
        total = _int(data.get("total")) or max(total, len(rows) + len(batch))
        rows.extend(b for b in batch if isinstance(b, dict))
        if len(batch) < _BID_PAGE_SIZE or len(rows) >= total:
            break
    else:
        if len(rows) < total:
            warnings.append(
                f"Only the first {len(rows)} of {total} bids were analyzed (page cap reached)."
            )
    return rows, total, warnings


def fetch_analysis_input(
    user_jwt: str,
    enquiry_id: str,
    *,
    include_expired: bool = False,
    now: datetime | None = None,
) -> AnalysisInput:
    """Retrieve and normalize the caller's enquiry and its comparable bids."""
    now = now or datetime.now(UTC)
    if not _UUID_RE.match(enquiry_id or ""):
        raise EnquiryAccessError("invalid enquiry id")

    _verify_ownership(user_jwt, enquiry_id)

    try:
        enquiry = normalize_enquiry(_data(get_json(f"/enquiries/{enquiry_id}", user_jwt=user_jwt)))
    except SwabiNotFoundError as exc:
        raise EnquiryAccessError("enquiry not found for this user") from exc

    raw_bids, total, warnings = _fetch_bid_pages(user_jwt, enquiry_id)

    eligible: list[BidSnapshot] = []
    excluded: dict[str, str] = {}
    for raw in raw_bids:
        bid = normalize_bid(raw)
        if bid is None:
            warnings.append("A bid without an id was skipped.")
            continue
        if bid.status in INELIGIBLE_BID_STATUSES:
            excluded[bid.id] = f"status={bid.status}"
            continue
        if bid.status and bid.status not in _KNOWN_BID_STATUSES:
            warnings.append(f"Bid {bid.id} has unrecognized status '{bid.status}'; kept.")
        if bid.amount is None:
            excluded[bid.id] = "missing or invalid amount"
            continue
        bid.is_expired = _validity_ended(bid.valid_until, now)
        if bid.status != "accepted" and bid.is_expired and not include_expired:
            excluded[bid.id] = f"validity ended {(bid.valid_until or '')[:10]}"
            continue
        if bid.currency_assumed:
            warnings.append(f"Bid {bid.id} has no currency; INR assumed per the bid contract.")
        if bid.has_negotiation:
            warnings.append(
                f"Bid {bid.id} has negotiation activity ({bid.negotiation_status}); "
                "the vendor's bid amount is used, not any pending counter-offer."
            )
        eligible.append(bid)
        if enquiry.budget is not None:
            warnings.append("Enquiry budget currency is not provided by the API; INR assumed.")

    log.info(
        "bid_analysis_input_loaded",
        extra={
            "event": "bid_analysis_input_loaded",
            "bids_reported": total,
            "bids_eligible": len(eligible),
            "bids_excluded": len(excluded),
        },
    )
    return AnalysisInput(
        enquiry=enquiry,
        bids=eligible,
        excluded_bids=excluded,
        total_bids_reported=total,
        warnings=warnings,
    )