"""Deterministic parsing of requirements the enquiry stores as free text."""

from __future__ import annotations

import re
from dataclasses import dataclass

_CANONICAL_GUIDES = (
    "English Speaking Guide",
    "Local Language Guide",
    "Photography Guide",
    "Adventure Guide",
    "Optional",
)
_GUIDE_RE = re.compile(r"\bguide\s*:\s*([^.;\n]+)[.;]?", re.IGNORECASE)


@dataclass(frozen=True)
class ParsedSpecialRequests:
    guide_preference: str | None  # None when absent
    guide_required: bool  # False when absent or "Optional"
    other_requests: str | None


def parse_special_requests(special_requests: str | None) -> ParsedSpecialRequests:
    if not special_requests or not special_requests.strip():
        return ParsedSpecialRequests(None, False, None)

    match = _GUIDE_RE.search(special_requests)
    if match is None:
        return ParsedSpecialRequests(None, False, special_requests.strip())

    raw = match.group(1).strip()
    canonical = next((g for g in _CANONICAL_GUIDES if g.casefold() == raw.casefold()), None)
    guide = canonical or raw
    remaining = (special_requests[: match.start()] + " " + special_requests[match.end():]).strip(" .;\n")
    return ParsedSpecialRequests(
        guide_preference=guide,
        guide_required=guide.casefold() != "optional",
        other_requests=re.sub(r"\s+", " ", remaining) or None,
    )