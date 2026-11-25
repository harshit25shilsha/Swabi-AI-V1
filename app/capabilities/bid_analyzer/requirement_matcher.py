"""Step 3.3: deterministic comparison of an extracted offer with the enquiry.

No LLM. Absence of vendor information is UNKNOWN, never a mismatch.
"""

from __future__ import annotations

import re

from app.capabilities.bid_analyzer.matching_schemas import (
    MatchResult,
    OfferMatchReport,
    RequirementMatch,
)
from app.capabilities.bid_analyzer.offer_schemas import (
    Availability,
    ExtractedOffer,
    ExtractionStatus,
    NormalizedOffer,
    ServiceItem,
)
from app.capabilities.bid_analyzer.requirements import (
    ParsedSpecialRequests,
    parse_special_requests,
)
from app.capabilities.bid_analyzer.schemas import EnquirySnapshot

M = MatchResult
Part = tuple[MatchResult, str]


# ------------------------------------------------------------- combining


def _combine(parts: list[Part]) -> tuple[MatchResult, bool, str]:
    results = [r for r, _ in parts]
    reason = "; ".join(t for _, t in parts)
    unverified = M.UNKNOWN in results
    if M.MISMATCH in results:
        return M.MISMATCH, unverified, reason
    if M.PARTIAL in results:
        return M.PARTIAL, unverified, reason
    if all(r == M.MATCH for r in results):
        return M.MATCH, False, reason
    if all(r == M.UNKNOWN for r in results):
        return M.UNKNOWN, True, reason
    return M.PARTIAL, True, reason  # some verified, some not stated


def _combine_counts(
    ok: list[str], missing: list[str], unverified: list[str]
) -> tuple[MatchResult, bool, str]:
    reason = "; ".join(
        text
        for text in (
            f"included: {', '.join(ok)}" if ok else "",
            f"not included: {', '.join(missing)}" if missing else "",
            f"not stated: {', '.join(unverified)}" if unverified else "",
        )
        if text
    )
    if missing and not ok and not unverified:
        result = M.MISMATCH
    elif missing:
        result = M.PARTIAL
    elif ok and unverified:
        result = M.PARTIAL
    elif ok:
        result = M.MATCH
    else:
        result = M.UNKNOWN
    return result, bool(unverified), reason


def _make(
    requirement: str,
    asked: str,
    offers: str | None,
    outcome: tuple[MatchResult, bool, str],
    evidence: list[str | None],
) -> RequirementMatch:
    result, unverified, reason = outcome
    return RequirementMatch(
        requirement=requirement,
        customer_asked=asked,
        vendor_offers=offers or None,
        result=result,
        unverified=unverified,
        reason=reason,
        evidence=[e for e in evidence if e],
    )


def _text(item: ServiceItem, *extra: str | None) -> str:
    return " ".join(p for p in (*extra, item.detail, item.evidence) if p)


# ---------------------------------------------------------- accommodation

_STAR_RE = re.compile(r"\b([1-5])\s*[-\s]?\s*(?:stars?|\*|★)", re.IGNORECASE)
_TYPE_WORDS = {
    "houseboat": ("houseboat", "house boat"),
    "resort": ("resort",),
    "hotel": ("hotel",),
    "homestay": ("homestay", "home stay"),
    "villa": ("villa",),
    "guesthouse": ("guest house", "guesthouse"),
    "apartment": ("apartment",),
}
_CLOSE_TYPES = {"hotel", "resort"}


def _types(text: str) -> set[str]:
    low = text.casefold()
    return {t for t, words in _TYPE_WORDS.items() if any(w in low for w in words)}


def match_accommodation(asked: str | None, offer: ExtractedOffer) -> RequirementMatch | None:
    if not asked or not asked.strip():
        return None
    item = offer.accommodation
    stars = f"{item.star_rating}-star" if item.star_rating else None
    offers = ", ".join(p for p in (stars, item.property_type, item.detail) if p)

    if item.status == Availability.EXCLUDED:
        return _make("accommodation", asked, offers, (M.MISMATCH, False, "accommodation is excluded"), [item.evidence])
    if item.status == Availability.UNKNOWN:
        return _make("accommodation", asked, None, (M.UNKNOWN, True, "accommodation not stated"), [])

    vendor_text = _text(item, item.property_type)
    req_stars = {int(s) for s in _STAR_RE.findall(asked)}
    req_types = _types(asked)
    vendor_types = _types(vendor_text)
    parts: list[Part] = []

    if req_types:
        if not vendor_types:
            parts.append((M.UNKNOWN, "property type not stated"))
        elif vendor_types & req_types:
            parts.append((M.MATCH, "property type matches"))
        elif vendor_types <= _CLOSE_TYPES and req_types <= _CLOSE_TYPES:
            parts.append((M.PARTIAL, f"asked {'/'.join(sorted(req_types))}, offered {'/'.join(sorted(vendor_types))}"))
        else:
            parts.append((M.MISMATCH, f"asked {'/'.join(sorted(req_types))}, offered {'/'.join(sorted(vendor_types))}"))

    if req_stars:
        lowest = min(req_stars)
        have = item.star_rating
        if have is None:
            parts.append((M.UNKNOWN, "star rating not stated"))
        elif have >= lowest:
            parts.append((M.MATCH, f"{have}-star meets the requested {lowest}-star"))
        elif have == lowest - 1:
            parts.append((M.PARTIAL, f"{have}-star is one below the requested {lowest}-star"))
        else:
            parts.append((M.MISMATCH, f"{have}-star is well below the requested {lowest}-star"))

    if not parts:
        parts.append((M.UNKNOWN, "requirement too generic to compare"))
    return _make("accommodation", asked, offers, _combine(parts), [item.evidence])


# -------------------------------------------------------------- transport

_VEHICLE_WORDS = (
    ("tempo_traveller", ("tempo", "traveller", "traveler")),
    ("coach", ("coach", "bus", "volvo")),
    ("suv", ("suv", "innova", "ertiga", "scorpio", "xuv", "fortuner")),
    ("sedan", ("sedan", "dzire", "etios", "verna")),
    ("cab", ("cab", "taxi", "car")),
)
_CAPACITY = {"cab": 1, "sedan": 1, "suv": 2, "tempo_traveller": 3, "coach": 4}
_COMPARABLE = {"sedan", "suv", "tempo_traveller"}  # requested classes we compare


def _vehicle_class(text: str) -> str | None:
    low = text.casefold()
    for name, words in _VEHICLE_WORDS:
        if any(re.search(rf"\b{re.escape(w)}", low) for w in words):
            return name
    return None


def _privacy(text: str) -> bool | None:
    low = text.casefold()
    if "shared" in low or "sharing" in low:
        return False
    if "private" in low or "dedicated" in low:
        return True
    return None


def match_transport(asked: str | None, offer: ExtractedOffer) -> RequirementMatch | None:
    if not asked or not asked.strip():
        return None
    item = offer.transport
    vendor_text = _text(item, item.vehicle)
    offers = ", ".join(p for p in (item.vehicle, item.detail) if p)

    if item.status == Availability.EXCLUDED:
        return _make("transport", asked, offers, (M.MISMATCH, False, "transport is excluded"), [item.evidence])
    if item.status == Availability.UNKNOWN:
        return _make("transport", asked, None, (M.UNKNOWN, True, "transport not stated"), [])

    req_class = _vehicle_class(asked)
    ven_class = _vehicle_class(vendor_text)
    req_private = _privacy(asked)
    ven_private = item.is_private if item.is_private is not None else _privacy(vendor_text)
    parts: list[Part] = []

    if req_private is True:
        if ven_private is True:
            parts.append((M.MATCH, "private transport"))
        elif ven_private is False:
            parts.append((M.MISMATCH, "shared transport offered, private requested"))
        else:
            parts.append((M.UNKNOWN, "private/shared not stated"))
    elif req_private is False and ven_private is not None:
        parts.append((M.MATCH, "shared transport acceptable"))

    if req_class in _COMPARABLE:
        if ven_class is None:
            parts.append((M.UNKNOWN, "vehicle type not stated"))
        else:
            diff = _CAPACITY[ven_class] - _CAPACITY[req_class]
            if diff >= 0:
                parts.append((M.MATCH, f"{ven_class.replace('_', ' ')} meets requested {req_class.replace('_', ' ')}"))
            elif diff == -1:
                parts.append((M.PARTIAL, f"smaller vehicle: offered {ven_class.replace('_', ' ')}, requested {req_class.replace('_', ' ')}"))
            else:
                parts.append((M.MISMATCH, f"much smaller vehicle: offered {ven_class.replace('_', ' ')}, requested {req_class.replace('_', ' ')}"))

    if "airport" in asked.casefold() and "airport" not in vendor_text.casefold():
        parts.append((M.UNKNOWN, "airport transfer not mentioned"))

    if not parts:
        parts.append((M.MATCH, "transport is included"))
    outcome = _combine(parts)
    # The vendor has confirmed transport is included, but may omit requested
    # qualifiers such as "private" or "sedan". That is partial evidence,
    # rather than a wholly unknown transport offer.
    if outcome[0] == M.UNKNOWN:
        outcome = (M.PARTIAL, True, outcome[2])
    return _make("transport", asked, offers, outcome, [item.evidence])


# ------------------------------------------------------------------ meals

_MEALS = ("breakfast", "lunch", "dinner")


def _requested_meals(values: list[str]) -> list[str]:
    wanted: set[str] = set()
    for value in values:
        low = value.casefold()
        if "all" in low or "full board" in low:
            wanted.update(_MEALS)
        wanted.update(m for m in _MEALS if m in low)
    return [m for m in _MEALS if m in wanted]


def match_meals(asked: list[str], offer: ExtractedOffer) -> RequirementMatch | None:
    wanted = _requested_meals(asked)
    if not wanted:
        return None
    ok: list[str] = []
    missing: list[str] = []
    unverified: list[str] = []
    evidence: list[str | None] = []
    for meal in wanted:
        item: ServiceItem = getattr(offer.meals, meal)
        evidence.append(item.evidence)
        {Availability.INCLUDED: ok, Availability.EXCLUDED: missing}.get(item.status, unverified).append(meal)
    offers = ", ".join(m for m in _MEALS if getattr(offer.meals, m).status == Availability.INCLUDED)
    return _make("meals", ", ".join(wanted), offers, _combine_counts(ok, missing, unverified), evidence)


# ------------------------------------------------------------- activities

_STOP = {"ride", "stay", "tour", "the", "a", "an", "and", "of", "in", "pass", "visit", "trip", "session", "experience"}
# Stemmed keys for the known activity picker; any key found in the vendor text matches.
_ACTIVITY_KEYS = {
    "shikara ride": ("shikara",),
    "gulmarg gondola": ("gondola",),
    "houseboat stay": ("houseboat",),
    "skiing": ("ski",),
    "trekking": ("trek",),
    "paragliding": ("paraglid",),
    "local sightseeing": ("sightsee",),
    "river rafting": ("raft",),
    "camel safari": ("camel",),
    "scuba diving": ("scuba", "div"),
}


def _stem(word: str) -> str:
    for suffix in ("ing", "es", "s", "e"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)]
            break
    if len(word) >= 2 and word[-1] == word[-2] and word[-1] not in "aeiou":
        word = word[:-1]
    return word


def _stems(text: str) -> set[str]:
    return {_stem(w) for w in re.findall(r"[a-z]+", text.casefold()) if w not in _STOP}


def _activity_matches(requested: str, candidate: str) -> bool:
    key = _ACTIVITY_KEYS.get(requested.casefold().strip())
    stems = _stems(candidate)
    if key is not None:
        return any(k in stems for k in key)
    wanted = _stems(requested)
    return bool(wanted) and wanted <= stems


def match_activities(asked: list[str], offer: ExtractedOffer) -> RequirementMatch | None:
    asked = [a for a in asked if a and a.strip()]
    if not asked:
        return None
    candidates: list[tuple[str, Availability, str | None]] = [
        (f"{a.name} {a.evidence or ''}", a.status, a.evidence) for a in offer.activities
    ]
    candidates += [(t, Availability.INCLUDED, t) for t in offer.other_inclusions]
    candidates += [(t, Availability.EXCLUDED, t) for t in offer.other_exclusions]

    ok: list[str] = []
    missing: list[str] = []
    unverified: list[str] = []
    evidence: list[str | None] = []
    for activity in asked:
        found = [c for c in candidates if _activity_matches(activity, c[0])]
        if any(c[1] == Availability.INCLUDED for c in found):
            ok.append(activity)
            evidence.append(next(c[2] for c in found if c[1] == Availability.INCLUDED))
        elif found:
            missing.append(activity)
            evidence.append(found[0][2])
        else:
            unverified.append(activity)
    offers = ", ".join(a.name for a in offer.activities if a.status == Availability.INCLUDED)
    return _make("activities", ", ".join(asked), offers, _combine_counts(ok, missing, unverified), evidence)


# ------------------------------------------------------------------ guide

_GUIDE_WORDS = {
    "english": ("english",),
    "local": ("local", "hindi", "regional", "native"),
    "photography": ("photograph",),
    "adventure": ("adventure",),
}


def _guide_kinds(text: str) -> set[str]:
    low = text.casefold()
    return {k for k, words in _GUIDE_WORDS.items() if any(w in low for w in words)}


def match_guide(parsed: ParsedSpecialRequests, offer: ExtractedOffer) -> RequirementMatch | None:
    if not parsed.guide_required or not parsed.guide_preference:
        return None
    asked = parsed.guide_preference
    item = offer.guide
    offers = item.detail or None

    if item.status == Availability.EXCLUDED:
        return _make("guide", asked, offers, (M.MISMATCH, False, "guide is excluded"), [item.evidence])
    if item.status == Availability.UNKNOWN:
        return _make("guide", asked, None, (M.UNKNOWN, True, "guide not stated"), [])

    wanted = _guide_kinds(asked)
    have = _guide_kinds(_text(item))
    if not wanted or wanted & have:
        outcome = (M.MATCH, False, "guide included")
    elif have:
        outcome = (M.PARTIAL, False, f"guide offered is {'/'.join(sorted(have))}; requested {asked}")
    else:
        outcome = (M.PARTIAL, True, "guide included; language/type not stated")
    return _make("guide", asked, offers, outcome, [item.evidence])


# ------------------------------------------------------------- public API


def match_offer(enquiry: EnquirySnapshot, normalized: NormalizedOffer) -> OfferMatchReport:
    req = enquiry.requirements
    parsed = parse_special_requests(req.special_requests)
    offer = normalized.offer

    candidates = (
        match_accommodation(req.accommodation, offer),
        match_transport(req.transport, offer),
        match_meals(req.meals, offer),
        match_activities(req.activities, offer),
        match_guide(parsed, offer),
    )
    not_evaluated: list[str] = []
    if parsed.other_requests:
        not_evaluated.append(f"Special requests not checked automatically: {parsed.other_requests}")
    if normalized.status != ExtractionStatus.OK:
        not_evaluated.append(f"Offer details unavailable (extraction {normalized.status.value}).")

    return OfferMatchReport(
        bid_id=normalized.bid_id,
        vendor_name=normalized.vendor_name,
        extraction_status=normalized.status,
        matches=[m for m in candidates if m is not None],
        not_evaluated=not_evaluated,
    )


def match_offers(enquiry: EnquirySnapshot, offers: list[NormalizedOffer]) -> list[OfferMatchReport]:
    return [match_offer(enquiry, o) for o in offers]
