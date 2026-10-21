"""Filters applied to the model's answer before returning it.

Two distinct concerns:

1. Internal paths — the model must never quote operational URLs
   (/admin/..., /payments/...) to a customer or vendor.
2. Personal financial promises — "you will get a refund of X" is a
   commitment the AI cannot make. Policy descriptions ("customers get
   100% refund if...") are allowed; personal promises are not.

Both checks are conservative and downgrade the response to
insufficient_context if they fire. They are tunable.
"""
from __future__ import annotations

import re

_FORBIDDEN_PATH_RE = re.compile(
    r"(?<![\w/])/(admin|payments|escrow|devices|internal)(?:/[\w\-/]*)?",
    re.IGNORECASE,
)

# "You will get a refund", "you will be credited X", "you will receive Y"
_FINANCIAL_PROMISE_RE = re.compile(
    r"\byou(?:\s+will|'ll)\s+"
    r"(?:"
    # "get a refund", "receive your money", "get the payment"
    r"(?:get|receive)\s+(?:a\s+|your\s+|the\s+)?"
    r"(?:refund|money|payment|amount|compensation|reimbursement|credit)"
    r"|"
    # "get refunded", "receive credited" (uncommon, kept for symmetry)
    r"(?:get|receive)\s+(?:refunded|credited|paid|reimbursed|compensated)"
    r"|"
    # "be refunded", "be credited", "be paid"
    r"be\s+(?:refunded|credited|paid|reimbursed|compensated)"
    r")",
    re.IGNORECASE,
)

def strip_forbidden_paths(answer: str) -> tuple[str, list[str]]:
    """Redact internal paths. Returns (clean_answer, matched_paths)."""
    matches = _FORBIDDEN_PATH_RE.findall(answer)
    if not matches:
        return answer, []

    clean = _FORBIDDEN_PATH_RE.sub("[internal path redacted]", answer)
    return clean, [f"/{m}" for m in matches]


def has_financial_promise(answer: str) -> bool:
    """True if the answer makes a personal financial commitment."""
    return bool(_FINANCIAL_PROMISE_RE.search(answer))
