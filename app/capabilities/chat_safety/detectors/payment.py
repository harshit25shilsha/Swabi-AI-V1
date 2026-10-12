import re

# Common UPI handles: user@okhdfcbank, user@ybl, user@paytm, etc.
_UPI_RE = re.compile(
    r"\b[a-zA-Z0-9.\-_]{2,}@(?:okhdfcbank|okicici|oksbi|okaxis|ybl|paytm|apl|ibl|axl|upi)\b",
    re.IGNORECASE,
)


def detect_payment(text: str) -> bool:
    if not text:
        return False
    return bool(_UPI_RE.search(text))
