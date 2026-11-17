import re

# Strip common separators so 987-654-3210, 987 654 3210, +91 98765 43210 all unify.
_SEPARATORS = re.compile(r"[\s\-\(\)\.]")

# Indian mobile: optional 91 country code, then 10 digits starting 6-9.
_PHONE_RE = re.compile(r"(?<!\d)(?:91)?[6-9]\d{9}(?!\d)")


def detect_phone(text: str) -> bool:
    if not text:
        return False
    compact = _SEPARATORS.sub("", text)
    return bool(_PHONE_RE.search(compact))
