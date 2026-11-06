import re

_URL_RE = re.compile(r"(?:https?://|www\.)[^\s]+", re.IGNORECASE)

# Bare domains with a common TLD, optionally a path.
_DOMAIN_RE = re.compile(
    r"\b[a-zA-Z0-9\-]+\.(?:com|net|org|io|co|in|me|app|dev|xyz|info|biz|tv|ly|link|site|online)(?:/[^\s]*)?",
    re.IGNORECASE,
)


def detect_url(text: str) -> bool:
    if not text:
        return False
    return bool(_URL_RE.search(text) or _DOMAIN_RE.search(text))
