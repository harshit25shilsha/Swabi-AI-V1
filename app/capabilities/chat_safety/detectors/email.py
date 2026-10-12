import re

_EMAIL_RE = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
    re.IGNORECASE,
)

_OBFUSCATED_EMAIL_RE = re.compile(
    r"[a-zA-Z0-9._%+\-]+\s*(?:\[at\]|\(at\)|\bat\b)\s*"
    r"[a-zA-Z0-9.\-]+\s*(?:\[dot\]|\(dot\)|\bdot\b)\s*"
    r"[a-zA-Z]{2,}",
    re.IGNORECASE,
)


def detect_email(text: str) -> bool:
    if not text:
        return False
    return bool(_EMAIL_RE.search(text) or _OBFUSCATED_EMAIL_RE.search(text))
