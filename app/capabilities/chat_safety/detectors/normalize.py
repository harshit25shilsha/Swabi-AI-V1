import re
import unicodedata


def normalize(text: str) -> str:
    if not text:
        return ""
    # NFKC: normalize compatible characters (full-width digits, etc.)
    text = unicodedata.normalize("NFKC", text)
    # Collapse runs of whitespace into a single space
    text = re.sub(r"\s+", " ", text).strip()
    return text
