import re

# Relational / arithmetic phrases commonly used to encode digits.
_ENCODING_PHRASES = [
    # "the number after five", "the number before ten"
    r"\bnumber\s+(?:after|before|between)\b",
    # "one less than", "two more than", "three less than"
    r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten)\s+"
    r"(?:less|more)\s+than\b",
    # "half of twelve", "double of five"
    r"\b(?:half|double|twice|thrice)\s+(?:of\s+)?\w+",
    # "the digit between four and six"
    r"\bdigit\s+between\b",
    # "next after", "previous before"
    r"\b(?:next|previous)\s+(?:number|digit|after|before)\b",
]

# Sequence connectors — a single phrase isn't enough, we need sequence structure.
_SEQUENCE_CONNECTORS = [
    r"\bthen\b",
    r"\bfollowed\s+by\b",
    r"\bnext\b",
    r"\bafter\s+that\b",
    r"\band\s+then\b",
    r",",
]

_ENCODING_RE = re.compile("|".join(_ENCODING_PHRASES), re.IGNORECASE)
_SEQUENCE_RE = re.compile("|".join(_SEQUENCE_CONNECTORS), re.IGNORECASE)


def detect_encoded_digits(text: str) -> bool:
    """
    Detect the *shape* of an arithmetic/relational digit encoding, e.g.
    'start with the number after five, then two less than ten, then twelve'.

    Returns True only when BOTH:
      - at least one encoding phrase is present, AND
      - at least one sequence connector is present.

    This avoids flagging a single innocent relational phrase in normal chat.
    """
    if not text:
        return False
    return bool(_ENCODING_RE.search(text)) and bool(_SEQUENCE_RE.search(text))
