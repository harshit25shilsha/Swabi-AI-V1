import re

# Digit-words -> value (English, transliterated Hindi, Devanagari).
_VALUES = {
    "zero": 0,
    "shunya": 0,
    "sifar": 0,
    "शून्य": 0,
    "one": 1,
    "ek": 1,
    "एक": 1,
    "two": 2,
    "do": 2,
    "दो": 2,
    "three": 3,
    "teen": 3,
    "tin": 3,
    "तीन": 3,
    "four": 4,
    "char": 4,
    "chaar": 4,
    "चार": 4,
    "five": 5,
    "panch": 5,
    "paanch": 5,
    "पांच": 5,
    "पाँच": 5,
    "six": 6,
    "chhe": 6,
    "chhah": 6,
    "che": 6,
    "छह": 6,
    "seven": 7,
    "saat": 7,
    "sat": 7,
    "सात": 7,
    "eight": 8,
    "aath": 8,
    "ath": 8,
    "आठ": 8,
    "nine": 9,
    "nau": 9,
    "नौ": 9,
    "ten": 10,
    "das": 10,
    "dus": 10,
    "दस": 10,
}

# Multipliers pair with a digit-word to form one number ("teen sau" = 300).
# They are consumed inline and do NOT count as a piece on their own.
_MULTIPLIERS = {"sau", "hundred", "thousand", "hazaar", "score", "crore", "lakh"}

_PLUS = {"plus", "jod", "jodo", "add", "more", "zyada", "jyada", "जोड़"}
_MINUS = {"minus", "ghata", "ghatao", "less", "kam", "कम", "घटा"}

_WORD = r"[\w\u0900-\u097F]+"
_TOKEN_RE = re.compile(_WORD, re.UNICODE)

# Two or more real operations in one message indicates an encoded sequence.
_MIN_PIECES = 2


def detect_arithmetic_digits(text: str) -> bool:
    """
    Detect digits hidden as small arithmetic, e.g.
    "do plus do, phir teen minus one, aur end mein nine" -> 4, 2, 9.

    Only real operators count as pieces. "teen sau ya paanch sau rupaye"
    is a legitimate price range (300 or 500 rupees), not an encoding,
    because multipliers alone do not count.
    """
    if not text:
        return False

    tokens = [t.lower() for t in _TOKEN_RE.findall(text)]
    pieces = 0
    i = 0
    while i < len(tokens):
        # Consume "<digit-word> <multiplier>" as one token-pair, no piece.
        if tokens[i] in _VALUES and i + 1 < len(tokens) and tokens[i + 1] in _MULTIPLIERS:
            i += 2
            continue

        # Binary operator form: <digit> <op> [than] <digit>
        if i + 2 < len(tokens):
            a, op = tokens[i], tokens[i + 1]
            j = i + 2
            if j < len(tokens) and tokens[j] == "than":
                j += 1
            b = tokens[j] if j < len(tokens) else ""
            if a in _VALUES and b in _VALUES and (op in _PLUS or op in _MINUS):
                pieces += 1
                i = j + 1
                continue

        i += 1

    return pieces >= _MIN_PIECES
