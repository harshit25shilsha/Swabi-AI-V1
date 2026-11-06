import re

_TOKEN_RE = re.compile(r"[\w\u0900-\u097F]+", re.UNICODE)


def detect_high_digit_density(text: str) -> bool:
    """
    Structural detector: catches any message where digits are scattered
    across many tokens (mixed into words, embedded, punctuated apart).

    Signals:
      - 5+ mixed tokens (containing both letters and digits), OR
      - 6+ total digits when the message has 5+ alphabetic tokens
        (a normal chat message rarely has this many digits).

    Does not care HOW the digits are hidden — only that they are many
    and scattered. Routes to the LLM with a "look for encoding" hint.
    """
    if not text:
        return False

    tokens = _TOKEN_RE.findall(text)
    if len(tokens) < 4:
        return False

    mixed = 0
    total_digits = 0
    alpha_tokens = 0

    for t in tokens:
        has_d = any(c.isdigit() for c in t)
        has_a = any(c.isalpha() for c in t)
        if has_d and has_a:
            mixed += 1
        if has_a:
            alpha_tokens += 1
        total_digits += sum(c.isdigit() for c in t)

    if mixed >= 5:
        return True
    if total_digits >= 6 and alpha_tokens >= 5:
        return True
    return False
