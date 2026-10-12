import re

# Digit words in English, transliterated Hindi, and Devanagari.
# NOTE: no ("no" in English), tera (Hindi "your"), sau/score/hazaar
# (multipliers) have been removed — they caused false positives.
_DIGIT_WORDS = {
    # English 0-19 and tens
    "zero", "one", "two", "three", "four", "five", "six", "seven",
    "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
    "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
    "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
    "eighty", "ninety",
    # Transliterated Hindi
    "shunya", "sifar",
    "ek", "do", "teen", "tin", "char", "chaar", "panch", "paanch",
    "chhe", "chhah", "che", "saat", "sat", "aath", "ath",
    "nau", "das", "dus",
    "gyarah", "gyara", "barah", "barha",
    "terah", "terha", "chaudah", "chauda",
    "pandrah", "pandra", "solah", "sola",
    "satrah", "satra", "atharah", "athara",
    "unnis", "unnees", "bees",
    # Devanagari
    "शून्य", "सिफ़र", "सिफर",
    "एक", "दो", "तीन", "चार", "पांच", "पाँच",
    "छह", "छः", "सात", "आठ", "नौ", "दस",
    "ग्यारह", "बारह", "तेरह", "चौदह", "पंद्रह",
    "सोलह", "सत्रह", "अठारह", "उन्नीस", "बीस",
}

# Words that follow a digit-word and extend it without adding a new digit.
# "teen sau" = 300, still 1 digit in the sequence.
# "do baar" = twice, still 1 count.
_MULTIPLIERS = {
    "sau", "hundred", "thousand", "hazaar", "score", "crore", "lakh",
    "baar", "baari",
}

# Only these words can appear between digit-words without resetting the run.
# Any other non-digit token resets. This replaces the earlier _MAX_GAP logic.
_CONNECTORS = {
    # English
    "and", "or", "then", "next", "followed",
    # Transliterated Hindi
    "ke", "baad", "phir", "fir", "firr", "aur", "ya",
    "uske", "se", "pehle",
    # Devanagari
    "और", "या", "फिर", "के", "बाद", "से", "पहले", "उसके",
}

# "do" is both Hindi "2" and the English verb. If followed by one of these
# English words, it is the verb — skip it.
_ENGLISH_AFTER_DO = {
    "you", "u", "ur", "i", "we", "they", "it", "he", "she",
    "this", "that", "these", "those", "the",
    "not", "n't", "please",
    "have", "has", "had",
    "want", "need", "know", "think",
    "go", "get", "see", "come",
    "something", "anything", "nothing", "everything",
    "any", "some",
    "my", "your", "his", "her", "our", "their",
    "and", "or", "but", "so",
    "one", "two", "three", "four", "five", "six", "seven",
    "eight", "nine", "ten",
}

_MIN_RUN = 3

_TOKEN_RE = re.compile(r"[\w\u0900-\u097F]+", re.UNICODE)


def _is_digit_word(tokens: list[str], i: int) -> bool:
    t = tokens[i]
    if t not in _DIGIT_WORDS:
        return False
    if t == "do" and i + 1 < len(tokens):
        if tokens[i + 1] in _ENGLISH_AFTER_DO:
            return False
    return True


def detect_number_word_sequence(text: str) -> bool:
    """
    Detect a run of digit-words that looks like a disguised phone number,
    e.g. "aath saat teen chaar" (8 7 3 4).

    Rules:
      - At least _MIN_RUN digit-words.
      - Only _CONNECTORS may sit between digit-words; any other word resets.
      - _MULTIPLIERS extend the preceding digit-word (do not add a new one).
      - "do" is skipped when followed by a common English word (verb use).
    """
    if not text:
        return False

    tokens = [t.lower() for t in _TOKEN_RE.findall(text)]
    run = 0
    i = 0
    while i < len(tokens):
        if _is_digit_word(tokens, i):
            run += 1
            if run >= _MIN_RUN:
                return True
            # Consume a following multiplier as part of this digit.
            if i + 1 < len(tokens) and tokens[i + 1] in _MULTIPLIERS:
                i += 2
            else:
                i += 1
            continue
        if tokens[i] in _CONNECTORS:
            i += 1
            continue
        # Anything else resets the run.
        run = 0
        i += 1
    return False
