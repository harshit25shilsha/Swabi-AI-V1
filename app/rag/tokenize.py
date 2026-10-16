"""Tokenizer shared by indexing and querying.

Deterministic, no external dependencies. Used at index time to build the
BM25 term list and at query time to tokenize the user question — the two
must match, or BM25 scores will be wrong.

Handles:
  * English words and numbers
  * Hinglish (Latin transliteration of Hindi)
  * Devanagari (\\u0900–\\u097F)
  * Mixed scripts in one token

Contract:
  ``tokenize(text) == tokenize(tokenize(text)[0])`` is not required;
  only that index and query paths call this same function.
"""

from __future__ import annotations

import re
import unicodedata

# Word = one or more letters (Latin or Devanagari) or digits.
# Runs of other characters (punctuation, spaces, symbols) are separators.
_TOKEN_RE = re.compile(r"[A-Za-z0-9\u0900-\u097F]+", re.UNICODE)

# Common English stopwords. Deliberately small: policy text uses words
# like "may", "will", "if" that carry meaning for retrieval.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "or",
        "the",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "with",
        "by",
        "from",
        "as",
        "that",
        "this",
        "these",
        "those",
        "it",
        "its",
        "i",
        "you",
        "we",
        "they",
        "he",
        "she",
    }
)


def normalize(text: str) -> str:
    """NFKC, lowercase, single-space. Applied before tokenization."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str, *, keep_stopwords: bool = False) -> list[str]:
    """Split text into tokens. Deterministic.

    Args:
        text: input string.
        keep_stopwords: if False, English stopwords are removed. Set to
            True when the caller needs every token (e.g. for evaluation
            term counts).
    """
    tokens = _TOKEN_RE.findall(normalize(text))
    if keep_stopwords:
        return tokens
    return [t for t in tokens if t not in _STOPWORDS]


def tokenize_for_query(text: str) -> list[str]:
    """Query-side tokenization. Same rules as index time."""
    return tokenize(text, keep_stopwords=False)
