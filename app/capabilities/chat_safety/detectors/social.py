def detect_social(text: str) -> bool:
    # Intentionally conservative: handles like "my insta is xyz" have too many
    # false positives with regex. Delegate to the LLM for Day 1.
    return False
