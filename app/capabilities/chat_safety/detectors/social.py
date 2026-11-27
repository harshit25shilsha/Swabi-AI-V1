import re

_PLATFORM = (
    r"(?:whats\s?app|telegram|instagram|signal|messenger|wechat|viber|line|"
    r"snapchat|discord)"
)
_ACTION = r"(?:connect|contact|chat|talk|message|continue|move|switch|baat)"
_PREPOSITION = r"(?:on|over|through|via|to|pe|par|mein|me)"

_MOVE_TO_PLATFORM_RE = re.compile(
    rf"\b{_ACTION}\b.{{0,35}}\b{_PREPOSITION}\b.{{0,20}}\b{_PLATFORM}\b"
    rf"|\b{_PLATFORM}\b.{{0,20}}\b{_PREPOSITION}\b.{{0,20}}\b{_ACTION}\b",
    re.IGNORECASE,
)


def detect_social(text: str) -> bool:
    """Detect explicit requests to move a conversation to a messaging platform."""
    return bool(text and _MOVE_TO_PLATFORM_RE.search(text))
