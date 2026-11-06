import pytest

from app.capabilities.chat_safety.detectors.url import detect_url


@pytest.mark.parametrize(
    "text",
    [
        "See https://example.com for details",
        "http://example.com/foo",
        "www.example.com",
        "bit.ly/abc",
    ],
)
def test_detects_url(text):
    assert detect_url(text) is True


def test_no_url():
    assert detect_url("Is the villa available?") is False
