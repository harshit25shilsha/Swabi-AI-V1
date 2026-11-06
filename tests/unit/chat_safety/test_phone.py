import pytest

from app.capabilities.chat_safety.detectors.phone import detect_phone


@pytest.mark.parametrize(
    "text",
    [
        "Call me at 9876543210",
        "My number is +91 98765 43210",
        "Reach me on +91-98765-43210",
        "987 654 3210",
    ],
)
def test_detects_phone(text):
    assert detect_phone(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "Is the car available tomorrow?",
        "Total price is 5000",
        "Available from 10th to 12th October",
    ],
)
def test_no_false_positive(text):
    assert detect_phone(text) is False
