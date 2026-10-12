import pytest

from app.capabilities.chat_safety.detectors.encoded_digits import detect_encoded_digits


@pytest.mark.parametrize("text", [
    "start with the number after five, then two less than ten, then twelve",
    "half of twelve, then the number after eight",
    "the digit between four and six, then seventeen minus ten",
    "one less than nine, then two more than five",
])
def test_detects_encoded_digit_sequence(text):
    assert detect_encoded_digits(text) is True


@pytest.mark.parametrize("text", [
    "the room is two more than the one I saw",   # phrase, no sequence
    "Is the villa available?",                    # neither
    "do din ka rent kitna hai?",                  # neither
    "the number after five",                      # phrase, no connector
    "then we can book it",                        # connector, no phrase
])
def test_no_false_positive(text):
    assert detect_encoded_digits(text) is False
