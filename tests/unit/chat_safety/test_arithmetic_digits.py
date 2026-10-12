import pytest

from app.capabilities.chat_safety.detectors.arithmetic_digits import detect_arithmetic_digits


@pytest.mark.parametrize("msg", [
    "do plus do, phir teen minus one, aur end mein nine",
    "teen plus char, then saat minus do",
    "ek kam das, do zyada paanch, phir aath",
    "one less than ten, two more than five, then eight",
    "दो plus दो, फिर तीन minus एक",
])
def test_detects_encoded_arithmetic(msg):
    assert detect_arithmetic_digits(msg)


@pytest.mark.parametrize("msg", [
    "do plus do is four, right?",
    "We need two rooms and one extra bed",
    "do zyada paanch nahi chahiye",
    "Is there one less than ten rooms left?",
    "Is the car available tomorrow?",
    "teen sau ya paanch sau rupaye mein ho jayega?",
    "two hundred or three hundred rupees?",
    "do hazaar ya teen hazaar tak chalega",
    "",
])
def test_ignores_normal_chat(msg):
    assert not detect_arithmetic_digits(msg)

@pytest.mark.parametrize("msg", [
    "teen sau ya paanch sau rupaye mein ho jayega?",   # 300 or 500 — legit
    "Paise sau rupaye mein ho jayenge",                # 100 rupees — legit
])
def test_multiplier_alone_is_not_arithmetic(msg):
    assert detect_arithmetic_digits(msg) is False
