from app.capabilities.chat_safety.detectors.email import detect_email


def test_plain_email():
    assert detect_email("Email me at john@gmail.com") is True


def test_obfuscated_email_brackets():
    assert detect_email("john [at] gmail [dot] com") is True


def test_obfuscated_email_words():
    assert detect_email("john AT gmail DOT com") is True


def test_no_email():
    assert detect_email("Is the villa available?") is False
