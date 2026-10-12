import pytest

from app.capabilities.chat_safety.detectors.number_words import detect_number_word_sequence


@pytest.mark.parametrize(
    "text",
    [
        "aath saat teen chaar, or firr uske bad gyarah terah satrah",
        "eight seven three four five six",
        "ek do teen char paanch chhe",
        "आठ सात तीन चार",
        "one two three four",
    ],
)
def test_detects_number_word_run(text):
    assert detect_number_word_sequence(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "do din ka rent kitna hai?",  # 2 digit-words, legit
        "teen raat ke liye booking",  # 1 digit-word, legit
        "paanch sau rupaye discount milega?",  # 1 digit-word, legit
        "Is the villa available?",  # no digit-words
        "aath saat",  # only 2, below threshold
    ],
)
def test_no_false_positive(text):
    assert detect_number_word_sequence(text) is False


@pytest.mark.parametrize(
    "msg",
    [
        "Do you have one room with two beds?",
        "I do want one room for two nights",
        "Can you do one thing? Two rooms please",
        "What do I need for one adult and two children?",
        "Do we need one or two cars?",
    ],
)
def test_no_false_positive_on_english_verb_do(msg):
    assert detect_number_word_sequence(msg) is False


@pytest.mark.parametrize(
    "msg",
    [
        "teen sau ya paanch sau rupaye mein ho jayega?",
        "No, I need one room with two beds for three nights",
        "ek room chahiye, do bed ke saath, teen raat ke liye",
        "do ya teen din ke liye villa chahiye, ek ya do room",
        "we are 4 people, ek car aur do driver chahiye",
        "tera number kya hai booking ke liye?",
        "I want two or three rooms",
        "Do you have one room with two beds?",
        "We need three nights and two adults",
        "Paise teen sau ya paanch sau mein",
        "Room one, room two, room three",
    ],
)
def test_no_false_positive_on_booking_chat(msg):
    assert detect_number_word_sequence(msg) is False


@pytest.mark.parametrize(
    "msg",
    [
        # Relational/arithmetic disguised sequences
        "fourteen ke baad sixteen, phir nineteen aur twenty one",
        "teen score ke baad paanch, phir do aur nau",
        "aath ka aadha nahi, seedha aath; phir teen teen aur ek",
        # Multi-run digit-word sequences
        "aath saat teen chaar, phir gyarah terah satrah",
        "pehle paanch, phir do do, uske baad nau aur chhe",
        "zero se shuru karo, teen baar chaar, phir saat",
    ],
)
def test_disguised_sequence_detected(msg):
    assert detect_number_word_sequence(msg) is True
