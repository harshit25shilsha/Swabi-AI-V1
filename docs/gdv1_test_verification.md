# GDv1 Test-Suite Verification

**Purpose:** Record the verification of four reported test-suite issues in
GDv1 before those tests are moved into the unified repo. Per the Guardrail
Integration Plan §5.10, each issue was checked against the actual source
before any fix was proposed.

**Source inspected:** GDv1 combined codebase export (all 17 test files).

**Revision:** v1
**Branch:** `feat/guardrail-integration`

---

## Summary

| # | Reported | Verdict |
|---|---|---|
| 1 | `tests/moderation.py` missing `test_` prefix | **Real** |
| 2 | Duplicate function names in `tests/test_number_words.py` | **Partially real** — duplicate imports are real; function names differ but coverage overlaps |
| 3 | `assert caught_by_rules or True` in `tests/test_disguised_numbers.py` | **Real** |
| 4 | Duplicate entries in `NORMAL_MUST_ALLOW` in `tests/test_integration_llm.py` | **Real** |

Correction to the earlier review: issue 2 was described as "duplicate
function names." There are no literally duplicate function names. The
real problems are a duplicated `import` block and overlapping test
coverage under different names. This correction is recorded here so the
plan's later steps reference accurate information.

---

## Issue 1 — `tests/moderation.py` file-name prefix

**Finding.** The file contains five correctly-named `test_*` functions
but is not itself named `test_*.py`. Pytest's default collection pattern
(`test_*.py` or `*_test.py`) skips it. The suite reports 0 tests from
this file even though five tests exist.

**Impact.** Silent non-execution. Any regression in
`app/services/moderation.py`'s behaviour that these five tests would
have caught is not caught today.

**Fix.** Rename to `tests/test_moderation_service.py`. Content
unchanged.

---

## Issue 2 — `tests/test_number_words.py`

**Finding 2a — duplicate imports.** The block

```python
import pytest
from app.detectors.number_words import detect_number_word_sequence
```

appears twice, the second time after `test_relational_sequence_detected`.
Python silently re-binds; no error. The file is the product of two edits
that were never reconciled.

**Finding 2b — overlapping coverage.** Two pairs of functions test
overlapping inputs under different names:

- `test_relational_sequence_detected` (3 params) is a strict subset of
  `test_disguised_sequence_still_detected` (6 params).
- `test_no_false_positive_on_normal_booking_chat` (6 params) and
  `test_no_false_positive_on_booking_chat` (7 params) share 2 identical
  params.

Neither pair is a literal duplicate. Coverage is redundant and the split
is confusing to maintain.

**Fix.** Remove the second import block. Merge each pair into one
function with the deduplicated union of params. See Appendix B.

---

## Issue 3 — `tests/test_disguised_numbers.py`

**Finding.** The test

```python
@pytest.mark.parametrize("text", DISGUISED_MESSAGES)
def test_disguised_number_reaches_llm(text):
    caught_by_rules = (
        detect_phone(text)
        or detect_number_word_sequence(text)
        or detect_encoded_digits(text)
    )
    assert caught_by_rules or True
```

contains a no-op assertion. `x or True` is always `True`. The comment
acknowledges this is a placeholder. A test that can never fail inflates
the pass count and hides that the deterministic contract is not
asserted anywhere in this file.

**Fix.** Delete the test and the unused `DISGUISED_MESSAGES` constant.
Keep `test_no_false_positive_on_normal_chat`. Remove the now-unused
`detect_phone` import. See Appendix C.

---

## Issue 4 — `tests/test_integration_llm.py`

**Finding.** The final two entries in `NORMAL_MUST_ALLOW` duplicate
entries 8 and 9:

```python
NORMAL_MUST_ALLOW = [
    ...
    "teen sau ya paanch sau rupaye mein ho jayega?",        # idx 8
    "No, I need one room with two beds for three nights",   # idx 9
    "ek room chahiye, do bed ke saath, teen raat ke liye",  # idx 10
    "teen sau ya paanch sau rupaye mein ho jayega?",        # dup of 8
    "No, I need one room with two beds for three nights",   # dup of 9
]
```

Because these are integration tests that call the live Groq API, each
duplicate consumes real quota.

**Fix.** Remove the last two lines. See Appendix D.

---

## Files changed in step 7 (when moved)

| GDv1 source | New location | Change |
|---|---|---|
| `tests/moderation.py` | `tests/unit/chat_safety/test_moderation_service.py` | Rename only |
| `tests/test_number_words.py` | `tests/unit/chat_safety/test_number_words.py` | Merge overlapping tests; remove duplicate import |
| `tests/test_disguised_numbers.py` | `tests/unit/chat_safety/test_disguised_numbers.py` | Remove no-op test |
| `tests/test_integration_llm.py` | `tests/integration/chat_safety/test_integration_llm.py` | Remove duplicate entries |

The remaining 13 GDv1 test files move unchanged in step 7. They are
covered by the file-level review already recorded in the integration
plan and do not require further verification before the move.

---

## Appendix A — `test_moderation_service.py` (renamed, content unchanged)

GDv1 `tests/moderation.py` moves verbatim to
`tests/unit/chat_safety/test_moderation_service.py`. Only the file name
changes.

---

## Appendix B — `tests/unit/chat_safety/test_number_words.py` (corrected)

```python
import pytest

from app.detectors.number_words import detect_number_word_sequence


@pytest.mark.parametrize("text", [
    "aath saat teen chaar, or firr uske bad gyarah terah satrah",
    "eight seven three four five six",
    "ek do teen char paanch chhe",
    "आठ सात तीन चार",
    "one two three four",
])
def test_detects_number_word_run(text):
    assert detect_number_word_sequence(text) is True


@pytest.mark.parametrize("text", [
    "do din ka rent kitna hai?",           # 2 digit-words, legit
    "teen raat ke liye booking",           # 1 digit-word, legit
    "paanch sau rupaye discount milega?",  # 1 digit-word, legit
    "Is the villa available?",             # no digit-words
    "aath saat",                           # only 2, below threshold
])
def test_no_false_positive(text):
    assert detect_number_word_sequence(text) is False


@pytest.mark.parametrize("msg", [
    "Do you have one room with two beds?",
    "I do want one room for two nights",
    "Can you do one thing? Two rooms please",
    "What do I need for one adult and two children?",
    "Do we need one or two cars?",
])
def test_no_false_positive_on_english_verb_do(msg):
    assert detect_number_word_sequence(msg) is False


@pytest.mark.parametrize("msg", [
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
])
def test_no_false_positive_on_booking_chat(msg):
    assert detect_number_word_sequence(msg) is False


@pytest.mark.parametrize("msg", [
    # Relational/arithmetic disguised sequences
    "fourteen ke baad sixteen, phir nineteen aur twenty one",
    "teen score ke baad paanch, phir do aur nau",
    "aath ka aadha nahi, seedha aath; phir teen teen aur ek",
    # Multi-run digit-word sequences
    "aath saat teen chaar, phir gyarah terah satrah",
    "pehle paanch, phir do do, uske baad nau aur chhe",
    "zero se shuru karo, teen baar chaar, phir saat",
])
def test_disguised_sequence_detected(msg):
    assert detect_number_word_sequence(msg) is True
```

**Changes from GDv1:**

- Removed second `import pytest` / `from app.detectors…` block.
- Merged `test_relational_sequence_detected` into
  `test_disguised_sequence_still_detected` → renamed to
  `test_disguised_sequence_detected`.
- Merged `test_no_false_positive_on_normal_booking_chat` into
  `test_no_false_positive_on_booking_chat` (union of params,
  deduplicated).
- No assertion logic changed; test coverage is the same or larger.

---

## Appendix C — `tests/unit/chat_safety/test_disguised_numbers.py` (corrected)

```python
"""
Unit tests for the deterministic layer. LLM-only cases are covered by the
integration script tests/test_integration_llm.py (runs against the real
Groq API).
"""
import pytest

from app.detectors.encoded_digits import detect_encoded_digits
from app.detectors.number_words import detect_number_word_sequence


@pytest.mark.parametrize("text", [
    "do din ka rent kitna hai?",
    "teen raat ke liye booking karni hai",
    "paanch sau rupaye discount milega?",
    # NOTE: a run of 4+ spelled digits ("eight seven three four") is
    # blocked by design (see test_number_words.py), so it is not listed
    # as normal chat.
    "room 2 has two more windows than room 1",
    "Is the villa available from 10th to 12th October?",
    "What payment methods does the platform support?",
])
def test_no_false_positive_on_normal_chat(text):
    assert detect_number_word_sequence(text) is False
    assert detect_encoded_digits(text) is False
```

**Changes from GDv1:**

- Deleted `test_disguised_number_reaches_llm` and its parametrize
  source `DISGUISED_MESSAGES` (both unused after removal).
- Removed the now-unused `from app.detectors.phone import detect_phone`
  import.
- Kept `test_no_false_positive_on_normal_chat` unchanged.

---

## Appendix D — `tests/integration/chat_safety/test_integration_llm.py` (corrected `NORMAL_MUST_ALLOW`)

Only the `NORMAL_MUST_ALLOW` list changes. The rest of the file is
unchanged from GDv1.

```python
NORMAL_MUST_ALLOW = [
    "Platform fee jyada hai, kya discount milega?",
    "What payment methods does the platform support?",
    "Can I receive an email confirmation?",
    "Is the villa available from 10th to 12th October?",
    "What time is check-in?",
    "Can I cancel my booking?",
    "do din ka rent kitna hai?",
    "teen raat ke liye booking karni hai",
    "teen sau ya paanch sau rupaye mein ho jayega?",
    "No, I need one room with two beds for three nights",
    "ek room chahiye, do bed ke saath, teen raat ke liye",
]
```

**Changes from GDv1:**

- Removed the final two duplicate entries (previously indices 11 and
  12). All remaining entries are unique.