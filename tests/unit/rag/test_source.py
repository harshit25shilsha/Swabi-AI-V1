"""Tests for app.rag.ingest.source — parsing, normalisation, hashing."""

from __future__ import annotations

import pytest

from app.rag.ingest.source import (
    SourceError,
    find_by_section_id,
    find_duplicate_section_ids,
    has_preamble,
    iter_descendants,
    normalise_text,
    parse_markdown,
    sha256_of,
    validate_unique_section_ids,
)

# ---------------------------------------------------------------------------
# Normalisation + hashing
# ---------------------------------------------------------------------------


def test_normalise_strips_bom_and_forces_lf():
    assert normalise_text("\ufeffhello\r\nworld\r") == "hello\nworld\n"


def test_normalise_strips_trailing_whitespace_per_line():
    assert normalise_text("a   \nb\t\n") == "a\nb\n"


def test_sha256_of_is_deterministic_across_line_endings():
    assert sha256_of("a\nb\n") == sha256_of("a\r\nb\r\n") == sha256_of("a\rb\r")


def test_sha256_of_is_hex_64():
    h = sha256_of("hello")
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


# ---------------------------------------------------------------------------
# Parsing basics
# ---------------------------------------------------------------------------


def test_parses_nested_headings():
    md = "# Top\ntop body\n## Mid\nmid body\n### Leaf\nleaf body\n"
    root = parse_markdown(md)
    assert len(root.children) == 1
    top = root.children[0]
    assert top.title == "Top"
    assert top.body == "top body"
    mid = top.children[0]
    assert mid.title == "Mid"
    assert mid.body == "mid body"
    assert mid.children[0].title == "Leaf"
    assert mid.children[0].body == "leaf body"


def test_extracts_section_id_with_various_separators():
    cases = [
        ("## 1.2 Privacy Policy", "1.2", "Privacy Policy"),
        ("## 1.2. Privacy Policy", "1.2", "Privacy Policy"),
        ("## 1.2 - Privacy Policy", "1.2", "Privacy Policy"),
        ("## 1.2 — Privacy Policy", "1.2", "Privacy Policy"),
        ("## 9.3.2 Rafting", "9.3.2", "Rafting"),
    ]
    for md_line, expected_id, expected_title in cases:
        root = parse_markdown(md_line + "\nbody\n")
        node = root.children[0]
        assert node.section_id == expected_id, md_line
        assert node.section_title == expected_title, md_line


def test_heading_without_section_id_has_none():
    root = parse_markdown("# Introduction\nbody\n")
    assert root.children[0].section_id is None
    assert root.children[0].section_title == "Introduction"


def test_empty_source_raises():
    with pytest.raises(SourceError):
        parse_markdown("")


def test_source_with_only_body_raises():
    with pytest.raises(SourceError):
        parse_markdown("just text, no headings\n")


# ---------------------------------------------------------------------------
# Fenced code
# ---------------------------------------------------------------------------


def test_backtick_fenced_heading_is_ignored():
    md = "# Real\nbefore\n```\n# Not a heading\n```\nafter\n"
    root = parse_markdown(md)
    assert len(root.children) == 1
    assert "# Not a heading" in root.children[0].body


def test_tilde_fenced_heading_is_ignored():
    md = "# Real\nbefore\n~~~\n# Fake\n~~~\nafter\n"
    root = parse_markdown(md)
    assert len(root.children) == 1


def test_longer_closing_fence_closes():
    """A ````` block can be closed by a longer `````` run."""
    md = "# Real\n``````\n# Not a heading\n```````\nafter\n"
    root = parse_markdown(md)
    assert len(root.children) == 1
    assert "# Not a heading" in root.children[0].body


def test_invalid_closing_fence_wrong_marker_does_not_close():
    """A ~~~ inside a ``` block does not close it."""
    md = "# Real\n```\n~~~\n# Not a heading\n```\nafter\n"
    root = parse_markdown(md)
    assert len(root.children) == 1
    assert "# Not a heading" in root.children[0].body


def test_invalid_closing_fence_too_short_does_not_close():
    """A `` inside a ``` block does not close it."""
    md = "# Real\n```\n``\n# Not a heading\n```\nafter\n"
    root = parse_markdown(md)
    assert len(root.children) == 1
    assert "# Not a heading" in root.children[0].body


def test_closing_fence_with_trailing_text_does_not_close():
    """A line with content after the fence marker does not close it."""
    md = "# Real\n```\n```not a closer\n# Not a heading\n```\nafter\n"
    root = parse_markdown(md)
    assert len(root.children) == 1
    assert "# Not a heading" in root.children[0].body


def test_opening_fence_with_four_spaces_does_not_open():
    """Indent > 3 spaces is not a valid opening fence."""
    md = "# Real\nbefore\n    ```\n# Heading In List\n    ```\nafter\n"
    root = parse_markdown(md)
    # The 4-space fence does not open a block, so the # heading is parsed.
    assert any(c.title == "Heading In List" for c in root.children)


def test_opening_fence_with_three_spaces_opens():
    md = "# Real\n   ```\n# Not a heading\n   ```\nafter\n"
    root = parse_markdown(md)
    assert len(root.children) == 1
    assert "# Not a heading" in root.children[0].body


def test_backtick_fence_with_backtick_in_info_is_not_opening():
    """CommonMark: backtick info strings cannot contain backticks."""
    md = "# Real\n``` `inline`\n# Not a fence\n"
    root = parse_markdown(md)
    # Not an opening fence, so the # line becomes a heading.
    titles = [c.title for c in root.children]
    assert "Not a fence" in titles


def test_fence_with_info_string_still_opens():
    md = "# Real\n```python\n# Not a heading\n```\nafter\n"
    root = parse_markdown(md)
    assert len(root.children) == 1


def test_unclosed_fence_consumes_rest():
    md = "# Real\nbefore\n```\n# Not a heading\nmore text\n"
    root = parse_markdown(md)
    assert len(root.children) == 1
    assert "# Not a heading" in root.children[0].body


# ---------------------------------------------------------------------------
# Preamble
# ---------------------------------------------------------------------------


def test_preamble_is_preserved():
    md = "Some preamble text.\n\n# First\nbody\n"
    root = parse_markdown(md)
    assert has_preamble(root)
    assert "Some preamble text." in root.body


def test_no_preamble_when_source_starts_with_heading():
    md = "# First\nbody\n"
    root = parse_markdown(md)
    assert not has_preamble(root)
    assert root.body == ""


def test_empty_preamble_when_only_whitespace_before_heading():
    md = "\n\n  \n# First\nbody\n"
    root = parse_markdown(md)
    assert not has_preamble(root)


# ---------------------------------------------------------------------------
# Duplicate ID detection
# ---------------------------------------------------------------------------


def test_duplicate_section_ids_detected():
    md = "## 1.1 A\nx\n## 1.1 B\ny\n"
    root = parse_markdown(md)
    assert "1.1" in find_duplicate_section_ids(root)


def test_unique_section_ids_passes_validation():
    md = "## 1.1 A\nx\n## 1.2 B\ny\n"
    root = parse_markdown(md)
    validate_unique_section_ids(root)


def test_duplicate_section_ids_raise_from_validation():
    md = "## 1.1 A\nx\n## 1.1 B\ny\n"
    root = parse_markdown(md)
    with pytest.raises(SourceError):
        validate_unique_section_ids(root)


# ---------------------------------------------------------------------------
# Tree queries
# ---------------------------------------------------------------------------


def test_find_by_section_id_returns_correct_node():
    md = "## 1.1 A\na\n### 1.1.1 Deep\nd\n## 1.2 B\nb\n"
    root = parse_markdown(md)
    node = find_by_section_id(root, "1.1.1")
    assert node is not None
    assert node.section_title == "Deep"
    assert find_by_section_id(root, "9.9") is None


def test_iter_descendants_is_depth_first_and_excludes_self():
    md = "## 1 A\na\n### 1.1 B\nb\n### 1.2 C\nc\n"
    root = parse_markdown(md)
    top = root.children[0]
    ids = [d.title for d in iter_descendants(top)]
    assert ids == ["1.1 B", "1.2 C"]


# ---------------------------------------------------------------------------
# Heading-marker edge cases
# ---------------------------------------------------------------------------


def test_single_hash_without_content_is_not_a_heading():
    """`#` alone on a line has no content; not a heading."""
    md = "#\nbody\n# Real\nbody\n"
    root = parse_markdown(md)
    titles = [c.title for c in root.children]
    assert titles == ["Real"]


def test_hash_with_only_spaces_is_not_a_heading():
    md = "#   \nbody\n# Real\nbody\n"
    root = parse_markdown(md)
    titles = [c.title for c in root.children]
    assert titles == ["Real"]


def test_hash_with_only_tab_is_not_a_heading():
    md = "#\t\nbody\n# Real\nbody\n"
    root = parse_markdown(md)
    titles = [c.title for c in root.children]
    assert titles == ["Real"]


def test_hash_without_space_is_not_a_heading():
    """CommonMark: `#foo` is not a heading."""
    md = "#foo\nbody\n# Real\nbody\n"
    root = parse_markdown(md)
    titles = [c.title for c in root.children]
    assert titles == ["Real"]


def test_hash_tab_then_content_is_a_heading():
    """Tab counts as the required separator."""
    md = "#\tReal\nbody\n"
    root = parse_markdown(md)
    assert root.children[0].title == "Real"


def test_hash_spaces_then_content_is_a_heading():
    md = "#     Real\nbody\n"
    root = parse_markdown(md)
    assert root.children[0].title == "Real"


def test_trailing_horizontal_whitespace_is_stripped_from_title():
    md = "# Real   \t\nbody\n"
    root = parse_markdown(md)
    assert root.children[0].title == "Real"


def test_seven_hashes_is_not_a_heading():
    """More than six hashes is not a heading."""
    md = "####### Real\nbody\n# Actual\nbody\n"
    root = parse_markdown(md)
    titles = [c.title for c in root.children]
    assert titles == ["Actual"]
