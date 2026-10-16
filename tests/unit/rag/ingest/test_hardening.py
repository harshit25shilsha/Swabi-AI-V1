"""Regression tests for the two Batch 2 hardening fixes."""
from __future__ import annotations

import copy

import pytest

from app.rag.ingest.chunker import _excluded_matchers, build_chunks
from app.rag.ingest.config import ConfigError, validate_config
from app.rag.ingest.source import parse_markdown
from tests.unit.rag.ingest.conftest import BASE_CFG


def _cfg():
    return copy.deepcopy(BASE_CFG)


# ---------------------------------------------------------------------------
# Fix 1 — Empty exclusion titles
# ---------------------------------------------------------------------------

def test_blank_exclusion_title_rejected_by_config_validation():
    cfg = _cfg()
    cfg["excluded_sections"] = [{"id": "0", "title": "   "}]
    with pytest.raises(ConfigError, match="title"):
        validate_config(cfg, release=False)


def test_blank_exclusion_id_rejected_by_config_validation():
    cfg = _cfg()
    cfg["excluded_sections"] = [{"id": "", "title": "Section 0"}]
    with pytest.raises(ConfigError, match="id"):
        validate_config(cfg, release=False)


def test_exclusion_with_only_title_accepted():
    cfg = _cfg()
    cfg["excluded_sections"] = [{"title": "SECTION 0"}]
    validate_config(cfg, release=False)


def test_matcher_skips_blank_title():
    cfg = _cfg()
    cfg["excluded_sections"] = [{"id": "0", "title": "   "}]
    ids, titles = _excluded_matchers(cfg)
    assert "0" in ids
    assert titles == [], "blank title must not be added"


def test_blank_title_does_not_exclude_every_section():
    md = (
        "# Doc\n\n## 1.1 Alpha\na\n\n## 1.2 Beta\nb\n\n"
        "## 9.9 Something\nbody\n"
    )
    cfg = _cfg()
    # Blank title + real id: only the id should exclude, and only
    # against that id.
    cfg["excluded_sections"] = [{"id": "9.9", "title": "   "}]

    # 1.1 and 1.2 must still be chunked.
    result = build_chunks(cfg, parse_markdown(md), chunk_size_cap=10_000)
    ids = sorted(c.metadata.section_id for c in result.chunks)
    assert ids == ["1.1", "1.2"]


def test_exclusion_targets_only_the_named_section():
    md = (
        "# Doc\n\n## 1.1 Alpha\na\n\n## 1.2 Beta\nb\n\n"
        "## 9.9 Excluded\nbody\n"
    )
    cfg = _cfg()
    cfg["excluded_sections"] = [{"id": "9.9", "title": "Excluded"}]

    result = build_chunks(cfg, parse_markdown(md), chunk_size_cap=10_000)
    ids = sorted(c.metadata.section_id for c in result.chunks)
    assert ids == ["1.1", "1.2"]
    # 9.9 is not present
    assert "9.9" not in ids


def test_excluded_by_title_substring_does_not_touch_unrelated_sections():
    md = (
        "# Doc\n\n## 1.1 Alpha\nalpha body\n\n## 1.2 Beta\nbeta body\n\n"
        "## SECTION 0 — PURPOSE AND USAGE\nmeta body\n"
    )
    cfg = _cfg()
    # Match by title only; no numeric id in the source.
    cfg["excluded_sections"] = [{"title": "PURPOSE AND USAGE"}]

    result = build_chunks(cfg, parse_markdown(md), chunk_size_cap=10_000)
    ids = sorted(c.metadata.section_id for c in result.chunks)
    assert ids == ["1.1", "1.2"]
    text = "\n".join(c.text for c in result.chunks)
    assert "alpha body" in text
    assert "beta body" in text
    assert "meta body" not in text


# ---------------------------------------------------------------------------
# Fix 2 — Ancestor/descendant claim overlap
# ---------------------------------------------------------------------------

def test_nested_claims_rejected():
    """One child claims Sub A; another claims a heading nested under Sub A."""
    md = (
        "# Doc\n\n## 1.1 Parent\nintro\n\n"
        "### Sub A\nsub-a body\n\n"
        "#### Sub A.1\nnested body\n\n"
        "### Sub B\nsub-b body\n"
    )
    cfg = _cfg()
    cfg["sections"] = [
        {
            "id": "1.1", "title": "Parent", "category": "A", "area": "swabi",
            "split_into": [
                {"id": "1.1.a", "title": "A", "category": "A",
                 "area": "swabi", "audience": "both",
                 "includes_subsections": ["Sub A"]},
                {"id": "1.1.b", "title": "B", "category": "A",
                 "area": "swabi", "audience": "both",
                 "includes_subsections": ["Sub A.1", "Sub B"]},
            ],
        },
    ]
    with pytest.raises(ConfigError, match="nested split-parent claims"):
        build_chunks(cfg, parse_markdown(md), chunk_size_cap=10_000)


def test_disjoint_claims_accepted():
    md = (
        "# Doc\n\n## 1.1 Parent\nintro\n\n"
        "### Sub A\nalpha\n\n"
        "### Sub B\nbeta\n"
    )
    cfg = _cfg()
    cfg["sections"] = [
        {
            "id": "1.1", "title": "Parent", "category": "A", "area": "swabi",
            "split_into": [
                {"id": "1.1.a", "title": "A", "category": "A",
                 "area": "swabi", "audience": "both",
                 "includes_subsections": ["Sub A"]},
                {"id": "1.1.b", "title": "B", "category": "A",
                 "area": "swabi", "audience": "both",
                 "includes_subsections": ["Sub B"]},
            ],
        },
    ]
    result = build_chunks(cfg, parse_markdown(md), chunk_size_cap=10_000)
    ids = sorted(c.metadata.section_id for c in result.chunks)
    assert ids == ["1.1.a", "1.1.b"]
