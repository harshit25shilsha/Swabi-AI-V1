"""Chunker — coverage validation and split behavior."""
from __future__ import annotations

import copy

import pytest

from app.rag.ingest.chunker import build_chunks
from app.rag.ingest.config import ConfigError
from app.rag.ingest.source import parse_markdown
from tests.unit.rag.ingest.conftest import BASE_CFG


def _cfg(**overrides):
    base = copy.deepcopy(BASE_CFG)
    base.update(overrides)
    return base


def _two_section_cfg():
    """Config matching a markdown with 1.1 Alpha and 1.2 Beta."""
    return _cfg()


# ---------------------------------------------------------------------------
# Coverage — unreferenced source headings
# ---------------------------------------------------------------------------

def test_unreferenced_source_heading_with_body_rejected():
    md = "# Doc\n\n## 1.1 Alpha\na\n\n## 1.9 Unreferenced\norphan body\n"
    cfg = _two_section_cfg()
    # Remove 1.2 from config so only the referenced 1.1 remains.
    cfg["sections"] = [s for s in cfg["sections"] if s["id"] == "1.1"]
    with pytest.raises(ConfigError, match="not referenced"):
        build_chunks(cfg, parse_markdown(md), chunk_size_cap=10_000)


def test_unreferenced_structural_container_is_allowed():
    md = (
        "# Doc\n\n## SECTION 1\n\n"
        "### 1.1 Alpha\na\n\n### 1.2 Beta\nb\n"
    )
    cfg = _cfg()
    build_chunks(cfg, parse_markdown(md), chunk_size_cap=10_000)


def test_unreferenced_container_with_body_rejected():
    md = (
        "# Doc\n\n## SECTION 1\ncontainer text\n\n"
        "### 1.1 Alpha\na\n\n### 1.2 Beta\nb\n"
    )
    cfg = _cfg()
    with pytest.raises(ConfigError, match="not referenced"):
        build_chunks(cfg, parse_markdown(md), chunk_size_cap=10_000)


def test_excluded_source_heading_with_body_allowed():
    md = (
        "# Doc\n\n## 1.1 Alpha\na\n\n## 1.2 Beta\nb\n\n"
        "## 9.9 Excluded\nbody under excluded\n"
    )
    cfg = _cfg()
    cfg["excluded_sections"] = ["9.9"]
    build_chunks(cfg, parse_markdown(md), chunk_size_cap=10_000)


def test_split_parent_unclaimed_subheading_rejected():
    md = "# Doc\n\n## 1.1 Alpha\n\n### Sub A\na\n\n### Orphan\norphan\n"
    cfg = _cfg()
    cfg["sections"] = [
        {
            "id": "1.1", "title": "Parent", "category": "A", "area": "swabi",
            "split_into": [
                {"id": "1.1.a", "title": "A", "category": "A", "area": "swabi",
                 "audience": "both", "includes_subsections": ["Sub A"]},
            ],
        },
    ]
    with pytest.raises(ConfigError, match="not claimed"):
        build_chunks(cfg, parse_markdown(md), chunk_size_cap=10_000)


def test_split_child_includes_parent_body():
    md = (
        "# Doc\n\n## 1.1 Alpha\nparent intro text\n\n"
        "### Sub A\nalpha\n\n### Sub B\nbeta\n"
    )
    cfg = _cfg()
    cfg["sections"] = [
        {
            "id": "1.1", "title": "Parent", "category": "A", "area": "swabi",
            "split_into": [
                {"id": "1.1.a", "title": "A", "category": "A", "area": "swabi",
                 "audience": "both", "includes_subsections": ["Sub A"]},
                {"id": "1.1.b", "title": "B", "category": "A", "area": "swabi",
                 "audience": "both", "includes_subsections": ["Sub B"]},
            ],
        },
    ]
    result = build_chunks(cfg, parse_markdown(md), chunk_size_cap=10_000)
    a = next(c for c in result.chunks if c.metadata.section_id == "1.1.a")
    assert "parent intro text" in a.text


# ---------------------------------------------------------------------------
# Cap enforcement
# ---------------------------------------------------------------------------

def test_chunk_size_cap_must_be_positive():
    md = "# Doc\n\n## 1.1 Alpha\na\n\n## 1.2 Beta\nb\n"
    with pytest.raises(ConfigError, match="positive"):
        build_chunks(_cfg(), parse_markdown(md), chunk_size_cap=0)


def test_cap_too_small_for_heading_rejected():
    md = "# Doc\n\n## 1.1 Alpha\nbody\n\n## 1.2 Beta\nbody\n"
    with pytest.raises(ConfigError, match="smaller than"):
        build_chunks(_cfg(), parse_markdown(md), chunk_size_cap=5)


def test_oversized_section_splits_under_cap():
    big = "word " * 200
    md = f"# Doc\n\n## 1.1 Alpha\n{big}\n\n## 1.2 Beta\n{big}\n"
    result = build_chunks(_cfg(), parse_markdown(md), chunk_size_cap=300)
    for c in result.chunks:
        assert len(c.text) <= 300, (
            f"chunk {c.metadata.section_id} len={len(c.text)}"
        )
    assert result.over_cap == []
