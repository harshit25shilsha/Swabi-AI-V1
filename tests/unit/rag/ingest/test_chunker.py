"""Chunker tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.rag.ingest import (
    build_chunks,
    load_config,
    parse_markdown,
    validate_config,
)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def cfg():
    c = load_config(FIXTURES / "minimal.yaml")
    validate_config(c, release=False)
    return c


@pytest.fixture(scope="module")
def root():
    return parse_markdown((FIXTURES / "minimal.md").read_text(encoding="utf-8"))


def _by_id(result, section_id):
    return next(c for c in result.chunks if c.metadata.section_id == section_id)


def test_basic_chunks(cfg, root):
    r = build_chunks(cfg, root, chunk_size_cap=100_000)
    ids = sorted(c.metadata.section_id for c in r.chunks)
    assert ids == ["1.1", "1.2", "2.1"]


def test_leaf_includes_nested_subheadings(cfg, root):
    r = build_chunks(cfg, root, chunk_size_cap=100_000)
    chunk = _by_id(r, "1.1")
    assert "alpha body" in chunk.text
    assert "nested body" in chunk.text
    assert "1.1.1 Nested" in chunk.text


def test_skipped_subsection_absent_from_text(cfg, root):
    r = build_chunks(cfg, root, chunk_size_cap=100_000)
    chunk = _by_id(r, "2.1")
    assert "Skipped subsection" not in chunk.text
    assert "SHOULD NOT APPEAR" not in chunk.text
    assert "kept body" in chunk.text
    assert any("Skipped subsection" in s for s in r.skipped_subsections)


def test_chunk_id_stable_when_unrelated_section_changes(cfg):
    base_md = (FIXTURES / "minimal.md").read_text(encoding="utf-8")
    edited = base_md.replace("beta body", "beta body MODIFIED")

    r1 = build_chunks(cfg, parse_markdown(base_md), chunk_size_cap=100_000)
    r2 = build_chunks(cfg, parse_markdown(edited), chunk_size_cap=100_000)

    assert _by_id(r1, "1.1").metadata.chunk_id == _by_id(r2, "1.1").metadata.chunk_id
    assert _by_id(r1, "1.2").metadata.chunk_id != _by_id(r2, "1.2").metadata.chunk_id


def test_chunk_id_isolated_across_documents(cfg, root):
    r1 = build_chunks(cfg, root, chunk_size_cap=100_000)
    cfg2 = {**cfg, "source_document": "Another Fixture"}
    r2 = build_chunks(cfg2, root, chunk_size_cap=100_000)
    assert _by_id(r1, "1.1").metadata.chunk_id != _by_id(r2, "1.1").metadata.chunk_id


def test_oversized_section_splits(cfg, root):
    r = build_chunks(cfg, root, chunk_size_cap=40)
    split_ids = [c.metadata.section_id for c in r.chunks
                 if c.metadata.section_id.startswith("1.1__p")]
    assert split_ids, "1.1 did not split"
    for cid in split_ids:
        assert "1.1 Alpha" in _by_id(r, cid).text
    assert r.over_cap == []


def test_review_flag_ids_derived_when_missing(cfg, root):
    cfg2 = {**cfg, "sections": [dict(s) for s in cfg["sections"]]}
    cfg2["sections"][0] = {**cfg2["sections"][0],
                           "review_flags": [{"text": "check A"},
                                            {"text": "check B"}]}
    r = build_chunks(cfg2, root, chunk_size_cap=100_000)
    flags = _by_id(r, "1.1").metadata.review_flags
    assert [f.id for f in flags] == ["1.1#0", "1.1#1"]
