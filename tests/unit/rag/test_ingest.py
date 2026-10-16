"""Ingestion pipeline tests. Requires langchain-text-splitters."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.rag.config import ConfigError

pytest.importorskip("langchain_text_splitters")


from app.rag.ingest import (  # noqa: E402
    IngestError,
    _compute_version,
    ingest,
    normalise_text,
    sha256_of,
)

FIXTURES = Path(__file__).parent / "fixtures"
CONFIG = FIXTURES / "minimal.yaml"


def test_normalise():
    assert normalise_text("a\r\nb\r") == "a\nb\n"


def test_sha256_stable_across_line_endings():
    assert sha256_of("a\nb\n") == sha256_of("a\r\nb\r\n")


def test_ingest_produces_chunks():
    r = ingest(CONFIG, chunk_size_cap=10_000)
    ids = sorted(c.metadata.section_id for c in r.chunks)
    assert "1.1" in ids
    assert "2.3" in ids
    assert "9.1.kashmir" in ids
    assert "9.1.goa" in ids
    assert "9.9" not in ids  # excluded

def test_excluded_section_absent():
    r = ingest(CONFIG, chunk_size_cap=10_000)
    text = "\n".join(c.text for c in r.chunks)
    assert "should not appear" not in text


def test_metadata_attached():
    r = ingest(CONFIG, chunk_size_cap=10_000)
    c = next(c for c in r.chunks if c.metadata.section_id == "1.1")
    assert c.metadata.area.value == "swabi"
    assert c.metadata.audience.value == "both"
    assert c.metadata.category == "A"


def test_chunk_id_deterministic():
    r1 = ingest(CONFIG, chunk_size_cap=10_000)
    r2 = ingest(CONFIG, chunk_size_cap=10_000)
    ids1 = sorted(c.metadata.chunk_id for c in r1.chunks)
    ids2 = sorted(c.metadata.chunk_id for c in r2.chunks)
    assert ids1 == ids2


def test_version_covers_cap_and_release():
    a = _compute_version(source_hash="a" * 64, config_hash="b" * 64,
                         chunk_size_cap=4000, release=False)
    b = _compute_version(source_hash="a" * 64, config_hash="b" * 64,
                         chunk_size_cap=1000, release=False)
    c = _compute_version(source_hash="a" * 64, config_hash="b" * 64,
                         chunk_size_cap=4000, release=True)
    assert a != b and a != c


def test_release_blocks_on_unconfirmed_flags(tmp_path):
    import copy

    import yaml
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    cfg = copy.deepcopy(cfg)
    cfg["source_hash_expected"] = sha256_of(
        (FIXTURES / "minimal.md").read_text(encoding="utf-8")
    )
    cfg["sections"][0]["review_flags"] = [
        {"id": "flag1", "text": "check", "confirmed": False}
    ]
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(IngestError, match="unconfirmed"):
        ingest(p, release=True, chunk_size_cap=10_000)


def test_missing_source_raises(tmp_path):
    import yaml
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    cfg["source_path"] = str(tmp_path / "nope.md")
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(ConfigError, match="source not found"):
        ingest(p, chunk_size_cap=10_000)

def test_leaf_is_one_chunk():
    """1.1 has subheadings but should be one chunk."""
    r = ingest(CONFIG, chunk_size_cap=10_000)
    ones = [c for c in r.chunks if c.metadata.section_id == "1.1"]
    assert len(ones) == 1
    assert "content A" in ones[0].text
    assert "content B" in ones[0].text


def test_no_duplicate_headings():
    r = ingest(CONFIG, chunk_size_cap=10_000)
    c = next(c for c in r.chunks if c.metadata.section_id == "1.1")
    # Heading text should appear once, not twice.
    assert c.text.count("1.1 Alpha") == 1


def test_skip_subsections_applied():
    r = ingest(CONFIG, chunk_size_cap=10_000)
    c = next(c for c in r.chunks if c.metadata.section_id == "2.3")
    assert "Basic 0%" in c.text
    assert "proposed redesign" not in c.text.lower()
    assert "SHOULD NOT APPEAR" not in c.text


def test_split_parent_children_produced():
    r = ingest(CONFIG, chunk_size_cap=10_000)
    ids = {c.metadata.section_id for c in r.chunks}
    assert "9.1.kashmir" in ids
    assert "9.1.goa" in ids
    kashmir = next(c for c in r.chunks if c.metadata.section_id == "9.1.kashmir")
    assert kashmir.metadata.destination == "Kashmir"
