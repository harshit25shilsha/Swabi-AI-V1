"""Regression tests for Batch 2 review fixes."""
from __future__ import annotations

import copy
from pathlib import Path

import pytest

from app.rag.ingest.chunker import build_chunks
from app.rag.ingest.config import ConfigError, validate_config
from app.rag.ingest.source import (
    SourceError,
    parse_markdown,
    validate_unique_titles_within_parents,
)
from app.rag.ingest.writer import (
    PublishError,
    compute_index_version,
    current_version,
    publish,
)
from app.rag.schemas import (
    Audience,
    Chunk,
    ChunkMetadata,
    IndexVersion,
    IngestionReport,
    KnowledgeArea,
)
from tests.unit.rag.ingest.conftest import BASE_CFG


def _cfg():
    return copy.deepcopy(BASE_CFG)


# ---------------------------------------------------------------------------
# 1. Config validation — malformed children
# ---------------------------------------------------------------------------

def test_split_child_missing_required_field_raises_config_error():
    cfg = _cfg()
    cfg["sections"].append({
        "id": "9.9", "title": "Parent", "category": "A", "area": "swabi",
        "split_into": [
            {"id": "9.9.a", "title": "A"},  # missing category/area/audience/includes
        ],
    })
    with pytest.raises(ConfigError, match="missing"):
        validate_config(cfg, release=False)


def test_split_child_missing_includes_subsections_raises_clear_error():
    cfg = _cfg()
    cfg["sections"].append({
        "id": "9.9", "title": "Parent", "category": "A", "area": "swabi",
        "split_into": [
            {"id": "9.9.a", "title": "A", "category": "A", "area": "swabi",
             "audience": "both"},  # no includes_subsections
        ],
    })
    with pytest.raises(ConfigError, match="includes_subsections"):
        validate_config(cfg, release=False)


# ---------------------------------------------------------------------------
# 2. Coverage — duplicate titles
# ---------------------------------------------------------------------------

def test_duplicate_titles_within_parent_rejected():
    md = (
        "# Doc\n\n## 1.1 Alpha\n\n"
        "### Overview\nfirst\n\n### Overview\nsecond\n"
    )
    root = parse_markdown(md)
    with pytest.raises(SourceError, match="duplicate subheading titles"):
        validate_unique_titles_within_parents(root, split_parent_ids={"1.1"})


def test_unique_titles_ok():
    md = "# Doc\n\n## 1.1 Alpha\n\n### A\na\n\n### B\nb\n"
    validate_unique_titles_within_parents(
        parse_markdown(md), split_parent_ids={"1.1"}
    )

# ---------------------------------------------------------------------------
# 3. Chunk size — atomic content longer than cap
# ---------------------------------------------------------------------------

def test_single_sentence_over_cap_raises_clear_error():
    # One unbroken token with no whitespace at all — no splitter can
    # reduce it below the cap.
    body = "x" * 500
    md = f"# Doc\n\n## 1.1 Alpha\n{body}\n\n## 1.2 Beta\nok\n"
    with pytest.raises(ConfigError, match="cannot be split"):
        build_chunks(_cfg(), parse_markdown(md), chunk_size_cap=200)

def test_heading_overhead_accounted_for():
    # Cap barely fits heading + a tiny body.
    md = "# Doc\n\n## 1.1 Alpha\nx\n\n## 1.2 Beta\nx\n"
    result = build_chunks(_cfg(), parse_markdown(md), chunk_size_cap=40)
    for c in result.chunks:
        assert len(c.text) <= 40, f"{c.metadata.section_id}: {len(c.text)}"
    assert result.over_cap == []


# ---------------------------------------------------------------------------
# 5. Release validation
# ---------------------------------------------------------------------------

def test_confirmed_string_false_rejected():
    cfg = _cfg()
    cfg["sections"][0]["review_flags"] = [
        {"text": "check", "confirmed": "false"},  # string, not bool
    ]
    with pytest.raises(ConfigError, match="boolean"):
        validate_config(cfg, release=False)


def test_confirmed_string_true_rejected():
    cfg = _cfg()
    cfg["sections"][0]["review_flags"] = [
        {"text": "check", "confirmed": "true"},
    ]
    with pytest.raises(ConfigError, match="boolean"):
        validate_config(cfg, release=False)


# ---------------------------------------------------------------------------
# 7. Safe publishing — concurrent-safe staging
# ---------------------------------------------------------------------------

def _chunk(section_id: str, text: str) -> Chunk:
    return Chunk(
        text=text,
        metadata=ChunkMetadata(
            chunk_id=f"id-{section_id}", document_id="doc",
            category="A", section_id=section_id, section_title=section_id,
            section_path=section_id,
            area=KnowledgeArea.SWABI, audience=Audience.BOTH,
            source="Test KB", source_version="1.0", last_updated="2026-10",
        ),
    )


def _manifest(version: str, chunk_count: int = 1) -> IndexVersion:
    return IndexVersion(
        index_version=version, pipeline_version="1",
        source_path="x.md", source_hash="a" * 64, source_version="1.0",
        config_hash="b" * 64, embedding_model=None,
        release=False, chunk_count=chunk_count, has_embeddings=False,
    )


def _report() -> IngestionReport:
    return IngestionReport(
        built_at="2026-10-16T00:00:00+00:00",
        source_path="x.md", source_hash="a" * 64, source_version="1.0",
        config_hash="b" * 64, config_path="y.yaml",
        pipeline_version="1", total_chunks=1,
    )


def test_manifest_chunk_count_mismatch_rejected(tmp_path: Path):
    with pytest.raises(PublishError, match="chunk_count mismatch"):
        publish(
            tmp_path, version="v1",
            chunks=[_chunk("1.1", "a"), _chunk("1.2", "b")],
            manifest=_manifest("v1", chunk_count=1),  # mismatch
            report=_report(), known_gaps={},
        )


def test_manifest_version_mismatch_rejected(tmp_path: Path):
    with pytest.raises(PublishError, match="version mismatch"):
        publish(
            tmp_path, version="v1",
            chunks=[_chunk("1.1", "a")],
            manifest=_manifest("v2", chunk_count=1),  # mismatch
            report=_report(), known_gaps={},
        )


def test_second_publish_to_same_version_is_idempotent(tmp_path: Path):
    publish(tmp_path, version="v1", chunks=[_chunk("1.1", "a")],
            manifest=_manifest("v1"), report=_report(), known_gaps={})
    publish(tmp_path, version="v1", chunks=[_chunk("1.1", "a")],
            manifest=_manifest("v1"), report=_report(), known_gaps={})
    assert current_version(tmp_path) == "v1"


def test_no_staging_left_behind_after_failure(tmp_path: Path):
    publish(tmp_path, version="v1", chunks=[_chunk("1.1", "a")],
            manifest=_manifest("v1"), report=_report(), known_gaps={})
    with pytest.raises(PublishError):
        publish(tmp_path, version="v1", chunks=[_chunk("1.1", "OTHER")],
                manifest=_manifest("v1"), report=_report(), known_gaps={})
    names = sorted(p.name for p in (tmp_path / "versions").iterdir())
    assert names == ["v1"]


# ---------------------------------------------------------------------------
# 8. Version fingerprint
# ---------------------------------------------------------------------------

def test_version_covers_chunk_size_cap():
    a = compute_index_version(
        source_hash="a" * 64, config_hash="b" * 64,
        pipeline_version="1", embedding_model=None,
        chunk_size_cap=4000, release=False,
    )
    b = compute_index_version(
        source_hash="a" * 64, config_hash="b" * 64,
        pipeline_version="1", embedding_model=None,
        chunk_size_cap=1000, release=False,
    )
    assert a != b


def test_version_covers_release_flag():
    a = compute_index_version(
        source_hash="a" * 64, config_hash="b" * 64,
        pipeline_version="1", embedding_model=None,
        chunk_size_cap=4000, release=False,
    )
    b = compute_index_version(
        source_hash="a" * 64, config_hash="b" * 64,
        pipeline_version="1", embedding_model=None,
        chunk_size_cap=4000, release=True,
    )
    assert a != b
