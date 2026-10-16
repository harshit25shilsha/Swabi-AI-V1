"""Writer — version identity and non-destructive publishing."""
from __future__ import annotations

from pathlib import Path

import pytest

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


def _manifest(version: str, *, release=False) -> IndexVersion:
    return IndexVersion(
        index_version=version, pipeline_version="1",
        source_path="x.md", source_hash="a" * 64, source_version="1.0",
        config_hash="b" * 64, embedding_model=None,
        release=release, chunk_count=1, has_embeddings=False,
    )


def _report() -> IngestionReport:
    return IngestionReport(
        built_at="2026-10-16T00:00:00+00:00",
        source_path="x.md", source_hash="a" * 64, source_version="1.0",
        config_hash="b" * 64, config_path="y.yaml",
        pipeline_version="1", total_chunks=1,
    )


# ---------------------------------------------------------------------------
# Version identity
# ---------------------------------------------------------------------------

def test_version_deterministic():
    a = compute_index_version(
        source_hash="a" * 64, config_hash="b" * 64,
        pipeline_version="1", embedding_model=None,
        chunk_size_cap=4000, release=False,
    )
    b = compute_index_version(
        source_hash="a" * 64, config_hash="b" * 64,
        pipeline_version="1", embedding_model=None,
        chunk_size_cap=4000, release=False,
    )
    assert a == b


def test_version_changes_with_chunk_size_cap():
    a = compute_index_version(
        source_hash="a" * 64, config_hash="b" * 64,
        pipeline_version="1", embedding_model=None,
        chunk_size_cap=4000, release=False,
    )
    b = compute_index_version(
        source_hash="a" * 64, config_hash="b" * 64,
        pipeline_version="1", embedding_model=None,
        chunk_size_cap=2000, release=False,
    )
    assert a != b


def test_version_changes_with_release_flag():
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


# ---------------------------------------------------------------------------
# Publishing
# ---------------------------------------------------------------------------

def test_fresh_publish_creates_version_and_current(tmp_path: Path):
    publish(
        tmp_path, version="v1",
        chunks=[_chunk("1.1", "text")],
        manifest=_manifest("v1"),
        report=_report(),
        known_gaps={},
    )
    assert (tmp_path / "versions" / "v1" / "chunks.jsonl").exists()
    assert current_version(tmp_path) == "v1"


def test_rerun_with_same_content_is_idempotent(tmp_path: Path):
    chunks = [_chunk("1.1", "text")]
    publish(tmp_path, version="v1", chunks=chunks,
            manifest=_manifest("v1"), report=_report(), known_gaps={})
    before = (tmp_path / "versions" / "v1" / "chunks.jsonl").read_bytes()

    publish(tmp_path, version="v1", chunks=chunks,
            manifest=_manifest("v1"), report=_report(), known_gaps={})
    after = (tmp_path / "versions" / "v1" / "chunks.jsonl").read_bytes()
    assert before == after


def test_same_version_different_content_rejected(tmp_path: Path):
    publish(tmp_path, version="v1", chunks=[_chunk("1.1", "text")],
            manifest=_manifest("v1"), report=_report(), known_gaps={})
    with pytest.raises(PublishError, match="different content"):
        publish(tmp_path, version="v1", chunks=[_chunk("1.1", "OTHER")],
                manifest=_manifest("v1"), report=_report(), known_gaps={})


def test_publish_failure_leaves_current_intact(tmp_path: Path):
    publish(tmp_path, version="v1", chunks=[_chunk("1.1", "text")],
            manifest=_manifest("v1"), report=_report(), known_gaps={})
    assert current_version(tmp_path) == "v1"

    # Attempt a version that will fail (different content, same version id).
    with pytest.raises(PublishError):
        publish(tmp_path, version="v1", chunks=[_chunk("1.1", "OTHER")],
                manifest=_manifest("v1"), report=_report(), known_gaps={})

    assert current_version(tmp_path) == "v1"


def test_second_version_switches_current(tmp_path: Path):
    publish(tmp_path, version="v1", chunks=[_chunk("1.1", "a")],
            manifest=_manifest("v1"), report=_report(), known_gaps={})
    publish(tmp_path, version="v2", chunks=[_chunk("1.1", "b")],
            manifest=_manifest("v2"), report=_report(), known_gaps={})
    assert current_version(tmp_path) == "v2"
    assert (tmp_path / "versions" / "v1").exists()


def test_staging_directory_cleaned_up_on_failure(tmp_path: Path):
    publish(tmp_path, version="v1", chunks=[_chunk("1.1", "text")],
            manifest=_manifest("v1"), report=_report(), known_gaps={})
    with pytest.raises(PublishError):
        publish(tmp_path, version="v1", chunks=[_chunk("1.1", "OTHER")],
                manifest=_manifest("v1"), report=_report(), known_gaps={})

    versions = list((tmp_path / "versions").iterdir())
    # Only v1 should remain — no staging-* left behind.
    names = sorted(p.name for p in versions)
    assert names == ["v1"], f"unexpected dirs: {names}"
