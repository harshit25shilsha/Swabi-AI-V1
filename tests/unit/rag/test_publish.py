"""Publication tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.rag.publish import PublishError, current_version, publish
from app.rag.schemas import (
    Audience,
    Chunk,
    ChunkMetadata,
    IndexVersion,
    IngestionReport,
    KnowledgeArea,
)


def _chunk(sid: str, text: str) -> Chunk:
    return Chunk(
        text=text,
        metadata=ChunkMetadata(
            chunk_id=f"id-{sid}", document_id="doc",
            category="A", section_id=sid, section_title=sid,
            section_path=sid,
            area=KnowledgeArea.SWABI, audience=Audience.BOTH,
            source="KB", source_version="1.0", last_updated="2026-10",
        ),
    )


def _manifest(version: str, count: int = 1) -> IndexVersion:
    return IndexVersion(
        index_version=version, pipeline_version="2A1",
        source_path="x.md", source_hash="a" * 64, source_version="1.0",
        config_hash="b" * 64, chunk_size_cap=4000,
        release=False, chunk_count=count,
    )


def _report() -> IngestionReport:
    return IngestionReport(
        built_at="2026-10-16T00:00:00+00:00",
        source_path="x.md", source_hash="a" * 64, source_version="1.0",
        config_hash="b" * 64, config_path="c.yaml",
        pipeline_version="2A1", total_chunks=1,
    )


def test_fresh_publish(tmp_path: Path):
    publish(tmp_path, version="v1", chunks=[_chunk("1.1", "x")],
            manifest=_manifest("v1"), report=_report(), known_gaps={})
    assert (tmp_path / "versions" / "v1" / "chunks.jsonl").exists()
    assert current_version(tmp_path) == "v1"


def test_rerun_same_content_is_idempotent(tmp_path: Path):
    chunks = [_chunk("1.1", "x")]
    publish(tmp_path, version="v1", chunks=chunks,
            manifest=_manifest("v1"), report=_report(), known_gaps={})
    publish(tmp_path, version="v1", chunks=chunks,
            manifest=_manifest("v1"), report=_report(), known_gaps={})
    assert current_version(tmp_path) == "v1"


def test_different_content_same_version_rejected(tmp_path: Path):
    publish(tmp_path, version="v1", chunks=[_chunk("1.1", "a")],
            manifest=_manifest("v1"), report=_report(), known_gaps={})
    with pytest.raises(PublishError, match="different content"):
        publish(tmp_path, version="v1", chunks=[_chunk("1.1", "b")],
                manifest=_manifest("v1"), report=_report(), known_gaps={})


def test_no_staging_left_behind(tmp_path: Path):
    publish(tmp_path, version="v1", chunks=[_chunk("1.1", "a")],
            manifest=_manifest("v1"), report=_report(), known_gaps={})
    with pytest.raises(PublishError):
        publish(tmp_path, version="v1", chunks=[_chunk("1.1", "b")],
                manifest=_manifest("v1"), report=_report(), known_gaps={})
    names = sorted(p.name for p in (tmp_path / "versions").iterdir())
    assert names == ["v1"]
