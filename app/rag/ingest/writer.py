"""Non-destructive artefact publishing.

Layout::

    data/rag_index/
        CURRENT
        versions/{version}/
            chunks.jsonl
            known_gaps.json
            index_version.json
            ingestion_report.json

Rules:
  * Content-addressed: version covers source, config, pipeline,
    embedding model, chunk_size_cap, release.
  * Fresh version: write to staging-<uuid>, validate, atomic rename.
  * Existing version: identical content → refresh report only.
                      Different content → PublishError.
  * CURRENT and staging use unique temp names (uuid4) — safe under
    concurrent runs.
  * Manifest is validated against the staged artefacts before publish:
    version matches directory name; chunk_count matches actual.
  * A failure at any point leaves CURRENT pointing at a valid version.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path

from app.rag.schemas import Chunk, IndexVersion, IngestionReport

CURRENT_NAME = "CURRENT"
VERSIONS_DIR = "versions"


class PublishError(Exception):
    pass


def compute_index_version(
    *,
    source_hash: str,
    config_hash: str,
    pipeline_version: str,
    embedding_model: str | None,
    chunk_size_cap: int,
    release: bool,
) -> str:
    h = hashlib.sha256()
    for part in (
        "swabi-rag-index/v1",
        source_hash,
        config_hash,
        pipeline_version,
        embedding_model or "none",
        str(chunk_size_cap),
        "1" if release else "0",
    ):
        h.update(part.encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()[:12]


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _serialize_chunks(chunks: list[Chunk]) -> str:
    return "\n".join(c.model_dump_json() for c in chunks) + "\n"


def _serialize_json(payload) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False,
                      sort_keys=True) + "\n"


def _write_staging(
    staging: Path,
    *,
    chunks: list[Chunk],
    manifest: IndexVersion,
    report: IngestionReport,
    known_gaps: dict[str, list[str]],
) -> None:
    staging.mkdir(parents=True, exist_ok=False)
    (staging / "chunks.jsonl").write_text(_serialize_chunks(chunks),
                                          encoding="utf-8")
    (staging / "known_gaps.json").write_text(_serialize_json(known_gaps),
                                             encoding="utf-8")
    (staging / "index_version.json").write_text(
        manifest.model_dump_json(indent=2) + "\n", encoding="utf-8",
    )
    (staging / "ingestion_report.json").write_text(
        report.model_dump_json(indent=2) + "\n", encoding="utf-8",
    )


def _validate_staged(
    staging: Path, *, expected_version: str, expected_chunks: int,
) -> None:
    lines = (staging / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
    payload = [line for line in lines if line.strip()]
    if len(payload) != expected_chunks:
        raise PublishError(
            f"staged chunk count mismatch: {len(payload)} != {expected_chunks}"
        )
    for line in payload:
        Chunk.model_validate_json(line)

    manifest = IndexVersion.model_validate_json(
        (staging / "index_version.json").read_text(encoding="utf-8")
    )
    if manifest.index_version != expected_version:
        raise PublishError(
            f"staged manifest version mismatch: {manifest.index_version} != "
            f"{expected_version}"
        )
    if manifest.chunk_count != expected_chunks:
        raise PublishError(
            f"staged manifest chunk_count mismatch: "
            f"{manifest.chunk_count} != {expected_chunks}"
        )


def _content_equal(a: Path, b: Path) -> bool:
    for name in ("chunks.jsonl", "known_gaps.json", "index_version.json"):
        pa, pb = a / name, b / name
        if not pa.exists() or not pb.exists():
            return False
        if pa.read_bytes() != pb.read_bytes():
            return False
    return True


def publish(
    out_root: Path,
    *,
    version: str,
    chunks: list[Chunk],
    manifest: IndexVersion,
    report: IngestionReport,
    known_gaps: dict[str, list[str]],
) -> Path:
    out_root.mkdir(parents=True, exist_ok=True)
    versions_root = out_root / VERSIONS_DIR
    versions_root.mkdir(exist_ok=True)

    final_dir = versions_root / version
    staging_dir = versions_root / f"{version}.staging-{uuid.uuid4().hex}"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)

    try:
        _write_staging(
            staging_dir,
            chunks=chunks, manifest=manifest, report=report,
            known_gaps=known_gaps,
        )
        _validate_staged(
            staging_dir,
            expected_version=version,
            expected_chunks=len(chunks),
        )

        if final_dir.exists():
            if not _content_equal(staging_dir, final_dir):
                raise PublishError(
                    f"version {version} already exists with different "
                    "content; bump pipeline_version or inputs"
                )
            (final_dir / "ingestion_report.json").write_text(
                report.model_dump_json(indent=2) + "\n", encoding="utf-8",
            )
            shutil.rmtree(staging_dir)
            _atomic_write_text(out_root / CURRENT_NAME, version + "\n")
            return final_dir

        os.replace(staging_dir, final_dir)
        _atomic_write_text(out_root / CURRENT_NAME, version + "\n")
        return final_dir
    except Exception:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
        raise


def current_version(out_root: Path) -> str | None:
    p = out_root / CURRENT_NAME
    if not p.exists():
        return None
    return p.read_text(encoding="utf-8").strip() or None
