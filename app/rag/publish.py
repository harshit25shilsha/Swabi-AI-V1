"""Atomic versioned publication of ingestion artifacts."""
from __future__ import annotations

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


def _atomic_write(path: Path, text: str) -> None:
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

    try:
        staging_dir.mkdir(parents=True, exist_ok=False)
        (staging_dir / "chunks.jsonl").write_text(
            "\n".join(c.model_dump_json() for c in chunks) + "\n",
            encoding="utf-8",
        )
        (staging_dir / "index_version.json").write_text(
            manifest.model_dump_json(indent=2) + "\n", encoding="utf-8",
        )
        (staging_dir / "known_gaps.json").write_text(
            json.dumps(known_gaps, indent=2, ensure_ascii=False,
                       sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (staging_dir / "ingestion_report.json").write_text(
            report.model_dump_json(indent=2) + "\n", encoding="utf-8",
        )

        # Validate reload
        payload = [
            Chunk.model_validate_json(line)
            for line in (staging_dir / "chunks.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
        if len(payload) != len(chunks):
            raise PublishError(
                f"chunk count mismatch after write: {len(payload)} != {len(chunks)}"
            )

        if final_dir.exists():
            # Compare content; identical -> idempotent.
            same = all(
                (staging_dir / n).read_bytes() == (final_dir / n).read_bytes()
                for n in ("chunks.jsonl", "index_version.json", "known_gaps.json")
            )
            if not same:
                raise PublishError(
                    f"version {version} exists with different content"
                )
            (final_dir / "ingestion_report.json").write_text(
                report.model_dump_json(indent=2) + "\n", encoding="utf-8",
            )
            shutil.rmtree(staging_dir)
            _atomic_write(out_root / CURRENT_NAME, version + "\n")
            return final_dir

        os.replace(staging_dir, final_dir)
        _atomic_write(out_root / CURRENT_NAME, version + "\n")
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
