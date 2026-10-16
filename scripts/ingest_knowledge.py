"""Thin CLI for knowledge ingestion.

    python scripts/ingest_knowledge.py --verify-yaml
    python scripts/ingest_knowledge.py --dry-run
    python scripts/ingest_knowledge.py                # non-release build
    python scripts/ingest_knowledge.py --release

Embeddings and Qdrant indexing are Batch 3.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import UTC, datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.rag.ingest import (  # noqa: E402
    PIPELINE_VERSION,
    ConfigError,
    PublishError,
    SourceError,
    build_chunks,
    collect_review_flags,
    compute_index_version,
    excluded_section_ids,
    has_preamble,
    load_config,
    parse_markdown,
    publish,
    sha256_of,
    validate_config,
    validate_unique_section_ids,
    validate_unique_titles_within_parents,
)
from app.rag.schemas import IndexVersion, IngestionReport  # noqa: E402

DEFAULT_CONFIG = Path("config/knowledge_sections.yaml")
DEFAULT_OUTPUT = Path("data/rag_index")
DEFAULT_CAP = 4000


def _collect_gaps(cfg: dict) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}

    def _walk(s: dict) -> None:
        if s.get("known_gaps"):
            out[s["id"]] = [str(x).strip() for x in s["known_gaps"]]
        for child in s.get("split_into") or []:
            _walk(child)

    for s in cfg["sections"]:
        _walk(s)
    return out


def _parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--release", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--verify-yaml", action="store_true")
    p.add_argument("--chunk-size-cap", type=int, default=DEFAULT_CAP)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    try:
        cfg = load_config(args.config)
        validate_config(cfg, release=args.release)

        if args.verify_yaml:
            print(f"YAML ok: {len(cfg['sections'])} sections, "
                  f"{len(cfg['categories'])} categories, "
                  f"release={'yes' if args.release else 'no'}")
            return 0

        config_hash = hashlib.sha256(args.config.read_bytes()).hexdigest()
        source_path = Path(cfg["source_path"])
        if not source_path.exists():
            raise ConfigError(f"source not found: {source_path}")
        source_text = source_path.read_text(encoding="utf-8")
        actual_hash = sha256_of(source_text)
        expected = cfg.get("source_hash_expected")
        if expected is not None and expected != actual_hash:
            raise ConfigError(
                f"source hash mismatch\n  expected: {expected}\n"
                f"  actual:   {actual_hash}"
            )

        root = parse_markdown(source_text)
        validate_unique_section_ids(root)

        split_parent_ids: set[str] = set()
        def _collect_split_parents(spec: dict) -> None:
            if "split_into" in spec:
                split_parent_ids.add(spec["id"])
            for child in spec.get("split_into") or []:
                _collect_split_parents(child)
        for spec in cfg["sections"]:
            _collect_split_parents(spec)

        validate_unique_titles_within_parents(root, split_parent_ids)

        build = build_chunks(
            cfg, root,
            chunk_size_cap=args.chunk_size_cap,
            excluded=excluded_section_ids(cfg),
        )

        if has_preamble(root):
            build.warnings.append(
                "preamble: text found before the first heading; not chunked"
            )

        # Release safety: check EVERY config flag, not just chunk flags.
        all_flags = collect_review_flags(cfg)
        unconfirmed = [f.id for f in all_flags if not f.confirmed]
        if args.release and unconfirmed:
            print("[error] --release blocked by unconfirmed review flags:",
                  file=sys.stderr)
            for fid in unconfirmed:
                print(f"  {fid}", file=sys.stderr)
            return 3
        if args.release and build.over_cap:
            print(f"[error] --release blocked by {len(build.over_cap)} "
                  "over-cap chunks:", file=sys.stderr)
            for cid in build.over_cap[:20]:
                print(f"  {cid}", file=sys.stderr)
            return 3

        version = compute_index_version(
            source_hash=actual_hash,
            config_hash=config_hash,
            pipeline_version=PIPELINE_VERSION,
            embedding_model=None,
            chunk_size_cap=args.chunk_size_cap,
            release=bool(args.release),
        )
        manifest = IndexVersion(
            index_version=version,
            pipeline_version=PIPELINE_VERSION,
            source_path=str(source_path),
            source_hash=actual_hash,
            source_version=cfg["source_version"],
            config_hash=config_hash,
            embedding_model=None,
            release=bool(args.release),
            chunk_count=len(build.chunks),
            has_embeddings=False,
        )

        per_area, per_audience, per_cat = {}, {}, {}
        for c in build.chunks:
            area = c.metadata.area.value
            aud = c.metadata.audience.value
            cat = c.metadata.category
            per_area[area] = per_area.get(area, 0) + 1
            per_audience[aud] = per_audience.get(aud, 0) + 1
            per_cat[cat] = per_cat.get(cat, 0) + 1

        report = IngestionReport(
            built_at=datetime.now(UTC).isoformat(),
            source_path=str(source_path),
            source_hash=actual_hash,
            source_version=cfg["source_version"],
            config_hash=config_hash,
            config_path=str(args.config),
            pipeline_version=PIPELINE_VERSION,
            total_chunks=len(build.chunks),
            per_area=per_area,
            per_audience=per_audience,
            per_category=per_cat,
            over_cap=build.over_cap,
            skipped_subsections=build.skipped_subsections,
            known_gaps=_collect_gaps(cfg),
            review_flags=all_flags,
            coverage=build.coverage,
            excluded_sections=excluded_section_ids(cfg),
            warnings=build.warnings,
        )

        print(f"version={version} release={manifest.release} "
              f"chunks={len(build.chunks)} over_cap={len(build.over_cap)} "
              f"unconfirmed_flags={len(unconfirmed)}")
        print(f"source_hash={actual_hash}")
        print(f"config_hash={config_hash}")

        if args.dry_run:
            print("[dry-run] no files written")
            return 0

        publish(
            args.output,
            version=version,
            chunks=build.chunks,
            manifest=manifest,
            report=report,
            known_gaps=report.known_gaps,
        )
        print(f"[ok] published to {args.output}/versions/{version}/")
        return 0

    except (ConfigError, SourceError, PublishError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"[error] unexpected: {exc}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
