"""Thin CLI for Phase 2 RAG ingestion.

    python scripts/rag_ingest.py --verify-config
    python scripts/rag_ingest.py --dry-run
    python scripts/rag_ingest.py               # non-release build
    python scripts/rag_ingest.py --release     # blocks on unconfirmed flags
                                               # or over-cap chunks
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.rag.config import ConfigError, load_config, validate_config  # noqa: E402
from app.rag.ingest import IngestError, ingest  # noqa: E402
from app.rag.publish import PublishError, publish  # noqa: E402

DEFAULT_CONFIG = Path("config/knowledge_sections.yaml")
DEFAULT_OUTPUT = Path("data/rag_index")
DEFAULT_CAP = 4000


def _args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--release", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--verify-config", action="store_true")
    p.add_argument("--chunk-size-cap", type=int, default=DEFAULT_CAP)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _args(argv)
    try:
        if args.verify_config:
            cfg = load_config(args.config)
            validate_config(cfg, release=args.release)
            print(f"config ok: {len(cfg['sections'])} sections, "
                  f"{len(cfg['categories'])} categories, "
                  f"release={'yes' if args.release else 'no'}")
            return 0

        result = ingest(
            args.config,
            release=args.release,
            chunk_size_cap=args.chunk_size_cap,
        )
        print(f"version={result.manifest.index_version} "
              f"release={result.manifest.release} "
              f"chunks={result.manifest.chunk_count} "
              f"over_cap={len(result.report.over_cap)}")
        print(f"per_area={result.report.per_area}")
        print(f"per_audience={result.report.per_audience}")
        print(f"excluded={len(result.report.excluded)} "
              f"skipped={len(result.report.skipped)} "
              f"warnings={len(result.report.warnings)}")

        if args.dry_run:
            print("[dry-run] no files written")
            return 0

        publish(
            args.output,
            version=result.manifest.index_version,
            chunks=result.chunks,
            manifest=result.manifest,
            report=result.report,
            known_gaps=result.known_gaps,
        )
        print(f"[ok] published to {args.output}/versions/"
              f"{result.manifest.index_version}/")
        return 0
    except (ConfigError, IngestError, PublishError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"[error] unexpected: {exc}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
