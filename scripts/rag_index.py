"""Push a published ingestion version into Qdrant.

Reads:  data/rag_index/versions/{version}/chunks.jsonl
Writes: Qdrant collection {collection}_v{version}, then swaps alias
        {collection} to point at it.

Idempotent: rerunning against the same version upserts the same points
(by deterministic point id) and re-asserts the alias.

    python scripts/rag_index.py                     # CURRENT version
    python scripts/rag_index.py --version abc123
    python scripts/rag_index.py --dry-run
    python scripts/rag_index.py --allow-non-release # development only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.core.config import settings  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402
from app.rag.embeddings import EmbeddingError, EmbeddingModel  # noqa: E402
from app.rag.qdrant_store import QdrantStore, QdrantStoreError  # noqa: E402
from app.rag.schemas import Chunk  # noqa: E402

DEFAULT_INDEX = Path("data/rag_index")


def _args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    p.add_argument("--version", type=str, default=None,
                   help="version dir under versions/; default = CURRENT")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--allow-non-release", action="store_true",
                   help="index a non-release artifact (development only)")
    return p.parse_args(argv)


def _resolve_version(index_root: Path, explicit: str | None) -> str:
    if explicit:
        return explicit
    current = index_root / "CURRENT"
    if not current.exists():
        raise SystemExit(
            f"no CURRENT file under {index_root}; run ingestion first "
            "or pass --version"
        )
    return current.read_text(encoding="utf-8").strip()


def _load_chunks(version_dir: Path) -> list[Chunk]:
    path = version_dir / "chunks.jsonl"
    if not path.exists():
        raise SystemExit(f"missing chunks.jsonl at {path}")
    return [
        Chunk.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main(argv=None) -> int:
    setup_logging()
    args = _args(argv)

    version = _resolve_version(args.index, args.version)
    version_dir = args.index / "versions" / version
    if not version_dir.exists():
        raise SystemExit(f"version directory not found: {version_dir}")

    manifest = json.loads(
        (version_dir / "index_version.json").read_text(encoding="utf-8")
    )
    release = bool(manifest.get("release", False))
    if not release and not args.allow_non_release:
        print(
            f"[error] version {version} is not a release build. "
            "Pass --allow-non-release for development indexing.",
            file=sys.stderr,
        )
        return 3

    chunks = _load_chunks(version_dir)
    if not chunks:
        raise SystemExit("no chunks to index")

    print(f"version={version} chunks={len(chunks)} release={release}")
    print(f"embedding_model={settings.rag_embedding_model} "
          f"dim={settings.rag_embedding_dim}")
    print(f"qdrant={settings.qdrant_url} "
          f"collection={settings.qdrant_collection}")

    if args.dry_run:
        print("[dry-run] no writes")
        return 0

    # -- Embed -------------------------------------------------------------
    try:
        embedder = EmbeddingModel()
        vectors = embedder.embed_documents([c.text for c in chunks])
    except EmbeddingError as exc:
        print(f"[error] embeddings: {exc}", file=sys.stderr)
        return 1

    # -- Store -------------------------------------------------------------
    store = QdrantStore()
    if not store.ping():
        print(f"[error] cannot reach Qdrant at {settings.qdrant_url}",
              file=sys.stderr)
        return 1

    try:
        store.ensure_collection(version, dim=settings.rag_embedding_dim)
        written = store.upsert_chunks(version, chunks, vectors)
    except QdrantStoreError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    count = store.count(version)
    if count < len(chunks):
        print(
            f"[error] collection has {count} points, expected {len(chunks)}",
            file=sys.stderr,
        )
        return 4

    # Only swap the alias after count verification.
    store.point_alias_at(version)

    print(f"[ok] indexed {written} points to {store.versioned_name(version)}")
    print(f"[ok] alias {store.alias_name()} -> "
          f"{store.versioned_name(version)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
