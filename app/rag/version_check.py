"""Verify the Qdrant alias points at the local CURRENT version.

The write side publishes to ``data/rag_index/CURRENT`` and atomically
swaps the Qdrant alias. If the two disagree, either the index was
rebuilt but not yet re-indexed into Qdrant, or a new version is being
promoted and reads are catching an inconsistent state.

Either way: refuse to answer. Serving a stale policy is worse than an
insufficient-context response.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.core.logging import get_logger
from app.rag.qdrant_store import QdrantStore, QdrantStoreError

log = get_logger(__name__)

_CURRENT_FILE = Path("data/rag_index/CURRENT")


@dataclass
class VersionStatus:
    ok: bool
    local_version: str
    alias_collection: str | None
    reason: str = ""


def check_version_authority(
    *,
    current_file: Path | None = None,
    store: QdrantStore | None = None,
) -> VersionStatus:
    """Compare local CURRENT with the Qdrant alias target."""
    path = current_file or _CURRENT_FILE

    if not path.exists():
        return VersionStatus(
            ok=False,
            local_version="",
            alias_collection=None,
            reason="CURRENT file missing",
        )

    local_version = path.read_text(encoding="utf-8").strip()
    if not local_version:
        return VersionStatus(
            ok=False,
            local_version="",
            alias_collection=None,
            reason="CURRENT file empty",
        )

    try:
        s = store or QdrantStore()
        alias_collection = s.current_versioned_collection()
        expected = s.versioned_name(local_version)
    except QdrantStoreError as exc:
        return VersionStatus(
            ok=False,
            local_version=local_version,
            alias_collection=None,
            reason=f"qdrant unavailable: {exc}",
        )

    if alias_collection != expected:
        return VersionStatus(
            ok=False,
            local_version=local_version,
            alias_collection=alias_collection,
            reason=(
                f"alias={alias_collection!r} does not match "
                f"expected={expected!r}"
            ),
        )

    return VersionStatus(
        ok=True,
        local_version=local_version,
        alias_collection=alias_collection,
    )
