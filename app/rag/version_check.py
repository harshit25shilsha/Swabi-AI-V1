
"""Verify Qdrant version authority with process-local validation caching.

The local CURRENT file is checked on every request. A successful Qdrant
alias validation is reused while CURRENT remains on the same immutable
index version.

Retrieval uses the physical versioned collection, not the alias. When
CURRENT changes, alias validation runs again before the new version is used.
Failed validations are never cached.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

from app.core.logging import get_logger
from app.rag.qdrant_store import QdrantStore, QdrantStoreError

log = get_logger(__name__)

_CURRENT_FILE = Path("data/rag_index/CURRENT")

# One successful validation per store/path/current-version combination.
_validated_versions: dict[
    tuple[str, int, str],
    VersionStatus,
] = {}

_version_cache_lock = threading.Lock()


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
    """Validate CURRENT against Qdrant, rechecking when CURRENT changes."""

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

    s = store or QdrantStore()

    cache_base = (str(path.resolve()), id(s))
    cache_key = (*cache_base, local_version)

    # Keep the lock through validation so concurrent first requests
    # do not all issue the same Qdrant alias request.
    with _version_cache_lock:
        cached = _validated_versions.get(cache_key)

        if cached is not None:
            return cached

        try:
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
                    f"alias={alias_collection!r} "
                    f"does not match expected={expected!r}"
                ),
            )

        status = VersionStatus(
            ok=True,
            local_version=local_version,
            alias_collection=alias_collection,
        )

        # Keep only the latest validated version for this store/path.
        old_keys = [
            key
            for key in _validated_versions
            if key[:2] == cache_base
        ]

        for key in old_keys:
            _validated_versions.pop(key, None)

        _validated_versions[cache_key] = status

        log.info(
            "rag version authority validated",
            extra={
                "event": "rag_version_authority_validated",
                "knowledge_version": local_version,
                "alias_collection": alias_collection,
                "validation_cache": "miss",
            },
        )

        return status
