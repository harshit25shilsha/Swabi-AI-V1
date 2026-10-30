"""Knowledge index version authority checks."""
from __future__ import annotations

from unittest.mock import MagicMock

from app.rag.qdrant_store import QdrantStoreError
from app.rag.version_check import check_version_authority


def test_version_authority_accepts_matching_current_and_alias(tmp_path):
    current = tmp_path / "CURRENT"
    current.write_text("abc123\n", encoding="utf-8")
    store = MagicMock()
    store.current_versioned_collection.return_value = "swabi_chunks_vabc123"
    store.versioned_name.return_value = "swabi_chunks_vabc123"

    status = check_version_authority(current_file=current, store=store)

    assert status.ok is True
    assert status.local_version == "abc123"


def test_version_authority_rejects_stale_qdrant_alias(tmp_path):
    current = tmp_path / "CURRENT"
    current.write_text("new-version", encoding="utf-8")
    store = MagicMock()
    store.current_versioned_collection.return_value = "swabi_chunks_vold-version"
    store.versioned_name.return_value = "swabi_chunks_vnew-version"

    status = check_version_authority(current_file=current, store=store)

    assert status.ok is False
    assert "does not match" in status.reason


def test_version_authority_rejects_missing_or_empty_current(tmp_path):
    missing = check_version_authority(current_file=tmp_path / "CURRENT")
    assert missing.ok is False
    assert missing.reason == "CURRENT file missing"

    current = tmp_path / "CURRENT"
    current.write_text(" \n", encoding="utf-8")
    empty = check_version_authority(current_file=current)
    assert empty.ok is False
    assert empty.reason == "CURRENT file empty"


def test_version_authority_fails_closed_when_qdrant_is_unavailable(tmp_path):
    current = tmp_path / "CURRENT"
    current.write_text("abc123", encoding="utf-8")
    store = MagicMock()
    store.current_versioned_collection.side_effect = QdrantStoreError(
        "connection failed"
    )

    status = check_version_authority(current_file=current, store=store)

    assert status.ok is False
    assert status.local_version == "abc123"
    assert status.reason == "qdrant unavailable: connection failed"
