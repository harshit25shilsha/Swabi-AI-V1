"""Qdrant store tests. qdrant-client is mocked."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.rag.qdrant_store import (
    QdrantStore,
    QdrantStoreError,
    _payload,
    _point_id,
)
from app.rag.schemas import (
    Audience,
    Chunk,
    ChunkMetadata,
    KnowledgeArea,
)


def _chunk(sid: str, text: str) -> Chunk:
    return Chunk(
        text=text,
        metadata=ChunkMetadata(
            chunk_id=f"id-{sid}",
            document_id="doc",
            category="A",
            section_id=sid,
            section_title=sid,
            section_path=sid,
            area=KnowledgeArea.SWABI,
            audience=Audience.BOTH,
            source="KB",
            source_version="1.0",
            last_updated="2026-10",
        ),
    )


# Names

def test_versioned_name():
    s = QdrantStore(collection="swabi_chunks")
    assert s.versioned_name("abc123") == "swabi_chunks_vabc123"
    assert s.alias_name() == "swabi_chunks"


# Point id

def test_point_id_deterministic():
    assert _point_id("chunk-abc") == _point_id("chunk-abc")


def test_point_id_valid_uuid_shape():
    import uuid
    uuid.UUID(_point_id("chunk-abc"))


# Payload

def test_payload_has_required_fields():
    p = _payload(_chunk("1.1", "text"))
    for key in ("chunk_id", "document_id", "category", "section_id",
                "section_title", "section_path", "area", "audience",
                "source", "source_version", "last_updated", "text",
                "review_flag_ids"):
        assert key in p, f"missing payload key: {key}"
    assert p["area"] == "swabi"
    assert p["audience"] == "both"



# Upsert

def test_upsert_count_mismatch_raises():
    s = QdrantStore(collection="c")
    with patch.object(s, "_get_client", return_value=MagicMock()):
        with pytest.raises(QdrantStoreError, match="count mismatch"):
            s.upsert_chunks(
                "v1", [_chunk("1.1", "a")],
                np.zeros((0, 4), dtype="float32"),
            )


def test_upsert_writes_points():
    s = QdrantStore(collection="c")
    fake = MagicMock()
    with patch.object(s, "_get_client", return_value=fake):
        n = s.upsert_chunks(
            "abc123",
            [_chunk("1.1", "a"), _chunk("1.2", "b")],
            np.ones((2, 4), dtype="float32"),
        )
    assert n == 2
    fake.upsert.assert_called_once()
    kwargs = fake.upsert.call_args.kwargs
    assert kwargs["collection_name"] == "c_vabc123"
    assert len(kwargs["points"]) == 2



# Alias swap

def test_alias_created_when_absent():
    s = QdrantStore(collection="c")
    fake = MagicMock()
    fake.get_aliases.return_value = MagicMock(aliases=[])

    with patch.object(s, "_get_client", return_value=fake):
        s.point_alias_at("v1")

    fake.update_collection_aliases.assert_called_once()
    ops = fake.update_collection_aliases.call_args.kwargs[
        "change_aliases_operations"
    ]
    assert len(ops) == 1  # create only


def test_alias_swapped_when_present_and_different():
    s = QdrantStore(collection="c")
    fake = MagicMock()
    fake.get_aliases.return_value = MagicMock(
        aliases=[MagicMock(alias_name="c", collection_name="c_vabc123")]
    )

    with patch.object(s, "_get_client", return_value=fake), \
         patch.object(s, "current_versioned_collection",
                      return_value="c_vabc123"):
        s.point_alias_at("def456")

    ops = fake.update_collection_aliases.call_args.kwargs[
        "change_aliases_operations"
    ]
    assert len(ops) == 2  # delete + create


def test_alias_noop_when_already_targeted():
    s = QdrantStore(collection="c")
    fake = MagicMock()
    fake.get_aliases.return_value = MagicMock(
        aliases=[MagicMock(alias_name="c", collection_name="c_vabc123")]
    )

    with patch.object(s, "_get_client", return_value=fake), \
         patch.object(s, "current_versioned_collection",
                      return_value="c_vabc123"):
        s.point_alias_at("abc123")

    fake.update_collection_aliases.assert_not_called()

# Ping


def test_ping_false_on_error():
    s = QdrantStore(collection="c")
    with patch.object(s, "_get_client",
                      side_effect=QdrantStoreError("no client")):
        assert s.ping() is False


def test_ping_true_on_success():
    s = QdrantStore(collection="c")
    fake = MagicMock()
    with patch.object(s, "_get_client", return_value=fake):
        assert s.ping() is True
