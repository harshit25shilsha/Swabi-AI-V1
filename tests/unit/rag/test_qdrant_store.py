"""Qdrant store tests. qdrant-client is mocked."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.rag.qdrant_store import (
    QdrantStore,
    QdrantStoreError,
    _current_content_filter,
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
    for key in (
        "chunk_id",
        "document_id",
        "category",
        "section_id",
        "section_title",
        "section_path",
        "area",
        "audience",
        "source",
        "source_version",
        "last_updated",
        "text",
        "review_flag_ids",
        "policy_id",
        "policy_status",
        "effective_from",
        "expires_on",
        "supersedes",
    ):
        assert key in p, f"missing payload key: {key}"
    assert p["area"] == "swabi"
    assert p["audience"] == "both"
    assert p["policy_status"] == "not_applicable"


def test_payload_formats_policy_dates_for_qdrant_datetime_index():
    chunk = _chunk("1.1", "policy")
    chunk.metadata.policy_id = "policy-1"
    chunk.metadata.policy_status = "approved"
    chunk.metadata.effective_from = "2026-10-01"
    chunk.metadata.expires_on = "2026-10-31"

    payload = _payload(chunk)
    assert payload["effective_from"] == "2026-10-01T00:00:00Z"
    assert payload["expires_on"] == "2026-10-31T00:00:00Z"


def test_current_content_filter_allows_nonpolicy_and_active_policies_only():
    query_filter = _current_content_filter(date(2026, 10, 28))
    branches = query_filter.should
    assert len(branches) == 2

    non_policy = branches[0]
    assert non_policy.key == "policy_status"
    assert non_policy.match.value == "not_applicable"

    active_policy = branches[1]
    must_by_key = {condition.key: condition for condition in active_policy.must}
    assert must_by_key["policy_status"].match.value == "approved"
    assert must_by_key["effective_from"].range.lte.date() == date(2026, 10, 28)
    assert len(active_policy.should) == 2
    expiry_range, no_expiry = active_policy.should
    assert expiry_range.range.gte.date() == date(2026, 10, 28)
    assert no_expiry.is_null.key == "expires_on"


def test_ensure_collection_creates_policy_lifecycle_indexes():
    store = QdrantStore(collection="c")
    client = MagicMock()
    client.get_collections.return_value.collections = []
    client.get_collection.return_value.payload_schema = {}
    with patch.object(store, "_get_client", return_value=client):
        store.ensure_collection("v1", dim=384)

    assert client.create_collection.call_count == 1
    indexed_fields = {
        call.kwargs["field_name"] for call in client.create_payload_index.call_args_list
    }
    assert indexed_fields == {"policy_status", "effective_from", "expires_on"}


# Upsert


def test_upsert_count_mismatch_raises():
    s = QdrantStore(collection="c")
    with patch.object(s, "_get_client", return_value=MagicMock()):
        with pytest.raises(QdrantStoreError, match="count mismatch"):
            s.upsert_chunks(
                "v1",
                [_chunk("1.1", "a")],
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


def test_keyword_search_ranks_matching_payloads_with_bm25():
    store = QdrantStore(collection="c")
    client = MagicMock()
    client.get_aliases.return_value.aliases = [MagicMock(alias_name="c", collection_name="c_v1")]
    client.scroll.return_value = (
        [
            MagicMock(
                payload={
                    "chunk_id": "refund",
                    "section_id": "1.1",
                    "section_path": "Refund policy",
                    "text": "Refunds are available after cancellation.",
                }
            ),
            MagicMock(
                payload={
                    "chunk_id": "escrow",
                    "section_id": "1.2",
                    "section_path": "Escrow",
                    "text": "Escrow protects every booking.",
                }
            ),
            MagicMock(
                payload={
                    "chunk_id": "trip",
                    "section_id": "1.3",
                    "section_path": "Trip",
                    "text": "Trips include hotels and transport.",
                }
            ),
            MagicMock(
                payload={
                    "chunk_id": "vendor",
                    "section_id": "1.4",
                    "section_path": "Vendor",
                    "text": "Vendors respond to customer enquiries.",
                }
            ),
            MagicMock(
                payload={
                    "chunk_id": "wallet",
                    "section_id": "1.5",
                    "section_path": "Wallet",
                    "text": "Wallet withdrawals follow milestone completion.",
                }
            ),
        ],
        None,
    )

    with patch.object(store, "_get_client", return_value=client):
        result = store.keyword_search(
            "refund cancellation",
            top_k=2,
            audience="customer",
            area="swabi",
        )

    assert [chunk.chunk_id for chunk in result] == ["refund"]
    assert result[0].score > 0
    kwargs = client.scroll.call_args.kwargs
    assert kwargs["collection_name"] == "c_v1"
    assert kwargs["with_payload"] is True
    assert kwargs["with_vectors"] is False
    assert kwargs["scroll_filter"].must[0].key == "audience"
    assert kwargs["scroll_filter"].must[1].key == "area"


def test_keyword_search_returns_empty_without_matching_terms():
    store = QdrantStore(collection="c")
    client = MagicMock()
    client.get_aliases.return_value.aliases = [
        MagicMock(alias_name="c", collection_name="c_no_match")
    ]
    client.scroll.return_value = (
        [MagicMock(payload={"chunk_id": "1", "text": "escrow booking"})],
        None,
    )

    with patch.object(store, "_get_client", return_value=client):
        result = store.keyword_search("refund cancellation")

    assert result == []


def test_keyword_search_pins_version_without_resolving_alias():
    store = QdrantStore(collection="c_pinned")
    client = MagicMock()
    client.scroll.return_value = ([], None)

    with patch.object(store, "_get_client", return_value=client):
        result = store.keyword_search("refund", index_version="version-1")

    assert result == []
    client.get_aliases.assert_not_called()
    assert client.scroll.call_args.kwargs["collection_name"] == "c_pinned_vversion-1"


def test_dense_search_pins_versioned_collection():
    store = QdrantStore(collection="c_dense_pinned")
    client = MagicMock()
    client.query_points.return_value.points = []

    with patch.object(store, "_get_client", return_value=client):
        result = store.search([0.1, 0.2], index_version="version-2")

    assert result == []
    assert client.query_points.call_args.kwargs["collection_name"] == "c_dense_pinned_vversion-2"


def test_keyword_search_reuses_index_for_same_published_scope():
    store = QdrantStore(collection="c_cache_reuse")
    client = MagicMock()
    client.get_aliases.return_value.aliases = [
        MagicMock(alias_name="c_cache_reuse", collection_name="c_cache_reuse_v1")
    ]
    client.scroll.return_value = (
        [
            MagicMock(
                payload={
                    "chunk_id": "refund",
                    "section_id": "1.1",
                    "section_path": "Refund policy",
                    "text": "Refunds are available after cancellation.",
                }
            ),
            MagicMock(
                payload={
                    "chunk_id": "escrow",
                    "section_id": "1.2",
                    "section_path": "Escrow",
                    "text": "Escrow protects every booking.",
                }
            ),
            MagicMock(
                payload={
                    "chunk_id": "trip",
                    "section_id": "1.3",
                    "section_path": "Trip",
                    "text": "Trips include hotels and transport.",
                }
            ),
        ],
        None,
    )

    with patch.object(store, "_get_client", return_value=client):
        first = store.keyword_search(
            "refund cancellation",
            audience="customer",
            index_version="v1",
        )
        second = store.keyword_search(
            "cancellation",
            audience="customer",
            index_version="v1",
        )

    assert first[0].chunk_id == "refund"
    assert second[0].chunk_id == "refund"
    client.get_aliases.assert_not_called()
    client.scroll.assert_called_once()


def test_keyword_search_cache_changes_when_published_alias_changes():
    store = QdrantStore(collection="c_cache_version")
    client = MagicMock()
    client.get_aliases.return_value.aliases = [
        MagicMock(alias_name="c_cache_version", collection_name="c_cache_version_v1")
    ]
    client.scroll.side_effect = [
        (
            [
                MagicMock(
                    payload={
                        "chunk_id": "old",
                        "section_id": "1.1",
                        "section_path": "Old",
                        "text": "refund cancellation old policy",
                    }
                ),
                MagicMock(payload={"text": "other"}),
                MagicMock(payload={"text": "another"}),
            ],
            None,
        ),
        (
            [
                MagicMock(
                    payload={
                        "chunk_id": "new",
                        "section_id": "1.2",
                        "section_path": "New",
                        "text": "refund cancellation updated policy",
                    }
                ),
                MagicMock(payload={"text": "other"}),
                MagicMock(payload={"text": "another"}),
            ],
            None,
        ),
    ]

    with patch.object(store, "_get_client", return_value=client):
        old_results = store.keyword_search("refund", audience="customer")
        client.get_aliases.return_value.aliases = [
            MagicMock(alias_name="c_cache_version", collection_name="c_cache_version_v2")
        ]
        new_results = store.keyword_search("refund", audience="customer")

    assert old_results[0].chunk_id == "old"
    assert new_results[0].chunk_id == "new"
    assert [call.kwargs["collection_name"] for call in client.scroll.call_args_list] == [
        "c_cache_version_v1",
        "c_cache_version_v2",
    ]


# Alias swap


def test_alias_created_when_absent():
    s = QdrantStore(collection="c")
    fake = MagicMock()
    fake.get_aliases.return_value = MagicMock(aliases=[])

    with patch.object(s, "_get_client", return_value=fake):
        s.point_alias_at("v1")

    fake.update_collection_aliases.assert_called_once()
    ops = fake.update_collection_aliases.call_args.kwargs["change_aliases_operations"]
    assert len(ops) == 1  # create only


def test_alias_swapped_when_present_and_different():
    s = QdrantStore(collection="c")
    fake = MagicMock()
    fake.get_aliases.return_value = MagicMock(
        aliases=[MagicMock(alias_name="c", collection_name="c_vabc123")]
    )

    with (
        patch.object(s, "_get_client", return_value=fake),
        patch.object(s, "current_versioned_collection", return_value="c_vabc123"),
    ):
        s.point_alias_at("def456")

    ops = fake.update_collection_aliases.call_args.kwargs["change_aliases_operations"]
    assert len(ops) == 2  # delete + create


def test_alias_noop_when_already_targeted():
    s = QdrantStore(collection="c")
    fake = MagicMock()
    fake.get_aliases.return_value = MagicMock(
        aliases=[MagicMock(alias_name="c", collection_name="c_vabc123")]
    )

    with (
        patch.object(s, "_get_client", return_value=fake),
        patch.object(s, "current_versioned_collection", return_value="c_vabc123"),
    ):
        s.point_alias_at("abc123")

    fake.update_collection_aliases.assert_not_called()


# Ping


def test_ping_false_on_error():
    s = QdrantStore(collection="c")
    with patch.object(s, "_get_client", side_effect=QdrantStoreError("no client")):
        assert s.ping() is False


def test_ping_true_on_success():
    s = QdrantStore(collection="c")
    fake = MagicMock()
    with patch.object(s, "_get_client", return_value=fake):
        assert s.ping() is True
