"""RagRetriever tests. EmbeddingModel and QdrantStore mocked."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from app.rag.retriever import (
    RagRetriever,
    RetrieverError,
    reciprocal_rank_fusion,
)
from app.rag.schemas import RetrievedChunk


def _chunk(i: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"id{i}",
        section_id=f"s{i}",
        section_path=f"p{i}",
        text=f"text{i}",
        score=0.8,
    )


@pytest.fixture
def embedder():
    e = MagicMock()
    e.embed_queries.return_value = np.ones((1, 384), dtype="float32")
    return e


@pytest.fixture
def store():
    s = MagicMock()
    s.search.return_value = [_chunk(1), _chunk(2)]
    s.keyword_search.return_value = [_chunk(2), _chunk(3)]
    return s


def test_empty_query_returns_empty(embedder, store):
    r = RagRetriever(embedder=embedder, store=store)
    assert r.search("", audience="customer") == []
    embedder.embed_queries.assert_not_called()


def test_search_passes_audience_and_params(embedder, store):
    r = RagRetriever(embedder=embedder, store=store)
    result = r.search(
        "how does escrow work",
        audience="customer",
        area="swabi",
        top_k=3,
        min_score=0.5,
    )
    assert len(result) == 3
    dense_kwargs = store.search.call_args.kwargs
    assert dense_kwargs["audience"] == "customer"
    assert dense_kwargs["area"] == "swabi"
    assert dense_kwargs["top_k"] == 12
    assert dense_kwargs["min_score"] == 0.5
    keyword_kwargs = store.keyword_search.call_args.kwargs
    assert keyword_kwargs["audience"] == "customer"
    assert keyword_kwargs["area"] == "swabi"
    assert keyword_kwargs["top_k"] == 12
    assert [chunk.chunk_id for chunk in result] == ["id2", "id1", "id3"]


def test_search_uses_defaults_when_none(embedder, store):
    r = RagRetriever(embedder=embedder, store=store)
    r.search("q", audience="customer")
    assert store.search.call_args.kwargs["top_k"] >= 4
    assert store.keyword_search.call_args.kwargs["top_k"] >= 4


def test_search_pins_both_modes_to_requested_index_version(embedder, store):
    r = RagRetriever(embedder=embedder, store=store)

    r.search("q", audience="customer", index_version="v123")

    assert store.search.call_args.kwargs["index_version"] == "v123"
    assert store.keyword_search.call_args.kwargs["index_version"] == "v123"


def test_search_passes_tuning_options_to_hybrid_fusion(embedder, store):
    r = RagRetriever(embedder=embedder, store=store)

    result = r.search(
        "q",
        audience="customer",
        rrf_rank_constant=20,
        candidate_multiplier=2,
    )

    assert store.search.call_args.kwargs["top_k"] == 10
    assert store.keyword_search.call_args.kwargs["top_k"] == 10
    assert [chunk.chunk_id for chunk in result] == ["id2", "id1", "id3"]
    assert result[0].score == pytest.approx((1 / 22) + (1 / 21))


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"rrf_rank_constant": 0}, "rrf_rank_constant must be at least 1"),
        ({"candidate_multiplier": 0}, "candidate_multiplier must be at least 1"),
    ],
)
def test_search_rejects_invalid_tuning_options(embedder, store, options, message):
    r = RagRetriever(embedder=embedder, store=store)
    with pytest.raises(ValueError, match=message):
        r.search("q", audience="customer", **options)


def test_embedding_error_wrapped(embedder, store):
    from app.rag.embeddings import EmbeddingError

    embedder.embed_queries.side_effect = EmbeddingError("boom")
    r = RagRetriever(embedder=embedder, store=store)
    with pytest.raises(RetrieverError, match="embedding failed"):
        r.search("q", audience="customer")


def test_qdrant_error_wrapped(embedder, store):
    from app.rag.qdrant_store import QdrantStoreError

    store.search.side_effect = QdrantStoreError("down")
    r = RagRetriever(embedder=embedder, store=store)
    with pytest.raises(RetrieverError, match="qdrant search failed"):
        r.search("q", audience="customer")


def test_bm25_error_wrapped(embedder, store):
    from app.rag.qdrant_store import QdrantStoreError

    store.keyword_search.side_effect = QdrantStoreError("down")
    r = RagRetriever(embedder=embedder, store=store)
    with pytest.raises(RetrieverError, match="BM25 search failed"):
        r.search("q", audience="customer")


def test_reciprocal_rank_fusion_combines_and_deduplicates():
    dense = [_chunk(1), _chunk(2)]
    keyword = [_chunk(2), _chunk(3)]

    result = reciprocal_rank_fusion([dense, keyword], top_k=3)

    assert [chunk.chunk_id for chunk in result] == ["id2", "id1", "id3"]
    assert result[0].score == pytest.approx((1 / 62) + (1 / 61))
    assert result[1].score == pytest.approx(1 / 61)


def test_reciprocal_rank_fusion_rejects_invalid_rank_constant():
    with pytest.raises(ValueError, match="rank_constant must be at least 1"):
        reciprocal_rank_fusion([[_chunk(1)]], top_k=1, rank_constant=0)


def test_empty_embedding_result_still_uses_keyword_search(embedder, store):
    embedder.embed_queries.return_value = np.zeros((0, 384), dtype="float32")
    r = RagRetriever(embedder=embedder, store=store)

    result = r.search("escrow", audience="customer")

    store.search.assert_not_called()
    store.keyword_search.assert_called_once()
    assert [chunk.chunk_id for chunk in result] == ["id2", "id3"]
