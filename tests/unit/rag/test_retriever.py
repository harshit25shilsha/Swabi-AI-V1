"""RagRetriever tests. EmbeddingModel and QdrantStore mocked."""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from app.rag.retriever import RagRetriever, RetrieverError
from app.rag.schemas import RetrievedChunk


def _chunk(i: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"id{i}", section_id=f"s{i}", section_path=f"p{i}",
        text=f"text{i}", score=0.8,
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
    assert len(result) == 2
    kwargs = store.search.call_args.kwargs
    assert kwargs["audience"] == "customer"
    assert kwargs["area"] == "swabi"
    assert kwargs["top_k"] == 3
    assert kwargs["min_score"] == 0.5


def test_search_uses_defaults_when_none(embedder, store):
    r = RagRetriever(embedder=embedder, store=store)
    r.search("q", audience="customer")
    kwargs = store.search.call_args.kwargs
    # settings defaults
    assert kwargs["top_k"] >= 1


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
