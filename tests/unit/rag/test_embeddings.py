"""Embeddings wrapper tests. fastembed is mocked."""
from __future__ import annotations

import numpy as np
import pytest

from app.rag.embeddings import EmbeddingError, EmbeddingModel


class _Backend:
    """Records the prefixes so we can assert E5 convention."""

    def __init__(self, dim: int = 4):
        self.dim = dim
        self.batches: list[tuple[str, ...]] = []

    def embed(self, items, batch_size=32):
        self.batches.append(tuple(items))
        for _ in items:
            yield np.ones(self.dim, dtype="float32")


def test_documents_use_passage_prefix(monkeypatch):
    backend = _Backend()
    m = EmbeddingModel(model_name="intfloat/multilingual-e5-small", dim=4)
    monkeypatch.setattr(m, "_get_model", lambda: backend)

    m.embed_documents(["hello"])
    assert backend.batches[0][0].startswith("passage: ")


def test_queries_use_query_prefix(monkeypatch):
    backend = _Backend()
    m = EmbeddingModel(model_name="intfloat/multilingual-e5-small", dim=4)
    monkeypatch.setattr(m, "_get_model", lambda: backend)

    m.embed_queries(["hello"])
    assert backend.batches[0][0].startswith("query: ")


def test_no_prefix_for_non_e5(monkeypatch):
    backend = _Backend()
    m = EmbeddingModel(model_name="BAAI/bge-small-en", dim=4)
    monkeypatch.setattr(m, "_get_model", lambda: backend)

    m.embed_queries(["hello"])
    assert not backend.batches[0][0].startswith(("query: ", "passage: "))


def test_dimension_mismatch_raises(monkeypatch):
    backend = _Backend(dim=8)
    m = EmbeddingModel(model_name="intfloat/multilingual-e5-small", dim=4)
    monkeypatch.setattr(m, "_get_model", lambda: backend)

    with pytest.raises(EmbeddingError, match="dim"):
        m.embed_documents(["hello"])


def test_vectors_are_l2_normalised(monkeypatch):
    class _B:
        def embed(self, items, batch_size=32):
            for _ in items:
                yield np.array([3.0, 4.0], dtype="float32")

    m = EmbeddingModel(model_name="intfloat/multilingual-e5-small", dim=2)
    monkeypatch.setattr(m, "_get_model", lambda: _B())

    v = m.embed_queries(["x"])
    assert np.isclose(np.linalg.norm(v[0]), 1.0, atol=1e-6)


def test_empty_input_returns_empty_array():
    m = EmbeddingModel(model_name="intfloat/multilingual-e5-small", dim=4)
    v = m.embed_documents([])
    assert v.shape == (0, 4)
