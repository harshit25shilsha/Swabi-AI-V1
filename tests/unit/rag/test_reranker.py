"""Tests for local cross-encoder reranking."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.rag.reranker import CrossEncoderReranker, RerankerError
from app.rag.schemas import RetrievedChunk


def _chunk(chunk_id: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        section_id="section",
        section_path="Policies",
        text=f"evidence {chunk_id}",
        score=0.1,
    )


def test_cross_encoder_reranks_candidates_and_replaces_scores():
    model = MagicMock()
    model.rerank_pairs.return_value = [0.2, 0.9, 0.9]
    reranker = CrossEncoderReranker("local/model", model=model)

    result = reranker.rerank(
        "question",
        [_chunk("first"), _chunk("second"), _chunk("third")],
        top_k=2,
    )

    assert [chunk.chunk_id for chunk in result] == ["second", "third"]
    assert [chunk.score for chunk in result] == [0.9, 0.9]
    model.rerank_pairs.assert_called_once_with(
        [
            ("question", "evidence first"),
            ("question", "evidence second"),
            ("question", "evidence third"),
        ]
    )


def test_cross_encoder_returns_empty_without_loading_model():
    reranker = CrossEncoderReranker("local/model")

    assert reranker.rerank("question", [], top_k=3) == []
    assert reranker.rerank("question", [_chunk("one")], top_k=0) == []


@pytest.mark.parametrize("scores", [[0.1], [float("nan"), 0.2]])
def test_cross_encoder_rejects_invalid_scores(scores):
    model = MagicMock()
    model.rerank_pairs.return_value = scores
    reranker = CrossEncoderReranker("local/model", model=model)

    with pytest.raises(RerankerError):
        reranker.rerank("question", [_chunk("one"), _chunk("two")], top_k=2)


def test_cross_encoder_rejects_negative_top_k():
    reranker = CrossEncoderReranker("local/model", model=MagicMock())

    with pytest.raises(ValueError, match="top_k cannot be negative"):
        reranker.rerank("question", [_chunk("one")], top_k=-1)
