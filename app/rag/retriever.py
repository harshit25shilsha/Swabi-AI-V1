"""Hybrid BM25 + dense retriever over Qdrant.

Wraps EmbeddingModel + QdrantStore into a single entry point for the
support service. Qdrant applies audience, area, and policy-date filters to
both result sets before Reciprocal Rank Fusion combines their rankings.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.core.config import settings
from app.core.logging import get_logger
from app.rag.embeddings import EmbeddingError, EmbeddingModel
from app.rag.qdrant_store import QdrantStore, QdrantStoreError
from app.rag.reranker import (
    DEFAULT_CROSS_ENCODER_MODEL,
    CrossEncoderReranker,
    Reranker,
    RerankerError,
)
from app.rag.schemas import RetrievedChunk

log = get_logger(__name__)

_RRF_RANK_CONSTANT = 60
_CANDIDATE_MULTIPLIER = 4


class RetrieverError(Exception):
    pass


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[RetrievedChunk]],
    *,
    top_k: int,
    rank_constant: int = _RRF_RANK_CONSTANT,
) -> list[RetrievedChunk]:
    """Combine ranked result lists, counting each chunk once."""
    if rank_constant < 1:
        raise ValueError("rank_constant must be at least 1")
    if top_k < 0:
        raise ValueError("top_k cannot be negative")

    fused: dict[str, tuple[RetrievedChunk, float]] = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking, start=1):
            previous = fused.get(chunk.chunk_id)
            score = 1 / (rank_constant + rank)
            if previous is None:
                fused[chunk.chunk_id] = (chunk, score)
            else:
                fused[chunk.chunk_id] = (previous[0], previous[1] + score)

    ordered = sorted(
        fused.values(),
        key=lambda item: item[1],
        reverse=True,
    )
    return [chunk.model_copy(update={"score": score}) for chunk, score in ordered[:top_k]]


class RagRetriever:
    def __init__(
        self,
        *,
        embedder: EmbeddingModel | None = None,
        store: QdrantStore | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.embedder = embedder or EmbeddingModel()
        self.store = store or QdrantStore()
        self.reranker = reranker
        if self.reranker is None and settings.rag_reranker_enabled:
            self.reranker = CrossEncoderReranker(
                settings.rag_reranker_model or DEFAULT_CROSS_ENCODER_MODEL
            )

    def search(
        self,
        query: str,
        *,
        audience: str,
        area: str | None = None,
        top_k: int | None = None,
        min_score: float | None = None,
        index_version: str | None = None,
        rrf_rank_constant: int = _RRF_RANK_CONSTANT,
        candidate_multiplier: int = _CANDIDATE_MULTIPLIER,
    ) -> list[RetrievedChunk]:
        if rrf_rank_constant < 1:
            raise ValueError("rrf_rank_constant must be at least 1")
        if candidate_multiplier < 1:
            raise ValueError("candidate_multiplier must be at least 1")
        if not query.strip():
            return []

        try:
            vectors = self.embedder.embed_queries([query])
        except EmbeddingError as exc:
            raise RetrieverError(f"embedding failed: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            raise RetrieverError(f"unexpected embedding error: {exc}") from exc

        limit = max(top_k or settings.rag_top_k, 1)
        candidate_limit = limit * candidate_multiplier
        score_threshold = min_score if min_score is not None else settings.rag_min_score

        try:
            dense_results = (
                self.store.search(
                    vectors[0].tolist(),
                    top_k=candidate_limit,
                    audience=audience,
                    area=area or None,
                    min_score=score_threshold,
                    index_version=index_version,
                )
                if len(vectors)
                else []
            )
        except QdrantStoreError as exc:
            raise RetrieverError(f"qdrant search failed: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            raise RetrieverError(f"unexpected retriever error: {exc}") from exc

        try:
            keyword_results = self.store.keyword_search(
                query,
                top_k=candidate_limit,
                audience=audience,
                area=area or None,
                index_version=index_version,
            )
        except QdrantStoreError as exc:
            raise RetrieverError(f"BM25 search failed: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            raise RetrieverError(f"unexpected retriever error: {exc}") from exc

        fused_results = reciprocal_rank_fusion(
            (dense_results, keyword_results),
            top_k=candidate_limit,
            rank_constant=rrf_rank_constant,
        )
        if self.reranker is None:
            return fused_results[:limit]
        try:
            return self.reranker.rerank(query, fused_results, top_k=limit)
        except RerankerError as exc:
            raise RetrieverError(f"reranking failed: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            raise RetrieverError(f"unexpected reranker error: {exc}") from exc
