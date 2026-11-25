"""Hybrid BM25 + dense retriever over Qdrant.

Wraps EmbeddingModel + QdrantStore into a single entry point for the
support service. Qdrant applies audience, area, and policy-date filters to
both result sets before Reciprocal Rank Fusion combines their rankings.

Retrieval-stage timings are logged without recording query or document text.
"""

from __future__ import annotations

from collections.abc import Sequence
from time import perf_counter

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
        request_started = perf_counter()
        timings: dict[str, float] = {}

        if rrf_rank_constant < 1:
            raise ValueError("rrf_rank_constant must be at least 1")
        if candidate_multiplier < 1:
            raise ValueError("candidate_multiplier must be at least 1")
        if not query.strip():
            return []

        # 1. Query embedding
        stage_started = perf_counter()
        try:
            vectors = self.embedder.embed_queries([query])
        except EmbeddingError as exc:
            timings["embedding_ms"] = round((perf_counter() - stage_started) * 1000, 2)
            self._log_failure("embedding", request_started, timings, exc)
            raise RetrieverError(f"embedding failed: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            timings["embedding_ms"] = round((perf_counter() - stage_started) * 1000, 2)
            self._log_failure("embedding", request_started, timings, exc)
            raise RetrieverError(f"unexpected embedding error: {exc}") from exc
        timings["embedding_ms"] = round((perf_counter() - stage_started) * 1000, 2)

        limit = max(top_k or settings.rag_top_k, 1)
        candidate_limit = limit * candidate_multiplier
        score_threshold = min_score if min_score is not None else settings.rag_min_score

        # 2. Dense Qdrant search
        stage_started = perf_counter()
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
            timings["dense_search_ms"] = round((perf_counter() - stage_started) * 1000, 2)
            self._log_failure("dense_search", request_started, timings, exc)
            raise RetrieverError(f"qdrant search failed: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            timings["dense_search_ms"] = round((perf_counter() - stage_started) * 1000, 2)
            self._log_failure("dense_search", request_started, timings, exc)
            raise RetrieverError(f"unexpected retriever error: {exc}") from exc
        timings["dense_search_ms"] = round((perf_counter() - stage_started) * 1000, 2)

        # 3. BM25 keyword search
        stage_started = perf_counter()
        try:
            keyword_results = self.store.keyword_search(
                query,
                top_k=candidate_limit,
                audience=audience,
                area=area or None,
                index_version=index_version,
            )
        except QdrantStoreError as exc:
            timings["bm25_search_ms"] = round((perf_counter() - stage_started) * 1000, 2)
            self._log_failure("bm25_search", request_started, timings, exc)
            raise RetrieverError(f"BM25 search failed: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            timings["bm25_search_ms"] = round((perf_counter() - stage_started) * 1000, 2)
            self._log_failure("bm25_search", request_started, timings, exc)
            raise RetrieverError(f"unexpected retriever error: {exc}") from exc
        timings["bm25_search_ms"] = round((perf_counter() - stage_started) * 1000, 2)

        # 4. Reciprocal Rank Fusion
        stage_started = perf_counter()
        fused_results = reciprocal_rank_fusion(
            (dense_results, keyword_results),
            top_k=candidate_limit,
            rank_constant=rrf_rank_constant,
        )
        timings["rrf_ms"] = round((perf_counter() - stage_started) * 1000, 2)

        # 5. Optional reranking
        if self.reranker is None:
            results = fused_results[:limit]
            timings["rerank_ms"] = 0.0
        else:
            stage_started = perf_counter()
            try:
                results = self.reranker.rerank(
                    query,
                    fused_results,
                    top_k=limit,
                )
            except RerankerError as exc:
                timings["rerank_ms"] = round((perf_counter() - stage_started) * 1000, 2)
                self._log_failure("reranking", request_started, timings, exc)
                raise RetrieverError(f"reranking failed: {exc}") from exc
            except Exception as exc:  # noqa: BLE001
                timings["rerank_ms"] = round((perf_counter() - stage_started) * 1000, 2)
                self._log_failure("reranking", request_started, timings, exc)
                raise RetrieverError(f"unexpected reranker error: {exc}") from exc
            timings["rerank_ms"] = round((perf_counter() - stage_started) * 1000, 2)

        total_ms = round((perf_counter() - request_started) * 1000, 2)

        log.info(
            "rag retrieval completed",
            extra={
                "event": "rag_retrieval_timing",
                "audience": audience,
                "area": area,
                "index_version": index_version,
                "reranker_enabled": self.reranker is not None,
                "dense_result_count": len(dense_results),
                "bm25_result_count": len(keyword_results),
                "fused_result_count": len(fused_results),
                "returned_result_count": len(results),
                **timings,
                "total_ms": total_ms,
            },
        )

        return results

    @staticmethod
    def _log_failure(
        stage: str,
        request_started: float,
        timings: dict[str, float],
        exc: Exception,
    ) -> None:
        """Log a failed retrieval stage without exposing query/document text."""
        log.warning(
            "rag retrieval stage failed",
            extra={
                "event": "rag_retrieval_stage_failed",
                "stage": stage,
                "error_type": type(exc).__name__,
                "total_elapsed_ms": round((perf_counter() - request_started) * 1000, 2),
                **timings,
            },
        )
