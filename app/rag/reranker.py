"""Cross-encoder reranking for hybrid retrieval candidates."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Protocol

from app.rag.schemas import RetrievedChunk

DEFAULT_CROSS_ENCODER_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"


class RerankerError(Exception):
    """Raised when a reranker cannot produce a valid ranking."""


class Reranker(Protocol):
    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievedChunk],
        *,
        top_k: int,
    ) -> list[RetrievedChunk]: ...


class _CrossEncoderModel(Protocol):
    def rerank_pairs(
        self,
        pairs: list[tuple[str, str]],
    ) -> Iterable[float]: ...


class CrossEncoderReranker:
    """Local FastEmbed cross-encoder with lazy model loading."""

    def __init__(
        self,
        model_name: str = DEFAULT_CROSS_ENCODER_MODEL,
        *,
        model: _CrossEncoderModel | None = None,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name cannot be empty")
        self.model_name = model_name
        self._model = model

    def _get_model(self) -> _CrossEncoderModel:
        if self._model is None:
            try:
                from fastembed.rerank.cross_encoder import TextCrossEncoder  # noqa: PLC0415
            except ImportError as exc:
                raise RerankerError(
                    "FastEmbed cross-encoder support is unavailable. "
                    "Install the RAG requirements with fastembed>=0.9.0."
                ) from exc
            try:
                self._model = TextCrossEncoder(self.model_name)
            except Exception as exc:  # noqa: BLE001
                raise RerankerError(
                    f"could not load cross-encoder model {self.model_name!r}: {exc}"
                ) from exc
        return self._model

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievedChunk],
        *,
        top_k: int,
    ) -> list[RetrievedChunk]:
        if top_k < 0:
            raise ValueError("top_k cannot be negative")
        if top_k == 0 or not candidates:
            return []

        pairs = [(query, chunk.text) for chunk in candidates]
        try:
            scores = [float(score) for score in self._get_model().rerank_pairs(pairs)]
        except RerankerError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise RerankerError(f"cross-encoder prediction failed: {exc}") from exc

        if len(scores) != len(candidates):
            raise RerankerError(
                f"cross-encoder returned {len(scores)} scores for {len(candidates)} candidates"
            )
        if not all(math.isfinite(score) for score in scores):
            raise RerankerError("cross-encoder returned a non-finite score")

        ranked = sorted(
            enumerate(zip(candidates, scores, strict=True)),
            key=lambda item: (-item[1][1], item[0]),
        )
        return [chunk.model_copy(update={"score": score}) for _, (chunk, score) in ranked[:top_k]]
