"""Dense embeddings via fastembed."""
from __future__ import annotations

import threading

import numpy as np

from app.core.config import settings
from app.core.logging import get_logger

log = get_logger(__name__)


class EmbeddingError(Exception):
    pass


def _needs_e5_prefix(model_name: str) -> bool:
    return "e5" in model_name.lower()


class EmbeddingModel:
    """Thin wrapper over fastembed's TextEmbedding.

    Thread-safe lazy load. The underlying model is shared via a
    class-level cache keyed by model name.
    """

    _model_cache: dict[str, object] = {}
    _cache_lock = threading.Lock()

    def __init__(
        self,
        model_name: str | None = None,
        *,
        dim: int | None = None,
    ) -> None:
        self.model_name = model_name or settings.rag_embedding_model
        self.dim = dim or settings.rag_embedding_dim

    # -- Model load

    def _get_model(self):
        with self._cache_lock:
            model = self._model_cache.get(self.model_name)
            if model is None:
                try:
                    from fastembed import TextEmbedding  # noqa: PLC0415
                except ImportError as exc:
                    raise EmbeddingError(
                        "fastembed is not installed. "
                        "Install with: pip install -r requirements-rag.txt"
                    ) from exc
                model = TextEmbedding(model_name=self.model_name)
                self._model_cache[self.model_name] = model
            return model

    # -- Public API

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        prefix = "passage: " if _needs_e5_prefix(self.model_name) else ""
        return self._embed([prefix + t for t in texts])

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        prefix = "query: " if _needs_e5_prefix(self.model_name) else ""
        return self._embed([prefix + t for t in texts])

    # -- Internal

    def _embed(self, prefixed_texts: list[str]) -> np.ndarray:
        if not prefixed_texts:
            return np.zeros((0, self.dim), dtype="float32")

        model = self._get_model()
        vectors = np.asarray(
            list(model.embed(
                prefixed_texts,
                batch_size=settings.rag_embedding_batch_size,
            )),
            dtype="float32",
        )

        if vectors.shape[1] != self.dim:
            raise EmbeddingError(
                f"model {self.model_name} produced dim {vectors.shape[1]}, "
                f"expected {self.dim}. Update RAG_EMBEDDING_DIM."
            )

        # L2-normalise so cosine == dot product (Qdrant Cosine distance).
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    def verify_ready(self) -> None:
        """Force a model load. Raises EmbeddingError if unavailable."""
        self.embed_queries(["ping"])
