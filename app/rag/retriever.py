"""Dense retriever over Qdrant.

Wraps EmbeddingModel + QdrantStore into a single entry point for the
support service. Uses qdrant-client directly because alias-based reads
(always the current published version) and MatchAny audience filters
work more cleanly here than through the LangChain wrapper. LangChain
returns in Batch 2E for hybrid fusion.
"""
from __future__ import annotations

from app.core.config import settings
from app.core.logging import get_logger
from app.rag.embeddings import EmbeddingError, EmbeddingModel
from app.rag.qdrant_store import QdrantStore, QdrantStoreError
from app.rag.schemas import RetrievedChunk

log = get_logger(__name__)


class RetrieverError(Exception):
    pass


class RagRetriever:
    def __init__(
        self,
        *,
        embedder: EmbeddingModel | None = None,
        store: QdrantStore | None = None,
    ) -> None:
        self.embedder = embedder or EmbeddingModel()
        self.store = store or QdrantStore()

    def search(
        self,
        query: str,
        *,
        audience: str,
        area: str | None = None,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> list[RetrievedChunk]:
        if not query.strip():
            return []

        try:
            vectors = self.embedder.embed_queries([query])
        except EmbeddingError as exc:
            raise RetrieverError(f"embedding failed: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            raise RetrieverError(f"unexpected embedding error: {exc}") from exc

        if len(vectors) == 0:
            return []

        try:
            return self.store.search(
                vectors[0].tolist(),
                top_k=top_k or settings.rag_top_k,
                audience=audience,
                area=area or None,
                min_score=(
                    min_score if min_score is not None
                    else settings.rag_min_score
                ),
            )
        except QdrantStoreError as exc:
            raise RetrieverError(f"qdrant search failed: {exc}") from exc
        except Exception as exc:    #noqa: BLE001
            raise RetrieverError(f"unexpected retriever error: {exc}") from exc

