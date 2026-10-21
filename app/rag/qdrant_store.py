"""Qdrant store wrapper."""
from __future__ import annotations

import uuid
from typing import Any

from app.core.config import settings
from app.core.logging import get_logger
from app.rag.schemas import Chunk, RetrievedChunk

log = get_logger(__name__)


class QdrantStoreError(Exception):
    pass


class QdrantStore:
    """Synchronous qdrant-client wrapper.

    The FastAPI route that will use retrieval (Batch 2C) runs in a thread
    pool, so sync is fine here.
    """

    def __init__(
        self,
        *,
        url: str | None = None,
        api_key: str | None = None,
        collection: str | None = None,
        timeout: int | None = None,
    ) -> None:
        self.url = url or settings.qdrant_url
        self.api_key = api_key or settings.qdrant_api_key or None
        self.collection = collection or settings.qdrant_collection
        self.timeout = timeout or settings.qdrant_timeout
        self._client = None

    # -- Client

    def _get_client(self):
        if self._client is None:
            try:
                from qdrant_client import QdrantClient  # noqa: PLC0415
            except ImportError as exc:
                raise QdrantStoreError(
                    "qdrant-client is not installed. "
                    "Install with: pip install -r requirements-rag.txt"
                ) from exc
            self._client = QdrantClient(
                url=self.url,
                api_key=self.api_key,
                timeout=self.timeout,
            )
        return self._client

    def ping(self) -> bool:
        try:
            self._get_client().get_collections()
            return True
        except Exception as exc:  # noqa: BLE001
            log.warning("qdrant ping failed: %s", exc)
            return False

    # -- Names

    def versioned_name(self, version: str) -> str:
        return f"{self.collection}_v{version}"

    def alias_name(self) -> str:
        return self.collection

    def current_versioned_collection(self) -> str | None:
        """Return the physical collection the alias points at, or None."""
        try:
            client = self._get_client()
            aliases = client.get_aliases().aliases
            for a in aliases:
                if a.alias_name == self.collection:
                    return a.collection_name
            return None
        except Exception as exc:  # noqa: BLE001
            log.warning("could not resolve alias: %s", exc)
            return None

    # -- Collection management

    def ensure_collection(self, version: str, *, dim: int) -> None:
        from qdrant_client.http import models as rest  # noqa: PLC0415

        name = self.versioned_name(version)
        client = self._get_client()
        existing = {c.name for c in client.get_collections().collections}
        if name in existing:
            return
        client.create_collection(
            collection_name=name,
            vectors_config=rest.VectorParams(
                size=dim,
                distance=rest.Distance.COSINE,
            ),
        )

    def upsert_chunks(
        self,
        version: str,
        chunks: list[Chunk],
        vectors,
        *,
        batch_size: int = 128,
    ) -> int:
        """Upsert dense vectors for a version. Idempotent by point id."""
        from qdrant_client.http import models as rest  # noqa: PLC0415

        if len(chunks) != len(vectors):
            raise QdrantStoreError(
                f"chunk/vector count mismatch: {len(chunks)} vs {len(vectors)}"
            )

        name = self.versioned_name(version)
        client = self._get_client()
        total = 0

        for start in range(0, len(chunks), batch_size):
            batch_chunks = chunks[start:start + batch_size]
            batch_vectors = vectors[start:start + batch_size]
            points = [
                rest.PointStruct(
                    id=_point_id(c.metadata.chunk_id),
                    vector=vec.tolist(),
                    payload=_payload(c),
                )
                for c, vec in zip(batch_chunks, batch_vectors, strict=True)
            ]
            client.upsert(collection_name=name, points=points, wait=True)
            total += len(points)

        return total

    def count(self, version: str) -> int:
        client = self._get_client()
        name = self.versioned_name(version)
        result = client.count(collection_name=name, exact=True)
        return int(result.count)

    # -- Alias swap

    def point_alias_at(self, version: str) -> None:
        """Atomically point the alias at the given versioned collection.

        Delete-old + create-new in a single API call. Qdrant executes the
        list atomically, so readers never see a partial state.
        """
        from qdrant_client.http import models as rest  # noqa: PLC0415

        client = self._get_client()
        alias = self.alias_name()
        target = self.versioned_name(version)

        existing_aliases = {a.alias_name for a in client.get_aliases().aliases}

        if alias in existing_aliases:
            current = self.current_versioned_collection()
            if current == target:
                return  # already correct, no-op
            operations = [
                rest.DeleteAliasOperation(
                    delete_alias=rest.DeleteAlias(alias_name=alias),
                ),
                rest.CreateAliasOperation(
                    create_alias=rest.CreateAlias(
                        collection_name=target, alias_name=alias,
                    ),
                ),
            ]
        else:
            operations = [
                rest.CreateAliasOperation(
                    create_alias=rest.CreateAlias(
                        collection_name=target, alias_name=alias,
                    ),
                ),
            ]

        client.update_collection_aliases(change_aliases_operations=operations)

    def delete_collection(self, version: str) -> None:
        client = self._get_client()
        name = self.versioned_name(version)
        try:
            client.delete_collection(collection_name=name)
        except Exception as exc:  # noqa: BLE001
            log.warning("delete_collection %s failed: %s", name, exc)

    def search(
        self,
        query_vector: list[float],
        *,
        top_k: int = 5,
        audience: str | None = None,
        area: str | None = None,
        min_score: float | None = None,
    ) -> list[RetrievedChunk]:
        """Vector search through the alias (always the current version).

        Filtering by audience and area happens inside Qdrant before
        ranking. ``audience="customer"`` matches chunks with
        audience=customer OR audience=both; a customer never receives
        vendor-only content.
        """
        from qdrant_client.http import models as rest  # noqa: PLC0415

        from app.rag.schemas import RetrievedChunk  # noqa: PLC0415

        client = self._get_client()

        must: list = []
        if audience and audience != "both":
            must.append(rest.FieldCondition(
                key="audience",
                match=rest.MatchAny(any=[audience, "both"]),
            ))
        if area and area != "both":
            must.append(rest.FieldCondition(
                key="area",
                match=rest.MatchAny(any=[area, "both"]),
            ))
        query_filter = rest.Filter(must=must) if must else None

        response = client.query_points(
            collection_name=self.alias_name(),
            query=query_vector,
            limit=top_k,
            query_filter=query_filter,
            score_threshold=min_score,
            with_payload=True,
        )
        hits = response.points

        out: list[RetrievedChunk] = []
        for hit in hits:
            payload = hit.payload or {}
            out.append(RetrievedChunk(
                chunk_id=str(payload.get("chunk_id", "")),
                section_id=str(payload.get("section_id", "")),
                section_path=str(payload.get("section_path", "")),
                text=str(payload.get("text", "")),
                score=float(hit.score),
            ))
        return out


# Helpers

def _point_id(chunk_id: str) -> str:
    """Deterministic UUID-shaped id from a chunk_id."""
    try:
        return str(uuid.UUID(chunk_id))
    except (ValueError, AttributeError):
        return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def _payload(chunk: Chunk) -> dict[str, Any]:
    """Payload stored alongside each vector.

    Flat by design. Retrieval in Batch 2C filters on ``audience`` and
    optionally ``area``; ``text`` is carried in the payload so the
    retrieval layer can feed it to the LLM without a second lookup.
    """
    m = chunk.metadata
    return {
        "chunk_id": m.chunk_id,
        "document_id": m.document_id,
        "category": m.category,
        "section_id": m.section_id,
        "section_title": m.section_title,
        "section_path": m.section_path,
        "area": m.area.value,
        "audience": m.audience.value,
        "source": m.source,
        "source_version": m.source_version,
        "last_updated": m.last_updated,
        "destination": m.destination,
        "activity": m.activity,
        "review_flag_ids": [f.id for f in m.review_flags if f.id],
        "text": chunk.text,
    }
