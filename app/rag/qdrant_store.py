"""Qdrant store wrapper."""

from __future__ import annotations

import re
import threading
import uuid
from collections import OrderedDict
from datetime import UTC, date, datetime, time
from typing import Any

from app.core.config import settings
from app.core.logging import get_logger
from app.rag.schemas import Chunk, RetrievedChunk

log = get_logger(__name__)

_BM25_CACHE_MAXSIZE = 32
_bm25_cache: OrderedDict[
    tuple[str, str, str, str | None, str | None, date],
    tuple[list[dict[str, Any]], list[list[str]], Any | None],
] = OrderedDict()
_bm25_cache_lock = threading.Lock()


class QdrantStoreError(Exception):
    pass


def _current_content_filter(today: date):
    from qdrant_client.http import models as rest  # noqa: PLC0415

    start_of_today = datetime.combine(today, time.min, tzinfo=UTC)
    end_of_today = datetime.combine(today, time.max, tzinfo=UTC)
    policy_is_current = rest.Filter(
        must=[
            rest.FieldCondition(
                key="policy_status",
                match=rest.MatchValue(value="approved"),
            ),
            rest.FieldCondition(
                key="effective_from",
                range=rest.DatetimeRange(lte=end_of_today),
            ),
        ],
        should=[
            rest.FieldCondition(
                key="expires_on",
                range=rest.DatetimeRange(gte=start_of_today),
            ),
            rest.IsNullCondition(
                is_null=rest.PayloadField(key="expires_on"),
            ),
        ],
    )
    non_policy = rest.FieldCondition(
        key="policy_status",
        match=rest.MatchValue(value="not_applicable"),
    )
    return rest.Filter(should=[non_policy, policy_is_current])


def _qdrant_datetime(value: str | None) -> str | None:
    if value is None:
        return None
    parsed = date.fromisoformat(value)
    return (
        datetime.combine(parsed, time.min, tzinfo=UTC)
        .isoformat(
            timespec="seconds",
        )
        .replace("+00:00", "Z")
    )


def _retrieval_filter(
    audience: str | None,
    area: str | None,
    *,
    today: date | None = None,
):
    from qdrant_client.http import models as rest  # noqa: PLC0415

    must: list = []
    if audience and audience != "both":
        must.append(
            rest.FieldCondition(
                key="audience",
                match=rest.MatchAny(any=[audience, "both"]),
            )
        )
    if area and area != "both":
        must.append(
            rest.FieldCondition(
                key="area",
                match=rest.MatchAny(any=[area, "both"]),
            )
        )
    return rest.Filter(
        must=[
            *must,
            _current_content_filter(today or datetime.now(UTC).date()),
        ],
    )


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold())


class QdrantStore:
    """Synchronous Qdrant client wrapper for dense and keyword retrieval."""

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
        if name not in existing:
            client.create_collection(
                collection_name=name,
                vectors_config=rest.VectorParams(
                    size=dim,
                    distance=rest.Distance.COSINE,
                ),
            )

        payload_schema = client.get_collection(name).payload_schema or {}
        indexes = {
            "policy_status": rest.PayloadSchemaType.KEYWORD,
            "effective_from": rest.PayloadSchemaType.DATETIME,
            "expires_on": rest.PayloadSchemaType.DATETIME,
        }
        for field_name, schema in indexes.items():
            if field_name not in payload_schema:
                client.create_payload_index(
                    collection_name=name,
                    field_name=field_name,
                    field_schema=schema,
                    wait=True,
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
            raise QdrantStoreError(f"chunk/vector count mismatch: {len(chunks)} vs {len(vectors)}")

        name = self.versioned_name(version)
        client = self._get_client()
        total = 0

        for start in range(0, len(chunks), batch_size):
            batch_chunks = chunks[start : start + batch_size]
            batch_vectors = vectors[start : start + batch_size]
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
                        collection_name=target,
                        alias_name=alias,
                    ),
                ),
            ]
        else:
            operations = [
                rest.CreateAliasOperation(
                    create_alias=rest.CreateAlias(
                        collection_name=target,
                        alias_name=alias,
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
        index_version: str | None = None,
    ) -> list[RetrievedChunk]:
        """Vector search through the alias (always the current version).

        Filtering by audience and area happens inside Qdrant before
        ranking. ``audience="customer"`` matches chunks with
        audience=customer OR audience=both; a customer never receives
        vendor-only content.
        """
        from app.rag.schemas import RetrievedChunk  # noqa: PLC0415

        client = self._get_client()
        query_filter = _retrieval_filter(audience, area)

        response = client.query_points(
            collection_name=self.versioned_name(index_version)
            if index_version
            else self.alias_name(),
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
            out.append(
                RetrievedChunk(
                    chunk_id=str(payload.get("chunk_id", "")),
                    section_id=str(payload.get("section_id", "")),
                    section_path=str(payload.get("section_path", "")),
                    text=str(payload.get("text", "")),
                    score=float(hit.score),
                )
            )
        return out

    def keyword_search(
        self,
        query: str,
        *,
        top_k: int = 5,
        audience: str | None = None,
        area: str | None = None,
        index_version: str | None = None,
    ) -> list[RetrievedChunk]:
        """Rank current, eligible chunks with BM25 over their stored text."""
        try:
            from rank_bm25 import BM25Okapi  # noqa: PLC0415
        except ImportError as exc:
            raise QdrantStoreError(
                "rank-bm25 is not installed. Install with: pip install -r requirements-rag.txt"
            ) from exc

        query_tokens = _tokenize(query)
        if not query_tokens or top_k <= 0:
            return []

        client = self._get_client()
        today = datetime.now(UTC).date()
        if index_version:
            collection_name = self.versioned_name(index_version)
        else:
            aliases = client.get_aliases().aliases
            collection_name = next(
                (
                    alias.collection_name
                    for alias in aliases
                    if alias.alias_name == self.alias_name()
                ),
                None,
            )
            if collection_name is None:
                raise QdrantStoreError(f"Qdrant alias {self.alias_name()!r} is not configured")

        cache_key = (
            self.url,
            self.collection,
            collection_name,
            audience,
            area,
            today,
        )
        with _bm25_cache_lock:
            cached_index = _bm25_cache.get(cache_key)
            if cached_index is not None:
                _bm25_cache.move_to_end(cache_key)

        if cached_index is None:
            query_filter = _retrieval_filter(audience, area, today=today)
            points = []
            offset = None
            while True:
                page, offset = client.scroll(
                    collection_name=collection_name,
                    scroll_filter=query_filter,
                    limit=256,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                points.extend(page)
                if offset is None:
                    break

            documents = [point.payload or {} for point in points]
            tokenized_documents = [_tokenize(str(payload.get("text", ""))) for payload in documents]
            bm25 = BM25Okapi(tokenized_documents) if any(tokenized_documents) else None
            cached_index = (documents, tokenized_documents, bm25)
            with _bm25_cache_lock:
                _bm25_cache[cache_key] = cached_index
                _bm25_cache.move_to_end(cache_key)
                while len(_bm25_cache) > _BM25_CACHE_MAXSIZE:
                    _bm25_cache.popitem(last=False)

        documents, tokenized_documents, bm25 = cached_index
        if bm25 is None:
            return []

        scores = bm25.get_scores(query_tokens)
        query_terms = set(query_tokens)
        ranked_indexes = sorted(
            (
                index
                for index, tokens in enumerate(tokenized_documents)
                if query_terms.intersection(tokens)
            ),
            key=lambda index: (
                -float(scores[index]),
                str(documents[index].get("chunk_id", "")),
            ),
        )
        return [
            RetrievedChunk(
                chunk_id=str(documents[index].get("chunk_id", "")),
                section_id=str(documents[index].get("section_id", "")),
                section_path=str(documents[index].get("section_path", "")),
                text=str(documents[index].get("text", "")),
                score=float(scores[index]),
            )
            for index in ranked_indexes[:top_k]
        ]


# Helpers


def _point_id(chunk_id: str) -> str:
    """Deterministic UUID-shaped id from a chunk_id."""
    try:
        return str(uuid.UUID(chunk_id))
    except (ValueError, AttributeError):
        return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def _payload(chunk: Chunk) -> dict[str, Any]:
    """Payload stored alongside each vector.

    Flat by design. Both retrieval modes filter on ``audience``, ``area``,
    and current policy dates; ``text`` lets the retrieval layer return
    content without a second lookup.
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
        "policy_id": m.policy_id,
        "policy_status": m.policy_status,
        "effective_from": _qdrant_datetime(m.effective_from),
        "expires_on": _qdrant_datetime(m.expires_on),
        "supersedes": m.supersedes,
        "destination": m.destination,
        "activity": m.activity,
        "review_flag_ids": [f.id for f in m.review_flags if f.id],
        "text": chunk.text,
    }
