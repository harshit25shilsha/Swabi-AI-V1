
"""Diagnose Qdrant dense-search latency without changing production code."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from app.core.config import settings  # noqa: E402
from app.rag.embeddings import EmbeddingModel  # noqa: E402
from app.rag.qdrant_store import QdrantStore, _retrieval_filter  # noqa: E402

QUERY = "How can I book a package on this platform?"
AUDIENCE = "customer"
AREA = None
VERSION = "a441a35b9643"
TOP_K = 20


def measure(label, fn):
    started = time.perf_counter()
    result = fn()
    elapsed = (time.perf_counter() - started) * 1000
    print(f"{label}: client_elapsed_ms={elapsed:.2f}")
    return result


def main():
    collection = f"{settings.qdrant_collection}_v{VERSION}"

    print(f"Qdrant URL: {settings.qdrant_url}")
    print(f"Collection: {collection}")
    print(f"Query: {QUERY}")

    # Generate one query vector and reuse it for every test.
    embedder = EmbeddingModel()
    vector = measure(
        "Embedding",
        lambda: embedder.embed_queries([QUERY])[0].tolist(),
    )

    store = QdrantStore()
    client = store._get_client()

    # Inspect collection configuration.
    info = measure(
        "Get collection info",
        lambda: client.get_collection(collection),
    )

    print("\n--- Collection information ---")
    print("Status:", info.status)
    print("Points count:", info.points_count)
    print("Indexed vectors:", info.indexed_vectors_count)
    print("Payload indexes:", info.payload_schema)

    query_filter = _retrieval_filter(
        AUDIENCE,
        AREA,
    )

    filter_json = query_filter.model_dump(
        mode="json",
        exclude_none=True,
    )

    # Test the same query through Qdrant Python SDK.
    print("\n--- Qdrant SDK: filtered query ---")

    for i in range(1, 4):
        response = measure(
            f"SDK filtered request {i}",
            lambda: client.query_points(
                collection_name=collection,
                query=vector,
                limit=TOP_K,
                query_filter=query_filter,
                with_payload=True,
            ),
        )
        print("Results:", len(response.points))

    # Test direct HTTP against the same Qdrant REST endpoint.
    print("\n--- Direct HTTP: filtered query ---")

    headers = {}
    if settings.qdrant_api_key:
        headers["api-key"] = settings.qdrant_api_key

    endpoint = (
        f"{settings.qdrant_url.rstrip('/')}"
        f"/collections/{collection}/points/query"
    )

    payload = {
        "query": vector,
        "limit": TOP_K,
        "filter": filter_json,
        "with_payload": True,
    }

    with httpx.Client(
        timeout=settings.qdrant_timeout,
        trust_env=False,
    ) as http:
        for i in range(1, 4):
            started = time.perf_counter()
            response = http.post(
                endpoint,
                json=payload,
                headers=headers,
            )
            elapsed = (time.perf_counter() - started) * 1000

            response.raise_for_status()
            body = response.json()

            print(
                f"HTTP filtered request {i}: "
                f"client_elapsed_ms={elapsed:.2f}, "
                f"server_time_ms={float(body.get('time', 0)) * 1000:.2f}"
            )

        # Compare with the same query without payload filters.
        print("\n--- Direct HTTP: unfiltered query ---")

        unfiltered_payload = {
            "query": vector,
            "limit": TOP_K,
            "with_payload": True,
        }

        for i in range(1, 4):
            started = time.perf_counter()
            response = http.post(
                endpoint,
                json=unfiltered_payload,
                headers=headers,
            )
            elapsed = (time.perf_counter() - started) * 1000

            response.raise_for_status()
            body = response.json()

            print(
                f"HTTP unfiltered request {i}: "
                f"client_elapsed_ms={elapsed:.2f}, "
                f"server_time_ms={float(body.get('time', 0)) * 1000:.2f}"
            )


if __name__ == "__main__":
    main()
