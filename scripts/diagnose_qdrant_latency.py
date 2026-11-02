
import os
import sys
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import httpx
import qdrant_client

# Add project root to Python import path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import settings
from app.rag.embeddings import EmbeddingModel
from app.rag.qdrant_store import QdrantStore, _retrieval_filter


QUERY = "what is the cancellation policy in this platform?"
AUDIENCE = "customer"
TOP_K = 20
INDEX_VERSION = "a441a35b9643"


def timed_httpx_test(
    *,
    trust_env: bool,
    url: str,
    headers: dict,
    body: dict,
) -> None:
    print(f"\nHTTPX trust_env={trust_env}")

    with httpx.Client(timeout=30, trust_env=trust_env) as session:
        for i in range(1, 4):
            started = perf_counter()

            response = session.post(
                url,
                headers=headers,
                json=body,
            )

            elapsed = (perf_counter() - started) * 1000
            response.raise_for_status()

            result = response.json()

            print(
                f"Request {i}: "
                f"client_total_ms={elapsed:.2f}, "
                f"qdrant_server_time={result.get('time')}, "
                f"status={response.status_code}"
            )



def inspect_object(obj, name: str, depth: int = 0, seen=None) -> None:
    """Inspect SDK internals without printing configuration values or secrets."""
    if seen is None:
        seen = set()

    if obj is None or depth > 5 or id(obj) in seen:
        return

    seen.add(id(obj))
    indent = "  " * depth

    print(
        f"{indent}{name}: "
        f"{type(obj).__module__}.{type(obj).__name__}"
    )

    # Inspect object attributes without exposing their values.
    attributes = getattr(obj, "__dict__", {})

    for attr_name, value in attributes.items():
        if value is None or isinstance(
            value,
            (str, int, float, bool, bytes),
        ):
            continue

        print(
            f"{indent}  {attr_name}: "
            f"{type(value).__module__}.{type(value).__name__}"
        )

        if depth < 5:
            inspect_object(
                value,
                attr_name,
                depth + 1,
                seen,
            )

    # Inspect selected public/private transport properties.
    for attr_name in (
        "http",
        "rest_client",
        "client",
        "session",
        "transport",
    ):
        try:
            value = getattr(obj, attr_name, None)
        except Exception:
            continue

        if value is None or isinstance(
            value,
            (str, int, float, bool, bytes),
        ):
            continue

        print(
            f"{indent}  property {attr_name}: "
            f"{type(value).__module__}.{type(value).__name__}"
        )

        if depth < 5:
            inspect_object(
                value,
                attr_name,
                depth + 1,
                seen,
            )


def test_sdk_operations(
    client,
    collection: str,
    vector: list,
    query_filter,
) -> None:
    print("\n--- QDRANT PYTHON CLIENT TESTS ---")

    # Test 1: Basic API request.
    for i in range(1, 4):
        started = perf_counter()

        client.get_collections()

        elapsed = (perf_counter() - started) * 1000
        print(f"get_collections {i}: {elapsed:.2f} ms")

    # Test 2: Scroll request.
    for i in range(1, 4):
        started = perf_counter()

        client.scroll(
            collection_name=collection,
            limit=1,
            with_payload=False,
            with_vectors=False,
        )

        elapsed = (perf_counter() - started) * 1000
        print(f"scroll {i}: {elapsed:.2f} ms")

    # Test 3: Vector query.
    for i in range(1, 4):
        started = perf_counter()

        response = client.query_points(
            collection_name=collection,
            query=vector,
            limit=TOP_K,
            query_filter=query_filter,
            with_payload=True,
        )

        elapsed = (perf_counter() - started) * 1000

        print(
            f"query_points {i}: "
            f"client_total_ms={elapsed:.2f}, "
            f"result_count={len(response.points)}"
        )


def test_sdk_underlying_httpx(
    client,
    url: str,
    body: dict,
) -> None:
    print("\n--- SDK UNDERLYING HTTPX CONNECTION TEST ---")

    try:
        sdk_httpx = client.http.client._client
        transport = sdk_httpx._transport
        pool = getattr(transport, "_pool", None)

        print("HTTPX client:", type(sdk_httpx).__name__)
        print("HTTPX base URL:", str(sdk_httpx.base_url))
        print("Transport:", type(transport).__name__)
        print("Pool:", type(pool).__name__ if pool else None)

        if pool is not None:
            print(
                "Max connections:",
                getattr(pool, "_max_connections", "unknown"),
            )
            print(
                "Max keepalive connections:",
                getattr(pool, "_max_keepalive_connections", "unknown"),
            )
            print(
                "Keepalive expiry:",
                getattr(pool, "_keepalive_expiry", "unknown"),
            )

        print(
            "SDK default Connection header:",
            sdk_httpx.headers.get("connection", "not set"),
        )

        for i in range(1, 4):
            started = perf_counter()

            response = sdk_httpx.post(
                url,
                json=body,
            )

            elapsed = (perf_counter() - started) * 1000
            response.raise_for_status()

            print(f"\nRequest {i}")
            print(f"Client total: {elapsed:.2f} ms")
            print(f"HTTP version: {response.http_version}")
            print(
                "Response Connection:",
                response.headers.get("connection", "not set"),
            )
            print(
                "Response Keep-Alive:",
                response.headers.get("keep-alive", "not set"),
            )
            print(
                "Response Server:",
                response.headers.get("server", "not set"),
            )
            print(
                "Pool connections after response:",
                len(getattr(pool, "_connections", [])),
            )

            response.close()

    except Exception as exc:
        print(
            "Connection diagnostic failed:",
            type(exc).__name__,
            str(exc),
        )



def main() -> None:
    print("Generating query embedding...")

    vector = EmbeddingModel().embed_queries([QUERY])[0].tolist()

    store = QdrantStore()
    collection = store.versioned_name(INDEX_VERSION)

    query_filter = _retrieval_filter(AUDIENCE, None)

    body = {
        "query": vector,
        "limit": TOP_K,
        "filter": query_filter.model_dump(
            mode="json",
            exclude_none=True,
        ),
        "with_payload": True,
        "with_vector": False,
    }

    url = (
        f"{settings.qdrant_url.rstrip('/')}"
        f"/collections/{collection}/points/query"
    )

    headers = {
        "Content-Type": "application/json",
    }

    if settings.qdrant_api_key:
        headers["api-key"] = settings.qdrant_api_key

    proxy_configured = any(
        os.environ.get(name)
        for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY")
    )

    print(f"Collection: {collection}")
    print(f"Vector dimensions: {len(vector)}")
    print(f"HTTPX proxy environment configured: {proxy_configured}")

    # Test 1: HTTPX with environment settings.
    timed_httpx_test(
        trust_env=True,
        url=url,
        headers=headers,
        body=body,
    )

    # Test 2: HTTPX without environment settings.
    timed_httpx_test(
        trust_env=False,
        url=url,
        headers=headers,
        body=body,
    )

    # Test 3: Inspect installed SDK and transport.
    print("\n--- SDK TRANSPORT DIAGNOSTICS ---")
    print("qdrant-client version:", version("qdrant-client"))
    print("httpx version:", httpx.__version__)
    print("qdrant_client module:", qdrant_client.__file__)

    # Initialize the actual client used by QdrantStore.
    client = store._get_client()
    test_sdk_underlying_httpx(
    client=client,
    url=url,
    body=body,
)
    print("\n--- QDRANT REMOTE INTERNALS ---")
    inspect_object(client._client, "QdrantRemote")

    print("\n--- PUBLIC REST CLIENT INTERNALS ---")
    inspect_object(client.http, "client.http")

    inspect_object(client, "qdrant_client")

    # Test 4: Compare SDK operations.
    test_sdk_operations(
        client=client,
        collection=collection,
        vector=vector,
        query_filter=query_filter,
    )

if __name__ == "__main__":
    main()
