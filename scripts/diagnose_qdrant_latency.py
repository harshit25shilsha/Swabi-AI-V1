
import os
import math
import socket
import ssl
import statistics
import sys
from qdrant_client import QdrantClient
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from urllib.parse import urlparse
import httpx
import qdrant_client

# Add project root to Python import path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import settings  # noqa: E402
from app.rag.embeddings import EmbeddingModel  # noqa: E402
from app.rag.qdrant_store import QdrantStore, _retrieval_filter  # noqa: E402

QUERY = "what is the cancellation policy in this platform?"
AUDIENCE = "customer"
TOP_K = 20


def test_connection_phases(url: str) -> None:
    """Measure DNS, TCP, and optional TLS setup independently."""
    print("\n--- DNS / TCP / TLS CONNECTION PHASES ---")

    parsed = urlparse(url)
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == "https" else 80)

    if not host:
        print("Could not determine Qdrant hostname.")
        return

    connection = None

    try:
        # DNS lookup
        started = perf_counter()
        addresses = socket.getaddrinfo(
            host,
            port,
            type=socket.SOCK_STREAM,
        )
        dns_ms = (perf_counter() - started) * 1000

        if not addresses:
            print("DNS returned no addresses.")
            return


        for index, (family, socktype, proto, _, sockaddr) in enumerate(
            addresses,
            start=1,
        ):

            connection = socket.socket(family, socktype, proto)
            connection.settimeout(5)

            try:
                started = perf_counter()
                connection.connect(sockaddr)
                tcp_ms = (perf_counter() - started) * 1000

                print(
                    f"Address {index}: {sockaddr} | "
                    f"TCP connected | {tcp_ms:.2f} ms"
                )

            except OSError as exc:
                print(
                    f"Address {index}: {sockaddr} | "
                    f"TCP failed | {type(exc).__name__}: {exc}"
                )

            finally:
                connection.close()


        print(f"Resolved host: {host}")
        print(f"Resolved address count: {len(addresses)}")
        print(f"DNS lookup: {dns_ms:.2f} ms")

        # TCP connection
        connection = socket.socket(family, socktype, proto)
        connection.settimeout(10)

        started = perf_counter()
        connection.connect(sockaddr)
        tcp_ms = (perf_counter() - started) * 1000

        print(f"TCP connect: {tcp_ms:.2f} ms")

        # TLS handshake, only for HTTPS
        if parsed.scheme == "https":
            context = ssl.create_default_context()

            started = perf_counter()
            connection = context.wrap_socket(
                connection,
                server_hostname=host,
            )
            tls_ms = (perf_counter() - started) * 1000

            print(f"TLS handshake: {tls_ms:.2f} ms")
            print(f"TLS version: {connection.version()}")

    except Exception as exc:
        print(
            "Connection phase diagnostic failed:",
            type(exc).__name__,
            str(exc),
        )

    finally:
        if connection is not None:
            connection.close()


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

def test_httpx_response_phases(
    *,
    url: str,
    headers: dict,
    body: dict,
) -> None:
    """Measure time to response headers separately from body download."""
    print("\n--- HTTPX RESPONSE PHASE TIMINGS ---")

    with httpx.Client(timeout=30, trust_env=False) as session:
        for i in range(1, 4):
            response = None

            try:
                request = session.build_request(
                    "POST",
                    url,
                    headers=headers,
                    json=body,
                )

                started = perf_counter()

                response = session.send(
                    request,
                    stream=True,
                )

                headers_ms = (perf_counter() - started) * 1000

                body_started = perf_counter()
                response.read()
                body_ms = (perf_counter() - body_started) * 1000

                response.raise_for_status()

                result = response.json()

                print(
                    f"Request {i}: "
                    f"headers_wait_ms={headers_ms:.2f}, "
                    f"body_download_ms={body_ms:.2f}, "
                    f"total_ms={headers_ms + body_ms:.2f}, "
                    f"qdrant_server_time={result.get('time')}, "
                    f"status={response.status_code}"
                )

            except Exception as exc:
                print(
                    f"Request {i} failed:",
                    type(exc).__name__,
                    str(exc),
                )

            finally:
                if response is not None:
                    response.close()

def compare_localhost_and_ipv4(
    *,
    url: str,
    headers: dict,
    body: dict,
) -> None:
    """Compare fresh HTTP connections using localhost and IPv4."""
    print("\n--- LOCALHOST VS IPV4 CONNECTION TEST ---")

    ipv4_url = url.replace("localhost", "127.0.0.1")

    targets = [
        ("localhost", url),
        ("127.0.0.1", ipv4_url),
    ]

    for label, target_url in targets:
        print(f"\nTesting {label}")

        for i in range(1, 4):
            try:
                # A fresh client forces a new HTTP connection.
                with httpx.Client(
                    timeout=30,
                    trust_env=False,
                ) as session:
                    started = perf_counter()

                    response = session.post(
                        target_url,
                        headers=headers,
                        json=body,
                    )

                    elapsed = (perf_counter() - started) * 1000
                    response.raise_for_status()

                    result = response.json()

                    print(
                        f"Request {i}: "
                        f"client_total_ms={elapsed:.2f}, "
                        f"qdrant_server_time={result.get('time')}"
                    )

            except Exception as exc:
                print(
                    f"Request {i} failed: "
                    f"{type(exc).__name__}: {exc}"
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

def benchmark_rest_vs_grpc(
    *,
    collection: str,
    vector: list[float],
    query_filter,
) -> None:
    """Compare REST and gRPC using identical Qdrant queries."""

    print("\n--- REST VS gRPC BENCHMARK ---")

    common_config = {
        "url": settings.qdrant_url,
        "api_key": settings.qdrant_api_key or None,
        "timeout": settings.qdrant_timeout,
    }

    rest_client = QdrantClient(
        **common_config,
        prefer_grpc=False,
    )

    grpc_client = QdrantClient(
        **common_config,
        prefer_grpc=True,
        grpc_port=6334,
    )

    clients = {
        "REST": rest_client,
        "gRPC": grpc_client,
    }

    def execute(client):
        return client.query_points(
            collection_name=collection,
            query=vector,
            limit=TOP_K,
            query_filter=query_filter,
            with_payload=True,
            with_vectors=False,
        )

    def result_ids(response):
        return [
            str(point.payload.get("chunk_id") or point.id)
            for point in response.points
        ]

    timings = {"REST": [], "gRPC": []}
    final_ids = {}

    try:
        # Warm up both transports and establish reusable connections.
        print("Warming up both transports...")

        for _ in range(3):
            for name, client in clients.items():
                response = execute(client)
                final_ids[name] = result_ids(response)

        # Alternate execution order to reduce ordering bias.
        print("Running 20 measured requests per transport...")

        for iteration in range(20):
            order = (
                ("REST", "gRPC")
                if iteration % 2 == 0
                else ("gRPC", "REST")
            )

            for name in order:
                started = perf_counter()
                response = execute(clients[name])
                elapsed = (perf_counter() - started) * 1000

                timings[name].append(elapsed)
                final_ids[name] = result_ids(response)

        def percentile(values, percentile_value):
            ordered = sorted(values)
            index = max(
                0,
                math.ceil(percentile_value * len(ordered)) - 1,
            )
            return ordered[index]

        for name, values in timings.items():
            print(f"\n{name}:")
            print(f"  Requests: {len(values)}")
            print(f"  Median: {statistics.median(values):.2f} ms")
            print(f"  P95: {percentile(values, 0.95):.2f} ms")
            print(f"  Min: {min(values):.2f} ms")
            print(f"  Max: {max(values):.2f} ms")

        rest_ids = final_ids["REST"]
        grpc_ids = final_ids["gRPC"]

        print("\n--- RESULT EQUIVALENCE ---")
        print(f"REST result count: {len(rest_ids)}")
        print(f"gRPC result count: {len(grpc_ids)}")
        print(f"Ordered chunk IDs identical: {rest_ids == grpc_ids}")
        print(f"Same chunk ID set: {set(rest_ids) == set(grpc_ids)}")

        if rest_ids != grpc_ids:
            print("REST IDs:", rest_ids)
            print("gRPC IDs:", grpc_ids)

    finally:
        rest_client.close()
        grpc_client.close()


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
    collection = store.current_versioned_collection()
    if not collection:
        raise RuntimeError(
            f"No active collection found for alias: {store.alias_name()}"
        )

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

    # Test DNS, TCP, and optional TLS independently.
    test_connection_phases(url)

    # Test response-header wait separately from body download.
    test_httpx_response_phases(
        url=url,
        headers=headers,
        body=body,
    )

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
    compare_localhost_and_ipv4(
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

    # Test 5: Compare REST vs gRPC performance.
    benchmark_rest_vs_grpc(
        collection=collection,
        vector=vector,
        query_filter=query_filter,
    )

if __name__ == "__main__":
    main()