"""Sequential latency benchmark for POST /api/v1/validate-message.

Measures end-to-end latency, including LLM inference. Requests are sent
sequentially, not concurrently. Use a separate load test for throughput.

Usage:
    python -m scripts.benchmark
"""

from __future__ import annotations

import statistics
import time

import httpx

from app.core.config import settings

URL = "http://127.0.0.1:8000/api/v1/validate-message"
N = 50
TIMEOUT_SECONDS = 60

if not settings.guardrail_api_token:
    raise RuntimeError("GUARDRAIL_API_TOKEN is missing from configuration")

HEADERS = {
    "Authorization": f"Bearer {settings.guardrail_api_token}",
    "Content-Type": "application/json",
}

# Mix of ordinary requests to measure typical end-to-end latency.
MESSAGES = [
    {"message": "I need a hotel room for two nights."},
    {"message": "Is the villa available from 10th to 12th October?"},
    {"message": "What payment methods does the platform support?"},
]


def main() -> None:
    timings: list[float] = []

    with httpx.Client(timeout=TIMEOUT_SECONDS) as client:
        # Warm up the connection and server request path.
        response = client.post(URL, headers=HEADERS, json=MESSAGES[0])
        response.raise_for_status()

        for i in range(N):
            payload = MESSAGES[i % len(MESSAGES)]

            start = time.perf_counter()
            response = client.post(URL, headers=HEADERS, json=payload)
            elapsed_ms = (time.perf_counter() - start) * 1000

            response.raise_for_status()
            timings.append(elapsed_ms)

    timings_sorted = sorted(timings)
    p95_index = max(0, int(len(timings_sorted) * 0.95) - 1)

    print(f"Requests: {len(timings)}")
    print(f"Average:  {statistics.mean(timings):.1f} ms")
    print(f"Median:   {statistics.median(timings):.1f} ms")
    print(f"Min:      {min(timings):.1f} ms")
    print(f"Max:      {max(timings):.1f} ms")
    print(f"P95:      {timings_sorted[p95_index]:.1f} ms")


if __name__ == "__main__":
    main()
