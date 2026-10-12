"""Thread-safe in-memory counters and latency tracker.

Shared across capabilities. The generic core is ``increment`` +
``observe_latency``; ``snapshot()`` currently returns the chat-safety
field set. When a second capability needs counters, ``snapshot()`` can
be generalised without changing the writer API.

Not persisted. Reset on process restart. Not shared across workers.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Any

_lock = threading.Lock()
_counters: dict[str, int] = {}
_latencies: deque[float] = deque(maxlen=1000)  # last 1000 requests for p95


def increment(name: str, amount: int = 1) -> None:
    """Increment a named counter. Thread-safe."""
    with _lock:
        _counters[name] = _counters.get(name, 0) + amount


def observe_latency(ms: float) -> None:
    """Record a request latency sample (milliseconds). Thread-safe."""
    with _lock:
        _latencies.append(ms)


def snapshot() -> dict[str, Any]:
    """Return a consistent snapshot of counters + p95 latency.

    Split-by-key fields (``blocks_by_category``, ``sources``, etc.) are
    derived from counters whose names follow the ``prefix:value`` pattern.
    """
    with _lock:
        counters = dict(_counters)
        latencies = sorted(_latencies)

    p95 = 0.0
    if latencies:
        idx = min(int(len(latencies) * 0.95), len(latencies) - 1)
        p95 = latencies[idx]

    blocks_by_category = {
        k.split(":", 1)[1]: v for k, v in counters.items() if k.startswith("blocks_by_category:")
    }
    sources = {k.split(":", 1)[1]: v for k, v in counters.items() if k.startswith("source:")}
    llm_calls_by_provider = {
        k.split(":", 1)[1]: v for k, v in counters.items() if k.startswith("llm_calls_by_provider:")
    }
    llm_failures_by_provider = {
        k.split(":", 1)[1]: v
        for k, v in counters.items()
        if k.startswith("llm_failures_by_provider:")
    }

    return {
        "requests_total": counters.get("requests_total", 0),
        "blocks_total": counters.get("blocks_total", 0),
        "allows_total": counters.get("allows_total", 0),
        "llm_calls_total": counters.get("llm_calls_total", 0),
        "llm_failures_total": counters.get("llm_failures_total", 0),
        "fail_open_total": counters.get("fail_open_total", 0),
        "blocks_by_category": blocks_by_category,
        "sources": sources,
        "llm_calls_by_provider": llm_calls_by_provider,
        "llm_failures_by_provider": llm_failures_by_provider,
        "latency_p95_ms": round(p95, 2),
    }


def reset() -> None:
    """Test helper — clears all counter and latency state."""
    with _lock:
        _counters.clear()
        _latencies.clear()
