"""Tests for the shared in-memory counters module."""

from __future__ import annotations

import pytest

from app.observability import counters


@pytest.fixture(autouse=True)
def _reset_counters():
    counters.reset()
    yield
    counters.reset()


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def test_increment_defaults_to_one():
    counters.increment("requests_total")
    counters.increment("requests_total")
    assert counters.snapshot()["requests_total"] == 2


def test_increment_accepts_amount():
    counters.increment("requests_total", 5)
    assert counters.snapshot()["requests_total"] == 5


def test_observe_latency_records_sample():
    counters.observe_latency(12.5)
    snap = counters.snapshot()
    assert snap["latency_p95_ms"] == 12.5


# ---------------------------------------------------------------------------
# Snapshot shape
# ---------------------------------------------------------------------------


def test_snapshot_returns_expected_keys():
    snap = counters.snapshot()
    for key in (
        "requests_total",
        "blocks_total",
        "allows_total",
        "llm_calls_total",
        "llm_failures_total",
        "fail_open_total",
        "blocks_by_category",
        "sources",
        "llm_calls_by_provider",
        "llm_failures_by_provider",
        "latency_p95_ms",
    ):
        assert key in snap


def test_snapshot_defaults_to_zero():
    snap = counters.snapshot()
    assert snap["requests_total"] == 0
    assert snap["blocks_total"] == 0
    assert snap["allows_total"] == 0
    assert snap["latency_p95_ms"] == 0.0
    assert snap["blocks_by_category"] == {}
    assert snap["sources"] == {}


# ---------------------------------------------------------------------------
# Split-by-prefix counters
# ---------------------------------------------------------------------------


def test_blocks_by_category_split():
    counters.increment("blocks_by_category:PHONE_NUMBER")
    counters.increment("blocks_by_category:PHONE_NUMBER")
    counters.increment("blocks_by_category:EMAIL")
    snap = counters.snapshot()
    assert snap["blocks_by_category"] == {"PHONE_NUMBER": 2, "EMAIL": 1}


def test_sources_split():
    counters.increment("source:deterministic")
    counters.increment("source:llm")
    counters.increment("source:llm")
    snap = counters.snapshot()
    assert snap["sources"] == {"deterministic": 1, "llm": 2}


def test_llm_calls_by_provider_split():
    counters.increment("llm_calls_by_provider:groq")
    counters.increment("llm_calls_by_provider:gemini")
    snap = counters.snapshot()
    assert snap["llm_calls_by_provider"] == {"groq": 1, "gemini": 1}


def test_llm_failures_by_provider_split():
    counters.increment("llm_failures_by_provider:groq")
    snap = counters.snapshot()
    assert snap["llm_failures_by_provider"] == {"groq": 1}


# ---------------------------------------------------------------------------
# p95
# ---------------------------------------------------------------------------


def test_p95_index_for_100_samples():
    """For n=100, idx = int(100*0.95) = 95 (0-based) → 96th smallest value."""
    for i in range(1, 101):  # values 1.0 .. 100.0
        counters.observe_latency(float(i))
    # sorted: latencies[95] == 96.0
    assert counters.snapshot()["latency_p95_ms"] == 96.0


def test_latency_deque_is_bounded_to_1000():
    """Deque maxlen=1000 keeps the last 1000 samples.

    After 1500 calls with values 0.0 .. 1499.0, the deque holds 500.0 .. 1499.0.
    idx = int(1000*0.95) = 950 (0-based); latencies[950] == 500.0 + 950 == 1450.0.
    """
    for i in range(1500):
        counters.observe_latency(float(i))
    assert counters.snapshot()["latency_p95_ms"] == 1450.0


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


def test_reset_clears_counters_and_latencies():
    counters.increment("requests_total", 10)
    counters.observe_latency(50.0)
    counters.reset()
    snap = counters.snapshot()
    assert snap["requests_total"] == 0
    assert snap["latency_p95_ms"] == 0.0
