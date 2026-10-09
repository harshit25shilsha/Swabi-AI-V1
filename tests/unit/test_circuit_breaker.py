"""Tests for the shared circuit breaker (state machine only).

Moderator-integration tests (provider ladder, all circuits open) land in
step 9 with the LLM adapter.
"""

from __future__ import annotations

from app.core.exceptions import (
    CircuitOpenError,
    GuardrailUnavailableError,
)
from app.observability import circuit_breaker as cbmod
from app.observability.circuit_breaker import CircuitBreaker, CircuitState

# ---------------------------------------------------------------------------
# CircuitOpenError wiring
# ---------------------------------------------------------------------------


def test_circuit_open_error_is_reexported():
    """GDv1 imports from this module must resolve to the shared class."""
    from app.observability.circuit_breaker import CircuitOpenError as Reexported

    assert Reexported is CircuitOpenError


def test_circuit_open_error_is_guardrail_unavailable():
    assert issubclass(CircuitOpenError, GuardrailUnavailableError)


# ---------------------------------------------------------------------------
# CLOSED behavior
# ---------------------------------------------------------------------------


def test_closed_allows_calls():
    cb = CircuitBreaker(failure_threshold=3, open_seconds=60)
    assert cb.state == CircuitState.CLOSED
    assert cb.allow_call() is True


def test_success_resets_failure_count():
    cb = CircuitBreaker(failure_threshold=3, open_seconds=60)
    cb.record_failure()
    cb.record_failure()
    cb.record_success()
    cb.record_failure()
    cb.record_failure()
    # Only two consecutive failures after the success — still CLOSED.
    assert cb.state == CircuitState.CLOSED


# ---------------------------------------------------------------------------
# Opening
# ---------------------------------------------------------------------------


def test_opens_after_threshold():
    cb = CircuitBreaker(failure_threshold=3, open_seconds=60)
    for _ in range(3):
        cb.record_failure()
    assert cb.state == CircuitState.OPEN


def test_open_rejects_calls():
    cb = CircuitBreaker(failure_threshold=1, open_seconds=60)
    cb.record_failure()
    assert cb.allow_call() is False


# ---------------------------------------------------------------------------
# Half-open transitions
# ---------------------------------------------------------------------------


def test_half_open_after_timeout(monkeypatch):
    fake = [100.0]
    monkeypatch.setattr(cbmod.time, "monotonic", lambda: fake[0])

    cb = CircuitBreaker(failure_threshold=1, open_seconds=60)
    cb.record_failure()
    assert cb.state == CircuitState.OPEN

    fake[0] += 61
    assert cb.state == CircuitState.HALF_OPEN


def test_half_open_success_closes(monkeypatch):
    fake = [100.0]
    monkeypatch.setattr(cbmod.time, "monotonic", lambda: fake[0])

    cb = CircuitBreaker(failure_threshold=1, open_seconds=60)
    cb.record_failure()
    fake[0] += 61
    assert cb.allow_call() is True  # probe allowed
    cb.record_success()
    assert cb.state == CircuitState.CLOSED


def test_half_open_failure_reopens(monkeypatch):
    fake = [100.0]
    monkeypatch.setattr(cbmod.time, "monotonic", lambda: fake[0])

    cb = CircuitBreaker(failure_threshold=1, open_seconds=60)
    cb.record_failure()
    fake[0] += 61
    assert cb.allow_call() is True
    cb.record_failure()
    assert cb.state == CircuitState.OPEN
    assert cb.allow_call() is False  # still open


def test_half_open_only_one_probe_at_a_time(monkeypatch):
    fake = [100.0]
    monkeypatch.setattr(cbmod.time, "monotonic", lambda: fake[0])

    cb = CircuitBreaker(failure_threshold=1, open_seconds=60)
    cb.record_failure()
    fake[0] += 61
    assert cb.allow_call() is True  # probe allowed
    assert cb.allow_call() is False  # second rejected


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_metrics_reports_state_and_counts():
    cb = CircuitBreaker(failure_threshold=2, open_seconds=60)
    cb.record_failure()
    cb.record_failure()  # trips

    m = cb.metrics()
    assert m["state"] == "open"
    assert m["trips_total"] == 1
    assert m["failures"] == 2


def test_metrics_rejection_count_increments_on_open_call():
    cb = CircuitBreaker(failure_threshold=1, open_seconds=60)
    cb.record_failure()
    cb.allow_call()
    cb.allow_call()
    assert cb.metrics()["rejections_total"] == 2


def test_reset_restores_closed_state():
    cb = CircuitBreaker(failure_threshold=1, open_seconds=60)
    cb.record_failure()
    cb.reset()
    m = cb.metrics()
    assert m["state"] == "closed"
    assert m["trips_total"] == 0
    assert m["rejections_total"] == 0
    assert m["failures"] == 0
