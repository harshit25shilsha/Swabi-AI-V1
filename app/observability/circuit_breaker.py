"""Simple thread-safe circuit breaker.

States:
  CLOSED    → calls pass through. Consecutive failures tracked.
  OPEN      → calls short-circuit immediately (fail-open at the caller).
              After ``open_seconds``, the next call triggers HALF_OPEN.
  HALF_OPEN → exactly one probe call is allowed.
              Success → CLOSED. Failure → OPEN for another ``open_seconds``.

Not a full library — no sliding windows, no per-error-type policies.
Sufficient for a single-instance guardrail service.

``CircuitOpenError`` is re-exported from ``app.core.exceptions`` so call
sites can catch the same class regardless of which module they import it
from.
"""

from __future__ import annotations

import logging
import threading
import time
from enum import StrEnum

from app.core.exceptions import CircuitOpenError

__all__ = ["CircuitBreaker", "CircuitState", "CircuitOpenError"]

logger = logging.getLogger("observability.circuit")


class CircuitState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, open_seconds: int = 60):
        self._failure_threshold = max(1, failure_threshold)
        self._open_seconds = max(1, open_seconds)
        self._lock = threading.Lock()

        self._state = CircuitState.CLOSED
        self._failures = 0
        self._opened_at = 0.0
        self._half_open_probe_in_flight = False

        # Lifetime counters for /metrics
        self._trips_total = 0
        self._rejections_total = 0

    @property
    def state(self) -> CircuitState:
        with self._lock:
            self._maybe_half_open()
            return self._state

    def allow_call(self) -> bool:
        """Return True if the caller may attempt the underlying call.

        False means short-circuit (fail-open at the service layer).
        """
        with self._lock:
            self._maybe_half_open()

            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_probe_in_flight:
                    self._rejections_total += 1
                    return False
                self._half_open_probe_in_flight = True
                return True

            # OPEN
            self._rejections_total += 1
            return False

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._half_open_probe_in_flight = False
            if self._state != CircuitState.CLOSED:
                logger.info(
                    "circuit_close",
                    extra={"from_state": self._state.value},
                )
                self._state = CircuitState.CLOSED

    def record_failure(self) -> None:
        with self._lock:
            self._half_open_probe_in_flight = False
            self._failures += 1

            if self._state == CircuitState.HALF_OPEN:
                self._open_locked()
                return

            if self._state == CircuitState.CLOSED and self._failures >= self._failure_threshold:
                self._open_locked()

    def _open_locked(self) -> None:
        self._state = CircuitState.OPEN
        self._opened_at = time.monotonic()
        self._trips_total += 1
        logger.warning(
            "circuit_open",
            extra={
                "failures": self._failures,
                "open_seconds": self._open_seconds,
            },
        )

    def _maybe_half_open(self) -> None:
        if self._state != CircuitState.OPEN:
            return
        if time.monotonic() - self._opened_at >= self._open_seconds:
            self._state = CircuitState.HALF_OPEN
            self._half_open_probe_in_flight = False
            logger.info("circuit_half_open")

    def metrics(self) -> dict:
        with self._lock:
            self._maybe_half_open()
            return {
                "state": self._state.value,
                "failures": self._failures,
                "trips_total": self._trips_total,
                "rejections_total": self._rejections_total,
            }

    def reset(self) -> None:
        """Test helper — restore initial state."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failures = 0
            self._opened_at = 0.0
            self._half_open_probe_in_flight = False
            self._trips_total = 0
            self._rejections_total = 0
