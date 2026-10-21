"""Per-request observability: timing, JSON log, counters.

Runs as the innermost middleware so it can observe the moderation result
the route stores on ``request.state``. It reads the request ID from the
outer request-ID middleware's contextvar; it does NOT generate its own.

Note: ``X-Request-ID`` is intentionally NOT set here. The outer
``request_id_middleware`` in ``app.main`` is the single source for that
response header.
"""

from __future__ import annotations

import logging
import time

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.logging import request_id_var
from app.observability import counters

logger = logging.getLogger("guardrail.request")


class ObservabilityMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request_id_var.get()

        # Reset per-request state on scope["state"] so the route can write
        # to it and we can read it back below.
        request.state.request_id = request_id
        request.state.moderation_result = None
        request.state.message_hash = None

        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.exception(
                "unhandled_exception",
                extra={
                    "request_id": request_id,
                    "path": request.url.path,
                    "latency_ms": round(elapsed_ms, 2),
                },
            )
            raise

        elapsed_ms = (time.perf_counter() - start) * 1000

        result = getattr(request.state, "moderation_result", None)
        if result is not None:
            counters.increment("requests_total")
            counters.observe_latency(elapsed_ms)
            counters.increment(f"source:{result.source}")

            if result.action.value == "BLOCK":
                counters.increment("blocks_total")
                if result.category is not None:
                    counters.increment(f"blocks_by_category:{result.category.value}")
            else:
                counters.increment("allows_total")

            logger.info(
                "moderation",
                extra={
                    "event": "moderation",
                    "request_id": request_id,
                    "action": result.action.value,
                    "category": (result.category.value if result.category else None),
                    "confidence": result.confidence,
                    "source": result.source,
                    "latency_ms": round(elapsed_ms, 2),
                    "message_hash": request.state.message_hash,
                    "prompt_version": result.prompt_version,
                },
            )

        return response
