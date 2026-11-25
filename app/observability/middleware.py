
"""Per-request observability for chat safety and support APIs."""

from __future__ import annotations

import logging
import time

from app.core.logging import request_id_var
from app.observability import counters

logger = logging.getLogger("guardrail.request")

SUPPORT_PATH = "/api/v1/support/ask"


class ObservabilityMiddleware:
    """ASGI middleware for request timing and observability."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        state = scope.setdefault("state", {})
        request_id = request_id_var.get()

        state["request_id"] = request_id
        state["moderation_result"] = None
        state["message_hash"] = None

        path = scope.get("path", "")
        method = scope.get("method", "")

        is_support_request = path == SUPPORT_PATH

        start = time.perf_counter()
        status_code = None

        async def send_wrapper(message):
            nonlocal status_code

            if message["type"] == "http.response.start":
                status_code = message["status"]

            await send(message)

            if (
                is_support_request
                and message["type"] == "http.response.body"
                and not message.get("more_body", False)
            ):
                elapsed_ms = (time.perf_counter() - start) * 1000

                logger.info(
                    "support_request_completed",
                    extra={
                        "event": "support_request_timing",
                        "request_id": request_id,
                        "method": method,
                        "path": path,
                        "status_code": status_code,
                        "total_request_ms": round(elapsed_ms, 2),
                    },
                )

        try:
            await self.app(scope, receive, send_wrapper)

        except Exception:
            elapsed_ms = (time.perf_counter() - start) * 1000

            logger.exception(
                "request_failed",
                extra={
                    "request_id": request_id,
                    "path": path,
                    "method": method,
                    "latency_ms": round(elapsed_ms, 2),
                },
            )
            raise

        # Preserve existing chat-safety observability.
        result = state.get("moderation_result")

        if result is not None:
            elapsed_ms = (time.perf_counter() - start) * 1000

            counters.increment("requests_total")
            counters.observe_latency(elapsed_ms)
            counters.increment(f"source:{result.source}")

            if result.action.value == "BLOCK":
                counters.increment("blocks_total")

                if result.category is not None:
                    counters.increment(
                        f"blocks_by_category:{result.category.value}"
                    )
            else:
                counters.increment("allows_total")

            logger.info(
                "moderation",
                extra={
                    "event": "moderation",
                    "request_id": request_id,
                    "action": result.action.value,
                    "category": (
                        result.category.value if result.category else None
                    ),
                    "confidence": result.confidence,
                    "source": result.source,
                    "latency_ms": round(elapsed_ms, 2),
                    "message_hash": state.get("message_hash"),
                    "prompt_version": result.prompt_version,
                },
            )
