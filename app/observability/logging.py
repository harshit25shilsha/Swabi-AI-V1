"""Helpers for safe, structured observability logging."""

from __future__ import annotations

from typing import Any

from app.core.logging import get_logger

log = get_logger("observability")


def log_llm_call(
    *,
    model: str,
    latency_ms: float,
    usage: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    log.info(
        "llm_call model=%s latency_ms=%.1f usage=%s extra=%s",
        model,
        latency_ms,
        usage or {},
        extra or {},
    )


def log_tool_call(
    *,
    tool_name: str,
    success: bool,
    error: str | None = None,
    latency_ms: float | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    log.info(
        "tool_call name=%s success=%s latency_ms=%s error=%s extra=%s",
        tool_name,
        success,
        f"{latency_ms:.1f}" if latency_ms is not None else "-",
        error or "-",
        extra or {},
    )


def log_swabi_request(
    *,
    method: str,
    path: str,
    status: int | None,
    latency_ms: float,
) -> None:
    log.info(
        "swabi_request method=%s path=%s status=%s latency_ms=%.1f",
        method,
        path,
        status if status is not None else "-",
        latency_ms,
    )
