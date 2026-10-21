"""Health and readiness endpoints."""

from __future__ import annotations

import time

from fastapi import APIRouter

from app.capabilities.chat_safety.prompts import PROMPT_VERSION
from app.core.config import settings
from app.core.redis import redis_manager

router = APIRouter(tags=["health"])

_START_TIME = time.time()


@router.get("/health")
async def health() -> dict:
    """Liveness probe. Superset of the Phase 0 and GDv1 health shapes."""
    return {
        # Phase 0 fields
        "status": "ok",
        "app": settings.app_name,
        "version": settings.app_version,
        "env": settings.app_env,
        # Guardrail fields (for backward compatibility with GDv1 consumers)
        "uptime_seconds": round(time.time() - _START_TIME, 1),
        "service_version": settings.service_version,
        "prompt_version": PROMPT_VERSION,
    }


@router.get("/ready")
async def ready() -> dict:
    """Readiness probe. Checks Redis when configured."""
    redis_status: bool | str
    if not redis_manager.enabled:
        redis_status = "skipped"
        status = "ready"
    else:
        ok = await redis_manager.ping()
        redis_status = ok
        status = "ready" if ok else "degraded"

    return {"status": status, "redis": redis_status}
