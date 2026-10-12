"""Health and readiness endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from app.core.config import settings
from app.core.redis import redis_manager

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict:
    """Liveness probe. Does not check external dependencies."""
    return {
        "status": "ok",
        "app": settings.app_name,
        "version": settings.app_version,
        "env": settings.app_env,
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
