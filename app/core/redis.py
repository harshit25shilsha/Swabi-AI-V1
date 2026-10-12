from __future__ import annotations

from typing import Any

import redis.asyncio as redis

from app.core.config import settings
from app.core.logging import get_logger

log = get_logger(__name__)


class RedisManager:
    def __init__(self, url: str | None = None) -> None:
        self.url = url if url is not None else settings.redis_url
        self._client: redis.Redis | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.url)

    @property
    def client(self) -> redis.Redis | None:
        return self._client

    async def connect(self) -> None:
        if not self.enabled:
            log.info("Redis disabled (no URL configured)")
            return
        try:
            self._client = redis.from_url(self.url, decode_responses=True)
            await self._client.ping()
            log.info("Redis connected")
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("Redis connection failed: %s", exc)
            self._client = None

    async def close(self) -> None:
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:  # pragma: no cover
                pass
            self._client = None

    async def ping(self) -> bool:
        if self._client is None:
            return False
        try:
            return bool(await self._client.ping())
        except Exception:
            return False

    async def get(self, key: str) -> str | None:
        if self._client is None:
            return None
        return await self._client.get(key)

    async def set(self, key: str, value: Any, ex: int | None = None) -> None:
        if self._client is None:
            return
        await self._client.set(key, value, ex=ex)

    async def delete(self, key: str) -> None:
        if self._client is None:
            return
        await self._client.delete(key)


redis_manager = RedisManager()
