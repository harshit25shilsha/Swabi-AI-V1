"""Small fail-open Redis response-cache primitives.

Only callers that can safely reuse a successful response should use this
module.  It intentionally does not cache exceptions or HTTP error responses.
"""

from __future__ import annotations

import hashlib
import json
import secrets
from typing import Any

from app.core.logging import get_logger
from app.core.redis import redis_manager

log = get_logger(__name__)


def normalize_message(message: str) -> str:
    """Normalize only insignificant whitespace without changing meaning."""
    return " ".join(message.split())


def make_cache_key(namespace: str, payload: dict[str, Any]) -> str:
    """Return a stable, opaque, versionable Redis key for a JSON payload."""
    canonical_payload = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical_payload.encode("utf-8")).hexdigest()
    return f"swabi:cache:{namespace}:{digest}"


async def get_json(key: str) -> dict[str, Any] | None:
    """Read a JSON object, treating Redis/payload failures as a cache miss."""
    try:
        raw = await redis_manager.get(key)
        if raw is None:
            return None
        value = json.loads(raw)
        if isinstance(value, dict):
            return value
        log.warning(
            "Response cache payload is not an object",
            extra={"event": "response_cache_invalid"},
        )
    except Exception:
        log.warning(
            "Response cache read failed; bypassing cache",
            extra={"event": "response_cache_read_failed"},
        )
    return None


async def set_json(key: str, value: dict[str, Any], *, ttl_seconds: int) -> None:
    """Best-effort JSON write. Cache failures never affect API success."""
    try:
        await redis_manager.set(
            key,
            json.dumps(value, ensure_ascii=False),
            ex=ttl_seconds,
        )
    except Exception:
        log.warning(
            "Response cache write failed",
            extra={"event": "response_cache_write_failed"},
        )


async def acquire_lock(key: str, *, ttl_seconds: int) -> tuple[bool, str | None]:
    """Try to own a short Redis lock.

    The first result says whether Redis was available. When it is available,
    a token means this caller owns the lock and ``None`` means another caller
    owns it. This distinction prevents Redis failures from adding a wait to
    every request.
    """
    client = redis_manager.client
    if client is None:
        return False, None

    token = secrets.token_urlsafe(24)
    try:
        acquired = await client.set(f"{key}:lock", token, ex=ttl_seconds, nx=True)
        return True, token if acquired else None
    except Exception:
        log.warning(
            "Response cache lock acquisition failed; bypassing lock",
            extra={"event": "response_cache_lock_failed"},
        )
        return False, None


async def release_lock(key: str, token: str) -> None:
    """Release a lock only if it still belongs to this caller."""
    client = redis_manager.client
    if client is None:
        return

    try:
        await client.eval(
            "if redis.call('GET', KEYS[1]) == ARGV[1] then "
            "return redis.call('DEL', KEYS[1]) else return 0 end",
            1,
            f"{key}:lock",
            token,
        )
    except Exception:
        log.warning(
            "Response cache lock release failed",
            extra={"event": "response_cache_lock_release_failed"},
        )
