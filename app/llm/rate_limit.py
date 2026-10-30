"""Token-per-minute pacing for Groq requests."""

from __future__ import annotations

import atexit
import math
import random
import re
import threading
import time
import uuid
from collections import deque
from collections.abc import Mapping

import redis
from redis.exceptions import RedisError

from app.core.config import settings
from app.core.exceptions import AIProviderError
from app.core.logging import get_logger

log = get_logger(__name__)

_RESERVE_TOKENS_SCRIPT = """
local now = redis.call('TIME')
local now_ms = now[1] * 1000 + math.floor(now[2] / 1000)
local window_ms = tonumber(ARGV[1])
local capacity = tonumber(ARGV[2])
local cost = tonumber(ARGV[3])
local cutoff = now_ms - window_ms

if capacity <= 0 then
    local reported_limit = redis.call('GET', KEYS[2])
    if not reported_limit then
        return 0
    end
    capacity = math.floor(tonumber(reported_limit) * tonumber(ARGV[5]))
end

if cost > capacity then
    return -1
end

redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', cutoff)
local entries = redis.call('ZRANGE', KEYS[1], 0, -1, 'WITHSCORES')
local used = 0
for i = 1, #entries, 2 do
    local amount = string.match(entries[i], '|(%d+)$')
    used = used + tonumber(amount)
end

if used + cost <= capacity then
    redis.call('ZADD', KEYS[1], now_ms, ARGV[4])
    redis.call('PEXPIRE', KEYS[1], window_ms * 2)
    return 0
end

local oldest = redis.call('ZRANGE', KEYS[1], 0, 0, 'WITHSCORES')
if #oldest == 0 then
    return 100
end
return math.max(1, tonumber(oldest[2]) + window_ms - now_ms)
"""

_RECONCILE_TOKENS_SCRIPT = """
local score = redis.call('ZSCORE', KEYS[1], ARGV[1])
local actual_tokens = tonumber(ARGV[3])
local now = redis.call('TIME')
local now_ms = now[1] * 1000 + math.floor(now[2] / 1000)
if not score then
    if actual_tokens > 0 then
        redis.call('ZADD', KEYS[1], now_ms, ARGV[2])
        redis.call('PEXPIRE', KEYS[1], tonumber(ARGV[4]))
    end
    return 0
end

redis.call('ZREM', KEYS[1], ARGV[1])
if actual_tokens > 0 then
    redis.call('ZADD', KEYS[1], score, ARGV[2])
    redis.call('PEXPIRE', KEYS[1], tonumber(ARGV[4]))
end
return 1
"""


class GroqTokenRateLimiter:
    """Reserve estimated request tokens in a rolling TPM window.

    Redis coordinates multiple app processes when ``REDIS_URL`` is set.
    Without Redis, reservations are shared by threads in this process only.
    """

    def __init__(
        self,
        *,
        limit: int | None,
        utilization: float,
        window_seconds: int = 60,
        redis_url: str = "",
        model: str = "",
    ) -> None:
        self._configured_capacity = (
            max(1, math.floor(limit * utilization)) if limit is not None else None
        )
        self.capacity = self._configured_capacity
        self._utilization = utilization
        self.window_seconds = window_seconds
        self._lock = threading.Lock()
        self._reservations: deque[tuple[float, str, int]] = deque()
        self._redis_url = redis_url
        self._model = model
        self._redis_client: redis.Redis | None = None
        self._redis_lock = threading.Lock()

    def wait_for_capacity(self, estimated_tokens: int) -> str:
        cost = max(1, estimated_tokens)
        if self.capacity is not None and cost > self.capacity:
            raise AIProviderError(
                "Estimated Groq request size "
                f"({cost} tokens) exceeds the active TPM budget "
                f"({self.capacity}); reduce prompt/output size or adjust "
                "GROQ_TPM_LIMIT/GROQ_TPM_TARGET_UTILIZATION."
            )

        reservation_id = f"{time.time_ns()}:{uuid.uuid4().hex}"
        while True:
            if self._redis_url:
                delay = self._reserve_distributed(cost, reservation_id)
            else:
                delay = self._reserve_local(cost, reservation_id)
            if delay <= 0:
                return reservation_id
            log.info(
                "Groq TPM pacing request",
                extra={
                    "estimated_tokens": cost,
                    "wait_seconds": round(delay, 2),
                    "model": self._model,
                },
            )
            time.sleep(delay + random.uniform(0.0, min(0.1, delay * 0.05)))

    def reconcile_reservation(
        self,
        reservation_id: str,
        reserved_tokens: int,
        actual_tokens: int,
    ) -> None:
        """Replace an estimate with provider-reported usage after a response."""
        actual = max(0, actual_tokens)
        if self._redis_url:
            self._reconcile_distributed(reservation_id, reserved_tokens, actual)
        else:
            self._reconcile_local(reservation_id, actual)

    def update_from_response(self, headers: Mapping[str, str], body: str = "") -> None:
        """Learn the provider's actual model quota when no override is set."""
        if self._configured_capacity is not None:
            return

        reported_limit: int | None = None
        raw_limit = headers.get("x-ratelimit-limit-tokens")
        if raw_limit:
            try:
                reported_limit = int(raw_limit)
            except (TypeError, ValueError):
                log.warning("Ignoring invalid Groq token-limit response header")

        if reported_limit is None:
            match = re.search(r"\bTPM:\s*Limit\s+([\d,]+)", body, re.IGNORECASE)
            if match:
                reported_limit = int(match.group(1).replace(",", ""))

        if reported_limit is None or reported_limit < 1:
            return

        self.capacity = max(1, math.floor(reported_limit * self._utilization))
        log.info(
            "Groq TPM limit detected",
            extra={
                "reported_limit": reported_limit,
                "target_capacity": self.capacity,
                "model": self._model,
            },
        )
        if self._redis_url:
            try:
                self._get_redis_client().set(
                    self._redis_limit_key,
                    reported_limit,
                    ex=86400,
                )
            except RedisError as exc:
                log.error(
                    "Could not publish Groq TPM limit to Redis; "
                    "cross-process quota coordination is unavailable: %s",
                    exc,
                )
                self._disable_redis()

    def _reserve_local(self, cost: int, reservation_id: str) -> float:
        now = time.monotonic()
        with self._lock:
            if self.capacity is None:
                return 0.0
            cutoff = now - self.window_seconds
            while self._reservations and self._reservations[0][0] <= cutoff:
                self._reservations.popleft()

            used = sum(tokens for _, _, tokens in self._reservations)
            if used + cost <= self.capacity:
                self._reservations.append((now, reservation_id, cost))
                return 0.0
            return max(
                0.001,
                self._reservations[0][0] + self.window_seconds - now,
            )

    def _reconcile_local(self, reservation_id: str, actual_tokens: int) -> None:
        with self._lock:
            updated: deque[tuple[float, str, int]] = deque()
            found = False
            for created_at, current_id, reserved_tokens in self._reservations:
                if current_id != reservation_id:
                    updated.append((created_at, current_id, reserved_tokens))
                elif actual_tokens > 0:
                    found = True
                    updated.append((created_at, current_id, actual_tokens))
                else:
                    found = True
            if not found and actual_tokens > 0 and self.capacity is not None:
                updated.append((time.monotonic(), reservation_id, actual_tokens))
            self._reservations = updated

    def _get_redis_client(self) -> redis.Redis:
        if self._redis_client is None:
            with self._redis_lock:
                if self._redis_client is None:
                    self._redis_client = redis.Redis.from_url(
                        self._redis_url,
                        decode_responses=True,
                        socket_connect_timeout=3,
                        socket_timeout=3,
                    )
        return self._redis_client

    @property
    def _redis_limit_key(self) -> str:
        return f"groq:tpm:limit:{self._model}"

    def _disable_redis(self) -> None:
        client = self._redis_client
        self._redis_client = None
        self._redis_url = ""
        if client is not None:
            client.close()

    def _reserve_distributed(self, cost: int, reservation_id: str) -> float:
        key = f"groq:tpm:{self._model}"
        request_member = f"{reservation_id}|{cost}"
        configured_capacity = self._configured_capacity or 0
        try:
            wait_ms = self._get_redis_client().eval(
                _RESERVE_TOKENS_SCRIPT,
                2,
                key,
                self._redis_limit_key,
                self.window_seconds * 1000,
                configured_capacity,
                cost,
                request_member,
                self._utilization,
            )
        except RedisError as exc:
            log.error(
                "Groq TPM limiter could not reach Redis; falling back to "
                "process-local pacing, so app workers are no longer coordinated: %s",
                exc,
            )
            self._disable_redis()
            return self._reserve_local(cost, reservation_id)
        wait_ms = int(wait_ms)
        if wait_ms < 0:
            raise AIProviderError(
                f"Estimated Groq request size ({cost} tokens) exceeds the "
                "active TPM budget configured for the shared provider quota."
            )
        return wait_ms / 1000

    def _reconcile_distributed(
        self,
        reservation_id: str,
        reserved_tokens: int,
        actual_tokens: int,
    ) -> None:
        key = f"groq:tpm:{self._model}"
        try:
            self._get_redis_client().eval(
                _RECONCILE_TOKENS_SCRIPT,
                1,
                key,
                f"{reservation_id}|{reserved_tokens}",
                f"{reservation_id}|{actual_tokens}",
                actual_tokens,
                self.window_seconds * 1000,
            )
        except RedisError as exc:
            log.error(
                "Could not reconcile Groq TPM reservation in Redis; "
                "cross-process quota coordination is unavailable: %s",
                exc,
            )
            self._disable_redis()

    def close(self) -> None:
        if self._redis_client is not None:
            self._redis_client.close()
            self._redis_client = None


groq_tpm_limiter = GroqTokenRateLimiter(
    limit=settings.groq_tpm_limit,
    utilization=settings.groq_tpm_target_utilization,
    window_seconds=settings.groq_tpm_window_seconds,
    redis_url=settings.redis_url,
    model=settings.llm_model,
)

atexit.register(groq_tpm_limiter.close)
