from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from redis.exceptions import RedisError

from app.core.exceptions import AIProviderError
from app.llm.rate_limit import GroqTokenRateLimiter


def test_local_limiter_waits_until_rolling_window_frees_tokens(monkeypatch):
    now = [0.0]
    sleeps: list[float] = []
    limiter = GroqTokenRateLimiter(
        limit=100,
        utilization=0.5,
        window_seconds=60,
    )
    monkeypatch.setattr("app.llm.rate_limit.time.monotonic", lambda: now[0])

    def sleep(delay: float) -> None:
        sleeps.append(delay)
        now[0] += delay

    monkeypatch.setattr("app.llm.rate_limit.time.sleep", sleep)
    monkeypatch.setattr("app.llm.rate_limit.random.uniform", lambda lower, upper: 0)

    limiter.wait_for_capacity(40)
    limiter.wait_for_capacity(20)

    assert sleeps == [pytest.approx(60)]
    assert now[0] >= 60


def test_local_limiter_reconciles_reservation_with_actual_usage(monkeypatch):
    now = [0.0]
    sleeps: list[float] = []
    limiter = GroqTokenRateLimiter(limit=8000, utilization=0.75)
    monkeypatch.setattr("app.llm.rate_limit.time.monotonic", lambda: now[0])

    def sleep(delay: float) -> None:
        sleeps.append(delay)
        now[0] += delay

    monkeypatch.setattr("app.llm.rate_limit.time.sleep", sleep)
    monkeypatch.setattr("app.llm.rate_limit.random.uniform", lambda lower, upper: 0)

    reservation_id = limiter.wait_for_capacity(3120)
    limiter.reconcile_reservation(reservation_id, 3120, 2170)
    limiter.wait_for_capacity(2925)

    assert sleeps == []


def test_local_limiter_tracks_actual_usage_from_first_auto_detected_call():
    limiter = GroqTokenRateLimiter(limit=None, utilization=0.75)
    reservation_id = limiter.wait_for_capacity(3120)

    limiter.update_from_response({"x-ratelimit-limit-tokens": "8000"})
    limiter.reconcile_reservation(reservation_id, 3120, 2170)

    reservations = list(limiter._reservations)
    assert len(reservations) == 1
    assert reservations[0][1:] == (reservation_id, 2170)


def test_limiter_rejects_request_larger_than_configured_budget():
    limiter = GroqTokenRateLimiter(limit=100, utilization=0.5)

    with pytest.raises(AIProviderError, match="exceeds the active TPM budget"):
        limiter.wait_for_capacity(51)


def test_unconfigured_limit_auto_detects_paid_tier_without_a_static_cap():
    limiter = GroqTokenRateLimiter(
        limit=None,
        utilization=0.75,
    )
    limiter.wait_for_capacity(100_000)

    limiter.update_from_response({"x-ratelimit-limit-tokens": "50000"})

    assert limiter.capacity == 37_500


def test_unconfigured_limit_learns_quota_from_rate_limit_error():
    limiter = GroqTokenRateLimiter(
        limit=None,
        utilization=0.75,
    )
    limiter.update_from_response(
        {},
        '{"error":{"message":"TPM: Limit 8000, Used 7000, Requested 2000"}}',
    )

    assert limiter.capacity == 6000


def test_configured_limit_is_an_explicit_override():
    limiter = GroqTokenRateLimiter(limit=8000, utilization=0.75)
    limiter.update_from_response({"x-ratelimit-limit-tokens": "50000"})

    assert limiter.capacity == 6000


def test_distributed_limiter_reserves_tokens_with_redis(monkeypatch):
    limiter = GroqTokenRateLimiter(
        limit=100,
        utilization=0.8,
        redis_url="redis://localhost:6379/0",
        model="test-model",
    )
    client = MagicMock()
    client.eval.side_effect = [250, 0]
    monkeypatch.setattr(limiter, "_get_redis_client", lambda: client)
    sleeps: list[float] = []
    monkeypatch.setattr("app.llm.rate_limit.time.sleep", sleeps.append)
    monkeypatch.setattr("app.llm.rate_limit.random.uniform", lambda lower, upper: 0)

    limiter.wait_for_capacity(25)

    assert len(client.eval.call_args_list) == 2
    assert client.eval.call_args.args[2] == "groq:tpm:test-model"
    assert sleeps == [pytest.approx(0.25)]


def test_distributed_limiter_reconciles_reservation_with_redis(monkeypatch):
    limiter = GroqTokenRateLimiter(
        limit=100,
        utilization=0.8,
        redis_url="redis://localhost:6379/0",
        model="test-model",
    )
    client = MagicMock()
    client.eval.return_value = 0
    monkeypatch.setattr(limiter, "_get_redis_client", lambda: client)

    reservation_id = limiter.wait_for_capacity(25)
    limiter.reconcile_reservation(reservation_id, 25, 10)

    assert client.eval.call_count == 2
    reconcile_call = client.eval.call_args
    assert reconcile_call.args[2] == "groq:tpm:test-model"
    assert reconcile_call.args[3] == f"{reservation_id}|25"
    assert reconcile_call.args[4] == f"{reservation_id}|10"
    assert reconcile_call.args[5] == 10


def test_distributed_limiter_rejects_request_larger_than_detected_quota(monkeypatch):
    limiter = GroqTokenRateLimiter(
        limit=None,
        utilization=0.75,
        redis_url="redis://localhost:6379/0",
        model="test-model",
    )
    client = MagicMock()
    client.eval.return_value = -1
    monkeypatch.setattr(limiter, "_get_redis_client", lambda: client)

    with pytest.raises(AIProviderError, match="exceeds the active TPM budget"):
        limiter.wait_for_capacity(7000)


def test_distributed_limiter_falls_back_to_local_when_redis_is_unavailable(
    monkeypatch, caplog
):
    limiter = GroqTokenRateLimiter(
        limit=100,
        utilization=0.8,
        redis_url="redis://localhost:6379/0",
        model="test-model",
    )
    client = MagicMock()
    client.eval.side_effect = RedisError("unavailable")
    monkeypatch.setattr(limiter, "_get_redis_client", lambda: client)

    limiter.wait_for_capacity(25)

    assert not limiter._redis_url
    assert "no longer coordinated" in caplog.text
