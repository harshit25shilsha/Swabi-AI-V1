from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from pydantic import BaseModel

from app.core.config import settings
from app.core.exceptions import AIValidationError
from app.llm.gateway import LLMGateway
from app.llm.groq import GroqProvider, _estimate_request_tokens, _retry_delay
from app.llm.schemas import LLMResponse, TokenUsage


class _Out(BaseModel):
    name: str
    age: int


def _response(content: str) -> LLMResponse:
    return LLMResponse(
        content=content,
        model="test-model",
        usage=TokenUsage(total_tokens=1),
        latency_ms=1.0,
    )


def test_sync_request_uses_provider_timeout():
    """Custom provider timeout must reach httpx per-request."""
    provider = GroqProvider(
        api_key="test-key",
        model="test-model",
        timeout=17,
        max_retries=0,
    )

    fake_resp = MagicMock(spec=httpx.Response)
    fake_resp.status_code = 200
    fake_resp.headers = {}
    fake_resp.text = ""
    fake_resp.json.return_value = {
        "choices": [{"message": {"content": '{"decision":"ALLOW"}'}}],
        "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
        "model": "test-model",
    }

    with patch("app.llm.groq._get_sync_client") as get_client:
        client = MagicMock()
        client.post.return_value = fake_resp
        get_client.return_value = client
        limiter = MagicMock()
        limiter.wait_for_capacity.return_value = "reservation"
        with patch("app.llm.groq.groq_tpm_limiter", limiter):
            provider.generate_sync("hi", json_mode=True)

    _, kwargs = client.post.call_args
    assert kwargs["timeout"] == 17
    assert kwargs["json"]["max_tokens"] == settings.llm_max_tokens
    limiter.reconcile_reservation.assert_called_once_with(
        "reservation",
        limiter.wait_for_capacity.call_args.args[0],
        4,
    )


def test_token_estimate_includes_completion_budget():
    estimated = _estimate_request_tokens(
        {
            "messages": [{"role": "user", "content": "abcd"}],
            "max_tokens": 20,
        }
    )

    assert estimated == 29


def test_groq_response_preserves_finish_reason():
    response = GroqProvider._parse_response(
        {
            "choices": [{
                "message": {"content": '{"name": "Swabi"'},
                "finish_reason": "length",
            }],
            "model": "test-model",
        },
        "fallback-model",
    )

    assert response.metadata["finish_reason"] == "length"


def test_retry_delay_honors_groq_wait_message():
    response = httpx.Response(
        429,
        text='{"error":{"message":"Please try again in 3.25s."}}',
    )

    with patch("app.llm.groq.random.uniform", return_value=0):
        assert _retry_delay(response, 0.2) == 3.25


def test_retry_delay_honors_retry_after_header():
    response = httpx.Response(429, headers={"Retry-After": "2"})

    with patch("app.llm.groq.random.uniform", return_value=0):
        assert _retry_delay(response, 0.2) == 2


def test_sync_retry_waits_for_provider_retry_after():
    provider = GroqProvider(
        api_key="test-key",
        model="test-model",
        max_retries=1,
    )
    rate_limiter = MagicMock()
    retry_response = httpx.Response(
        429,
        headers={"Retry-After": "1"},
        text="rate limit",
    )
    success_response = httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": "ok"}}],
            "model": "test-model",
        },
    )
    client = MagicMock()
    client.post.side_effect = [retry_response, success_response]
    sleeps: list[float] = []

    with (
        patch("app.llm.groq._get_sync_client", return_value=client),
        patch("app.llm.groq.groq_tpm_limiter", rate_limiter),
        patch("app.llm.groq.time.sleep", sleeps.append),
        patch("app.llm.groq.random.uniform", return_value=0),
    ):
        response = provider.generate_sync("hi")

    assert response.content == "ok"
    assert sleeps == [1]
    assert rate_limiter.wait_for_capacity.call_count == 2


def test_sync_requests_respect_concurrency_limit():
    active = 0
    max_active = 0
    lock = threading.Lock()

    def post(*args, **kwargs):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        try:
            time.sleep(0.02)
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "ok"}}],
                    "model": "test-model",
                },
            )
        finally:
            with lock:
                active -= 1

    provider = GroqProvider(
        api_key="test-key",
        model="test-model",
        max_retries=0,
    )
    with (
        patch("app.llm.groq._request_semaphore", threading.BoundedSemaphore(1)),
        patch("app.llm.groq._get_sync_client") as get_client,
        ThreadPoolExecutor(max_workers=2) as executor,
    ):
        client = MagicMock()
        client.post.side_effect = post
        get_client.return_value = client
        list(executor.map(provider.generate_sync, ("first", "second")))

    assert max_active == 1


@pytest.mark.asyncio
async def test_generate_returns_response():
    provider = AsyncMock()
    provider.generate.return_value = _response("hello")
    gw = LLMGateway(provider=provider)

    resp = await gw.generate("hi")
    assert resp.content == "hello"
    provider.generate.assert_awaited_once()


@pytest.mark.asyncio
async def test_generate_structured_ok():
    provider = AsyncMock()
    provider.generate.return_value = _response('{"name": "Swabi", "age": 1}')
    gw = LLMGateway(provider=provider)

    out = await gw.generate_structured("prompt", _Out)
    assert out.name == "Swabi"
    assert out.age == 1


@pytest.mark.asyncio
async def test_generate_structured_invalid_raises():
    provider = AsyncMock()
    provider.generate.return_value = _response("not json")
    gw = LLMGateway(provider=provider)

    with pytest.raises(AIValidationError):
        await gw.generate_structured("prompt", _Out)


def test_sync_structured_output_retries_truncation_with_more_tokens():
    provider = MagicMock()
    truncated = _response('{"name": "Swabi"')
    truncated.metadata["finish_reason"] = "length"
    provider.generate_sync.side_effect = [
        truncated,
        _response('{"name": "Swabi", "age": 1}'),
    ]
    gw = LLMGateway(provider=provider)

    output = gw.generate_structured_sync(
        "prompt",
        _Out,
        max_tokens=512,
    )

    assert output.name == "Swabi"
    assert provider.generate_sync.call_count == 2
    assert provider.generate_sync.call_args_list[0].kwargs["max_tokens"] == 512
    assert provider.generate_sync.call_args_list[1].kwargs["max_tokens"] == 1024


def test_sync_structured_output_keeps_budget_for_non_truncation_error():
    provider = MagicMock()
    provider.generate_sync.side_effect = [
        _response("not json"),
        _response('{"name": "Swabi", "age": 1}'),
    ]
    gw = LLMGateway(provider=provider)

    output = gw.generate_structured_sync(
        "prompt",
        _Out,
        max_tokens=512,
    )

    assert output.age == 1
    assert [
        call.kwargs["max_tokens"]
        for call in provider.generate_sync.call_args_list
    ] == [512, 512]
