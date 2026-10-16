from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from pydantic import BaseModel

from app.core.exceptions import AIValidationError
from app.llm.gateway import LLMGateway
from app.llm.groq import GroqProvider
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
    fake_resp.text = ""
    fake_resp.json.return_value = {
        "choices": [{"message": {"content": '{"decision":"ALLOW"}'}}],
        "usage": {},
        "model": "test-model",
    }

    with patch("app.llm.groq._get_sync_client") as get_client:
        client = MagicMock()
        client.post.return_value = fake_resp
        get_client.return_value = client

        provider.generate_sync("hi", json_mode=True)

    _, kwargs = client.post.call_args
    assert kwargs["timeout"] == 17


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
