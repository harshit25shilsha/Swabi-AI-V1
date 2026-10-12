from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from app.core.exceptions import AIValidationError
from app.llm.gateway import LLMGateway
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
