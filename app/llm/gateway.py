"""Provider-independent LLM gateway."""

from __future__ import annotations

import json
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from app.core.exceptions import AIValidationError
from app.core.logging import get_logger
from app.llm.groq import GroqProvider
from app.llm.schemas import LLMResponse
from app.observability.logging import log_llm_call

log = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMGateway:
    """Single entry point for all LLM usage."""

    def __init__(self, provider: GroqProvider | None = None) -> None:
        self.provider = provider or GroqProvider()

    async def generate(self, prompt: str, **kwargs: object) -> LLMResponse:
        response = await self.provider.generate(prompt, **kwargs)
        log_llm_call(
            model=response.model,
            latency_ms=response.latency_ms,
            usage=response.usage.model_dump(),
        )
        return response

    async def generate_structured(
        self,
        prompt: str,
        response_model: type[T],
        **kwargs: object,
    ) -> T:
        """Ask the LLM for JSON and validate it against a Pydantic model.

        Retries once if the JSON does not validate.
        """
        schema = json.dumps(response_model.model_json_schema(), indent=2)
        structured_prompt = (
            f"{prompt}\n\n"
            "Respond ONLY with valid JSON that matches this JSON schema. "
            "Do not include markdown fences or commentary.\n\n"
            f"JSON schema:\n{schema}"
        )

        last_error: Exception | None = None
        for attempt in range(2):
            response = await self.generate(
                structured_prompt,
                json_mode=True,
                **kwargs,
            )
            try:
                return response_model.model_validate_json(response.content)
            except ValidationError as exc:
                last_error = exc
                log.warning(
                    "Structured output validation failed (attempt %d): %s",
                    attempt + 1,
                    exc,
                )
        raise AIValidationError(f"LLM did not return valid structured output: {last_error}")
