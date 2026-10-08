"""Groq provider with bounded retries."""

from __future__ import annotations

import asyncio
import time

import httpx

from app.core.config import settings
from app.core.exceptions import AIProviderError, AITimeoutError
from app.core.logging import get_logger
from app.llm.schemas import LLMResponse, TokenUsage


log = get_logger(__name__)

GROQ_CHAT_URL = settings.groq_chat_url

# HTTP statuses worth retrying
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class GroqProvider:
    """Minimal Groq chat-completions client."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout: int | None = None,
        max_retries: int | None = None,
    ) -> None:
        self.api_key = api_key or settings.groq_api_key
        self.model = model or settings.llm_model
        self.timeout = timeout or settings.llm_timeout
        self.max_retries = max_retries if max_retries is not None else settings.llm_max_retries
        if not self.api_key:
            log.warning("GROQ_API_KEY is not configured")

    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.2,
        json_mode: bool = False,
        **extra: object,
    ) -> LLMResponse:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, object] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        payload.update(extra)

        attempt = 0
        delay = 0.5
        last_exc: Exception | None = None

        while attempt <= self.max_retries:
            attempt += 1
            start = time.perf_counter()
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    resp = await client.post(
                        GROQ_CHAT_URL,
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                        },
                        json=payload,
                    )
            except httpx.TimeoutException as exc:
                last_exc = AITimeoutError(f"Groq timeout after {self.timeout}s")
                log.warning("Groq timeout (attempt %d/%d)", attempt, self.max_retries + 1)
                if attempt > self.max_retries:
                    raise last_exc from exc
                await asyncio.sleep(delay)
                delay *= 2
                continue
            except httpx.HTTPError as exc:
                last_exc = AIProviderError(f"Groq network error: {exc}")
                log.warning("Groq network error (attempt %d): %s", attempt, exc)
                if attempt > self.max_retries:
                    raise last_exc from exc
                await asyncio.sleep(delay)
                delay *= 2
                continue

            if resp.status_code in RETRYABLE_STATUS and attempt <= self.max_retries:
                log.warning("Groq retryable status %s (attempt %d)", resp.status_code, attempt)
                await asyncio.sleep(delay)
                delay *= 2
                continue

            if resp.status_code >= 400:
                raise AIProviderError(f"Groq HTTP {resp.status_code}: {resp.text[:300]}")

            data = resp.json()
            latency_ms = (time.perf_counter() - start) * 1000
            try:
                choice = data["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as exc:
                raise AIProviderError(f"Groq malformed response: {exc}") from exc

            usage = data.get("usage") or {}
            return LLMResponse(
                content=choice,
                model=data.get("model", self.model),
                usage=TokenUsage(
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    total_tokens=usage.get("total_tokens", 0),
                ),
                latency_ms=latency_ms,
                metadata={"provider": "groq", "attempt": attempt},
            )

        # Should not be reached
        raise AIProviderError(f"Groq failed after retries: {last_exc}")
