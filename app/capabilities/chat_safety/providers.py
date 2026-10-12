"""Unified interface over multiple LLM providers.

Each provider exposes:
  - name: short identifier
  - is_available(): whether credentials are configured
  - classify(system_prompt, user_message) -> raw JSON string

NOTE: This module is temporary. Step 9 replaces it with an adapter over
the shared LLMGateway, and deletes it.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from google import genai
from google.genai import types as genai_types
from groq import Groq

from app.core.config import settings

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    name: str

    @abstractmethod
    def is_available(self) -> bool: ...

    @abstractmethod
    def classify(self, system_prompt: str, user_message: str) -> str:
        """Return the raw response text (expected to be JSON)."""
        ...


class GroqProvider(LLMProvider):
    name = "groq"

    def __init__(self):
        self._client = None

    def is_available(self) -> bool:
        return bool(settings.GROQ_API_KEY)

    def _get_client(self) -> Groq:
        if self._client is None:
            self._client = Groq(
                api_key=settings.GROQ_API_KEY,
                timeout=settings.GROQ_TIMEOUT_SECONDS,
            )
        return self._client

    def classify(self, system_prompt: str, user_message: str) -> str:
        client = self._get_client()
        completion = client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            response_format={"type": "json_object"},
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
        )
        return completion.choices[0].message.content or "{}"


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self):
        self._client = None

    def is_available(self) -> bool:
        return bool(settings.GEMINI_API_KEY)

    def _get_client(self) -> genai.Client:
        if self._client is None:
            self._client = genai.Client(api_key=settings.GEMINI_API_KEY)
        return self._client

    def classify(self, system_prompt: str, user_message: str) -> str:
        client = self._get_client()
        response = client.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=user_message,
            config=genai_types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=settings.LLM_TEMPERATURE,
                max_output_tokens=settings.LLM_MAX_TOKENS,
                response_mime_type="application/json",
            ),
        )
        return response.text or "{}"


def get_providers() -> list[LLMProvider]:
    """Return the ordered list of providers to try."""
    providers: list[LLMProvider] = [GroqProvider()]
    if settings.LLM_FALLBACK_ENABLED:
        providers.append(GeminiProvider())
    return [p for p in providers if p.is_available()]
