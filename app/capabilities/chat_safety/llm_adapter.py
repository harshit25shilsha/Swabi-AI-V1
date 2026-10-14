"""Chat Safety LLM adapter.

Routes Groq classifications through the shared ``LLMGateway`` (which uses
httpx under the hood) and Gemini classifications through the
``google-genai`` SDK. Preserves GDv1's provider ladder (Groq → Gemini),
per-provider circuit breakers, JSON output in both providers, and returns
the raw response text for the moderator to parse.

The moderator holds the circuit-breaker registry; the adapter does not
know about circuits. The adapter only knows how to call a provider by
name.
"""

from __future__ import annotations

import concurrent.futures

from google import genai
from google.genai import types as genai_types

from app.core.config import settings
from app.core.logging import get_logger
from app.llm.gateway import LLMGateway

log = get_logger(__name__)


def _classify_gemini(self, system: str, user: str) -> str:
    client = self._get_gemini_client()

    def _call():
        return client.models.generate_content(
            model=settings.gemini_model,
            contents=user,
            config=genai_types.GenerateContentConfig(
                system_instruction=system,
                temperature=settings.llm_temperature,
                max_output_tokens=settings.llm_max_tokens,
                response_mime_type="application/json",
            ),
        )

    # google-genai retries 503 internally, which can push past 40s.
    # Enforce our own wall-clock cap so the fail-open path stays bounded.
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(_call)
        try:
            response = future.result(timeout=settings.gemini_timeout_seconds)
        except concurrent.futures.TimeoutError as exc:
            raise TimeoutError(
                f"Gemini wall-clock timeout after {settings.gemini_timeout_seconds}s"
            ) from exc

    return response.text or "{}"


class ChatSafetyLLMAdapter:
    """Synchronous adapter over Groq (via gateway) + Gemini (direct)."""

    def __init__(self, gateway: LLMGateway | None = None) -> None:
        self.gateway = gateway or LLMGateway()
        self._gemini_client: genai.Client | None = None

    # -- Provider discovery

    def available_providers(self) -> list[str]:
        """Return provider names in the order they should be tried.

        Groq is always first. Gemini is included only if the fallback is
        enabled in settings and a Gemini API key is configured.
        """
        names: list[str] = ["groq"]
        if settings.llm_fallback_enabled and settings.gemini_api_key:
            names.append("gemini")
        return names

    # -- Dispatch

    def classify(self, provider_name: str, system: str, user: str) -> str:
        """Dispatch a classification to a specific provider by name.

        Returns the raw response text (expected to be JSON). The
        moderator parses and validates it.
        """
        if provider_name == "groq":
            return self._classify_groq(system, user)
        if provider_name == "gemini":
            return self._classify_gemini(system, user)
        raise ValueError(f"unknown provider: {provider_name!r}")

    # -- Provider implementations

    def _classify_groq(self, system: str, user: str) -> str:
        response = self.gateway.generate_sync(
            user,
            system=system,
            json_mode=True,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )
        return response.content

    def _classify_gemini(self, system: str, user: str) -> str:
        client = self._get_gemini_client()
        response = client.models.generate_content(
            model=settings.gemini_model,
            contents=user,
            config=genai_types.GenerateContentConfig(
                system_instruction=system,
                temperature=settings.llm_temperature,
                max_output_tokens=settings.llm_max_tokens,
                response_mime_type="application/json",
            ),
        )
        return response.text or "{}"

    def _get_gemini_client(self) -> genai.Client:
        if self._gemini_client is None:
            self._gemini_client = genai.Client(api_key=settings.gemini_api_key)
        return self._gemini_client


# Module-level singleton. Constructed lazily on first use so importing
# this module does not require a gateway or Gemini credentials.
_adapter: ChatSafetyLLMAdapter | None = None


def get_adapter() -> ChatSafetyLLMAdapter:
    """Return the process-wide adapter instance, creating it on first use."""
    global _adapter
    if _adapter is None:
        _adapter = ChatSafetyLLMAdapter()
    return _adapter
