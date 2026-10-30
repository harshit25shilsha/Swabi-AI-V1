"""Groq provider with bounded retries.

Two call paths:

* ``generate()`` — async, opens a fresh ``httpx.AsyncClient`` per attempt.
  Intended for future async capabilities (Phase 1+). Does not pool
  connections yet; a follow-up will share a long-lived ``AsyncClient``.
* ``generate_sync()`` — synchronous mirror, uses a module-level shared
  ``httpx.Client`` so TCP + TLS connections are pooled and kept alive.
  Used by Chat Safety, whose call chain is synchronous.

Both paths share the same payload construction, retry policy, and
response parsing. The only intentional differences are the HTTP client
type and the sleep primitive.
"""

from __future__ import annotations

import asyncio
import atexit
import email.utils
import random
import re
import threading
import time
from datetime import UTC, datetime

import httpx

from app.core.config import settings
from app.core.exceptions import AIProviderError, AITimeoutError
from app.core.logging import get_logger
from app.llm.rate_limit import groq_tpm_limiter
from app.llm.schemas import LLMResponse, TokenUsage

log = get_logger(__name__)

GROQ_CHAT_URL = settings.groq_chat_url

# HTTP statuses worth retrying
RETRYABLE_STATUS = {429, 500, 502, 503, 504}

# Initial retry delay for network errors and non-rate-limit transient errors.
_INITIAL_RETRY_DELAY = 0.2
_RETRY_JITTER_MAX_SECONDS = 0.5

_RETRY_AFTER_SECONDS = re.compile(
    r"try again in\s+(\d+(?:\.\d+)?)\s*(milliseconds?|ms|seconds?|secs?|s)\b",
    re.IGNORECASE,
)


# Module-level shared sync client
#
# Reused across all sync calls so TCP + TLS connections are pooled and kept
# alive. ``httpx.Client`` is documented as thread-safe; FastAPI's thread pool
# can call it concurrently.
#
# Created lazily on first use so importing this module does not open a
# socket. A shutdown hook closes it cleanly on interpreter exit; FastAPI's
# lifespan also closes it explicitly.

_sync_client: httpx.Client | None = None
_sync_client_lock = threading.Lock()
_request_semaphore = threading.BoundedSemaphore(
    settings.groq_max_concurrent_requests
)


def _retry_delay(response: httpx.Response, fallback: float) -> float:
    """Use Groq's advertised retry time when available, with small jitter."""
    advised: float | None = None
    retry_after = response.headers.get("Retry-After")
    if retry_after:
        try:
            advised = max(0.0, float(retry_after))
        except ValueError:
            try:
                retry_at = email.utils.parsedate_to_datetime(retry_after)
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=UTC)
                advised = max(
                    0.0,
                    (retry_at - datetime.now(UTC)).total_seconds(),
                )
            except (TypeError, ValueError, OverflowError):
                advised = None

    if advised is None:
        match = _RETRY_AFTER_SECONDS.search(response.text)
        if match:
            advised = float(match.group(1))
            if match.group(2).lower() in {"ms", "millisecond", "milliseconds"}:
                advised /= 1000

    delay = max(fallback, advised or 0.0)
    return delay + random.uniform(0.0, min(delay * 0.1, _RETRY_JITTER_MAX_SECONDS))


def _estimate_request_tokens(payload: dict[str, object]) -> int:
    """Conservatively estimate prompt plus completion tokens without a tokenizer."""
    messages = payload.get("messages", [])
    text_chars = sum(
        len(str(message.get("content", "")).encode("utf-8"))
        for message in messages
        if isinstance(message, dict)
    )
    prompt_estimate = (text_chars + 3) // 4 + 8 * len(messages)
    completion_budget = payload.get(
        "max_completion_tokens",
        payload.get("max_tokens", settings.llm_max_tokens),
    )
    try:
        completion_estimate = max(0, int(completion_budget))
    except (TypeError, ValueError):
        completion_estimate = settings.llm_max_tokens
    return prompt_estimate + completion_estimate


async def _acquire_request_slot() -> None:
    while not _request_semaphore.acquire(blocking=False):
        await asyncio.sleep(0.05)


def _get_sync_client() -> httpx.Client:
    global _sync_client
    if _sync_client is None:
        with _sync_client_lock:
            if _sync_client is None:
                _sync_client = httpx.Client(
                    timeout=settings.llm_timeout,
                    limits=httpx.Limits(
                        max_keepalive_connections=20,
                        max_connections=50,
                        keepalive_expiry=30.0,
                    ),
                )
    return _sync_client


def close_sync_client() -> None:
    """Close the shared sync client.

    Called on FastAPI shutdown (``app.main.lifespan``) and again via
    ``atexit`` as a safety net. Idempotent.
    """
    global _sync_client
    with _sync_client_lock:
        if _sync_client is not None:
            _sync_client.close()
            _sync_client = None


atexit.register(close_sync_client)


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

    # Internal — shared payload construction

    def _build_payload(
        self,
        prompt: str,
        system: str | None,
        temperature: float,
        json_mode: bool,
        extra: dict[str, object],
    ) -> dict[str, object]:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, object] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if "max_tokens" not in extra and "max_completion_tokens" not in extra:
            payload["max_tokens"] = settings.llm_max_tokens
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        payload.update(extra)
        return payload

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _is_json_validate_failure(status_code: int, body: str, json_mode: bool) -> bool:
        """Groq returns 400 / json_validate_failed when a reasoning model
        burns its token budget on internal reasoning and never emits the
        JSON. Retrying once without response_format usually succeeds."""
        return status_code == 400 and "json_validate_failed" in body and json_mode

    @staticmethod
    def _parse_response(data: dict, model_fallback: str) -> LLMResponse:
        try:
            completion = data["choices"][0]
            choice = completion["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise AIProviderError(f"Groq malformed response: {exc}") from exc

        usage = data.get("usage") or {}
        return LLMResponse(
            content=choice,
            model=data.get("model", model_fallback),
            usage=TokenUsage(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
            ),
            latency_ms=0.0,  # caller fills this in
            metadata={
                "provider": "groq",
                "finish_reason": completion.get("finish_reason"),
            },
        )

    # Async

    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.2,
        json_mode: bool = False,
        **extra: object,
    ) -> LLMResponse:
        payload = self._build_payload(prompt, system, temperature, json_mode, extra)

        attempt = 0
        delay = _INITIAL_RETRY_DELAY
        last_exc: Exception | None = None
        estimated_tokens = _estimate_request_tokens(payload)

        while attempt <= self.max_retries:
            attempt += 1
            start = time.perf_counter()
            try:
                await _acquire_request_slot()
                try:
                    reservation_id = await asyncio.to_thread(
                        groq_tpm_limiter.wait_for_capacity,
                        estimated_tokens,
                    )
                    async with httpx.AsyncClient(timeout=self.timeout) as client:
                        resp = await client.post(
                            GROQ_CHAT_URL,
                            headers=self._headers(),
                            json=payload,
                        )
                finally:
                    _request_semaphore.release()
            except httpx.TimeoutException as exc:
                last_exc = AITimeoutError(f"Groq timeout after {self.timeout}s")
                log.warning(
                    "Groq timeout (attempt %d/%d)",
                    attempt,
                    self.max_retries + 1,
                )
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

            groq_tpm_limiter.update_from_response(
                resp.headers,
                resp.text if resp.status_code >= 400 else "",
            )
            if resp.status_code in RETRYABLE_STATUS and attempt <= self.max_retries:
                log.warning(
                    "Groq retryable status %s (attempt %d)",
                    resp.status_code,
                    attempt,
                )
                await asyncio.sleep(_retry_delay(resp, delay))
                delay *= 2
                continue

            if resp.status_code >= 400:
                body = resp.text
                if (
                    self._is_json_validate_failure(resp.status_code, body, json_mode)
                    and attempt <= self.max_retries
                ):
                    log.warning(
                        "Groq json_validate_failed (async); retrying without response_format"
                    )
                    payload.pop("response_format", None)
                    json_mode = False
                    await asyncio.sleep(0.2)
                    continue
                raise AIProviderError(f"Groq HTTP {resp.status_code}: {body[:300]}")

            data = resp.json()
            latency_ms = (time.perf_counter() - start) * 1000
            result = self._parse_response(data, self.model)
            result.latency_ms = latency_ms
            result.metadata = {
                "provider": "groq",
                "attempt": attempt,
                "mode": "async",
                "finish_reason": result.metadata.get("finish_reason"),
            }
            if result.usage.total_tokens > 0:
                groq_tpm_limiter.reconcile_reservation(
                    reservation_id,
                    estimated_tokens,
                    result.usage.total_tokens,
                )
            return result

        # Unreachable in practice: the loop either returns or raises.
        raise AIProviderError(f"Groq failed after retries: {last_exc}")

    # Sync

    def generate_sync(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.2,
        json_mode: bool = False,
        **extra: object,
    ) -> LLMResponse:
        """Synchronous mirror of ``generate``.

        Uses the module-level shared ``httpx.Client`` so connections are
        pooled. Same payload, same retry policy, same response parsing.
        """
        payload = self._build_payload(prompt, system, temperature, json_mode, extra)

        attempt = 0
        delay = _INITIAL_RETRY_DELAY
        last_exc: Exception | None = None
        estimated_tokens = _estimate_request_tokens(payload)

        while attempt <= self.max_retries:
            attempt += 1
            start = time.perf_counter()
            try:
                client = _get_sync_client()
                with _request_semaphore:
                    reservation_id = groq_tpm_limiter.wait_for_capacity(
                        estimated_tokens
                    )
                    resp = client.post(
                        GROQ_CHAT_URL,
                        headers=self._headers(),
                        json=payload,
                        timeout=self.timeout,
                    )
            except httpx.TimeoutException as exc:
                last_exc = AITimeoutError(f"Groq timeout after {self.timeout}s")
                log.warning(
                    "Groq sync timeout (attempt %d/%d)",
                    attempt,
                    self.max_retries + 1,
                )
                if attempt > self.max_retries:
                    raise last_exc from exc
                time.sleep(delay)
                delay *= 2
                continue
            except httpx.HTTPError as exc:
                last_exc = AIProviderError(f"Groq network error: {exc}")
                log.warning("Groq sync network error (attempt %d): %s", attempt, exc)
                if attempt > self.max_retries:
                    raise last_exc from exc
                time.sleep(delay)
                delay *= 2
                continue

            groq_tpm_limiter.update_from_response(
                resp.headers,
                resp.text if resp.status_code >= 400 else "",
            )
            if resp.status_code in RETRYABLE_STATUS and attempt <= self.max_retries:
                log.warning(
                    "Groq sync retryable status %s (attempt %d)",
                    resp.status_code,
                    attempt,
                )
                time.sleep(_retry_delay(resp, delay))
                delay *= 2
                continue

            if resp.status_code >= 400:
                body = resp.text
                if (
                    self._is_json_validate_failure(resp.status_code, body, json_mode)
                    and attempt <= self.max_retries
                ):
                    log.warning("Groq json_validate_failed; retrying without response_format")
                    payload.pop("response_format", None)
                    json_mode = False
                    time.sleep(0.2)
                    continue
                raise AIProviderError(f"Groq HTTP {resp.status_code}: {body[:300]}")

            data = resp.json()
            latency_ms = (time.perf_counter() - start) * 1000
            result = self._parse_response(data, self.model)
            result.latency_ms = latency_ms
            result.metadata = {
                "provider": "groq",
                "attempt": attempt,
                "mode": "sync",
                "finish_reason": result.metadata.get("finish_reason"),
            }
            if result.usage.total_tokens > 0:
                groq_tpm_limiter.reconcile_reservation(
                    reservation_id,
                    estimated_tokens,
                    result.usage.total_tokens,
                )
            return result

        raise AIProviderError(f"Groq sync failed after retries: {last_exc}")
