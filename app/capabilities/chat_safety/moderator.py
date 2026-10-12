"""Chat Safety LLM classifier — provider ladder, circuits, consensus.

Orchestrates the provider ladder (Groq → Gemini), per-provider circuit
breakers, and the consensus loop. Public entry point:
``moderate_with_llm(message) -> (LLMResult, provider_name)``.

NOTE: This module is temporary. Step 9 replaces it with an adapter over
the shared LLMGateway, and deletes it.
"""

from __future__ import annotations

import json
import logging
import re

from pydantic import ValidationError

from app.capabilities.chat_safety.prompts import PROMPT_VERSION, SYSTEM_PROMPT
from app.capabilities.chat_safety.providers import get_providers
from app.capabilities.chat_safety.schemas import Action, LLMResult
from app.core.config import settings
from app.observability import counters
from app.observability.circuit_breaker import CircuitBreaker

__all__ = [
    "SYSTEM_PROMPT",
    "PROMPT_VERSION",
    "get_all_circuits",
    "moderate_with_llm",
]

logger = logging.getLogger(__name__)


_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)

# One circuit per provider. Exposed to /metrics for operational visibility.
_provider_circuits: dict[str, CircuitBreaker] = {}


def _get_circuit(provider_name: str) -> CircuitBreaker:
    if provider_name not in _provider_circuits:
        _provider_circuits[provider_name] = CircuitBreaker(
            failure_threshold=settings.LLM_CIRCUIT_FAILURE_THRESHOLD,
            open_seconds=settings.LLM_CIRCUIT_OPEN_SECONDS,
        )
    return _provider_circuits[provider_name]


def get_all_circuits() -> dict[str, CircuitBreaker]:
    """Return the provider→circuit map. Used by /metrics."""
    return _provider_circuits


def _parse_llm_json(raw: str) -> dict:
    raw = (raw or "").strip()
    m = _FENCE_RE.match(raw)
    if m:
        raw = m.group(1).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end > start:
            return json.loads(raw[start : end + 1])
        raise


def _coerce_result(data: dict) -> LLMResult:
    if data.get("category") in ("", "null", "ALLOW"):
        data["category"] = None
    try:
        return LLMResult(**data)
    except ValidationError:
        logger.warning("Invalid LLM category %r; coercing", data.get("category"))
        if data.get("decision") == "BLOCK":
            return LLMResult(
                decision="BLOCK",
                category="OTHER_PROHIBITED_CONTENT",
                confidence=float(data.get("confidence", 0.5)),
            )
        return LLMResult(decision="ALLOW", category=None, confidence=0.5)


def _classify_with_fallback(message: str) -> tuple[LLMResult, str]:
    """Walk the provider ladder. Returns (result, provider_name).

    Raises if every provider fails or is circuit-open.
    """
    providers = get_providers()
    if not providers:
        raise RuntimeError("no LLM providers configured")

    last_exc: Exception | None = None
    for provider in providers:
        circuit = _get_circuit(provider.name)

        if not circuit.allow_call():
            logger.info(
                "provider_skipped_circuit_open",
                extra={"provider": provider.name},
            )
            continue

        counters.increment("llm_calls_total")
        counters.increment(f"llm_calls_by_provider:{provider.name}")
        try:
            raw = provider.classify(SYSTEM_PROMPT, message)
            data = _parse_llm_json(raw)
            result = _coerce_result(data)
            circuit.record_success()
            return result, provider.name
        except Exception as e:
            circuit.record_failure()
            counters.increment("llm_failures_total")
            counters.increment(f"llm_failures_by_provider:{provider.name}")
            logger.warning(
                "provider_failed",
                extra={"provider": provider.name, "error": str(e)},
            )
            last_exc = e
            continue

    assert last_exc is not None  # noqa: S101  (fixed in step 9)
    raise last_exc


def moderate_with_llm(message: str) -> tuple[LLMResult, str]:
    """Consensus retry across the provider ladder.

    Returns (result, provider_name). BLOCK short-circuits.
    """
    attempts = max(1, settings.LLM_CONSENSUS_ATTEMPTS)
    allows: list[tuple[LLMResult, str]] = []
    last_exc: Exception | None = None

    for i in range(attempts):
        try:
            result, provider = _classify_with_fallback(message)
        except Exception as e:
            last_exc = e
            logger.warning("consensus attempt %d/%d failed", i + 1, attempts)
            continue

        if result.decision == Action.BLOCK:
            return result, provider
        allows.append((result, provider))

    if not allows:
        assert last_exc is not None  # noqa: S101  (fixed in step 9)
        raise last_exc

    # highest-confidence ALLOW, keep its provider
    best_result, best_provider = max(allows, key=lambda rp: rp[0].confidence)
    return best_result, best_provider
