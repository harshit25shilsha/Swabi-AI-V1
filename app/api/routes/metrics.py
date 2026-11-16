"""Prometheus metrics endpoint.

Unauthenticated by design — bind behind the firewall or reverse proxy.
"""

from __future__ import annotations

from fastapi import APIRouter, Response

from app.capabilities.chat_safety.moderator import get_all_circuits
from app.observability import counters

router = APIRouter(tags=["observability"])

# Providers we always emit metrics for, even before they are used.
# Keeps Prometheus time series stable across restarts.
_KNOWN_PROVIDERS = ("groq", "gemini")


@router.get("/metrics")
def metrics() -> Response:
    snap = counters.snapshot()
    lines: list[str] = []

    # --- Request counters
    lines.extend(
        [
            "# HELP guardrail_requests_total Total moderation requests",
            "# TYPE guardrail_requests_total counter",
            f"guardrail_requests_total {snap['requests_total']}",
            "",
            "# HELP guardrail_allows_total Allowed messages",
            "# TYPE guardrail_allows_total counter",
            f"guardrail_allows_total {snap['allows_total']}",
            "",
            "# HELP guardrail_blocks_total Blocked messages",
            "# TYPE guardrail_blocks_total counter",
            f"guardrail_blocks_total {snap['blocks_total']}",
            "",
        ]
    )

    # --- Blocks by category
    lines.extend(
        [
            "# HELP guardrail_blocks_by_category Blocks by category",
            "# TYPE guardrail_blocks_by_category counter",
        ]
    )
    for cat, count in sorted(snap["blocks_by_category"].items()):
        lines.append(f'guardrail_blocks_by_category{{category="{cat}"}} {count}')
    lines.append("")

    # --- Sources
    lines.extend(
        [
            "# HELP guardrail_sources Decisions by source",
            "# TYPE guardrail_sources counter",
        ]
    )
    for src, count in sorted(snap["sources"].items()):
        lines.append(f'guardrail_sources{{source="{src}"}} {count}')
    lines.append("")

    # --- LLM calls by provider
    lines.extend(
        [
            "# HELP guardrail_llm_calls_by_provider LLM calls per provider",
            "# TYPE guardrail_llm_calls_by_provider counter",
        ]
    )
    calls_by_provider = snap.get("llm_calls_by_provider", {})
    for provider in sorted(set(_KNOWN_PROVIDERS) | set(calls_by_provider)):
        count = calls_by_provider.get(provider, 0)
        lines.append(f'guardrail_llm_calls_by_provider{{provider="{provider}"}} {count}')
    lines.append("")

    # --- LLM failures by provider
    lines.extend(
        [
            "# HELP guardrail_llm_failures_by_provider LLM failures per provider",
            "# TYPE guardrail_llm_failures_by_provider counter",
        ]
    )
    failures_by_provider = snap.get("llm_failures_by_provider", {})
    for provider in sorted(set(_KNOWN_PROVIDERS) | set(failures_by_provider)):
        count = failures_by_provider.get(provider, 0)
        lines.append(f'guardrail_llm_failures_by_provider{{provider="{provider}"}} {count}')
    lines.append("")

    # --- Aggregate LLM counters
    lines.extend(
        [
            "# HELP guardrail_llm_calls_total Total LLM calls across all providers",
            "# TYPE guardrail_llm_calls_total counter",
            f"guardrail_llm_calls_total {snap['llm_calls_total']}",
            "",
            "# HELP guardrail_llm_failures_total Total LLM failures across all providers",
            "# TYPE guardrail_llm_failures_total counter",
            f"guardrail_llm_failures_total {snap['llm_failures_total']}",
            "",
            "# HELP guardrail_fail_open_total Fail-open responses",
            "# TYPE guardrail_fail_open_total counter",
            f"guardrail_fail_open_total {snap['fail_open_total']}",
            "",
            "# HELP guardrail_latency_p95_ms Request latency p95 (ms)",
            "# TYPE guardrail_latency_p95_ms gauge",
            f"guardrail_latency_p95_ms {snap['latency_p95_ms']}",
            "",
        ]
    )

    # --- Circuit breaker state, per provider
    circuits = get_all_circuits()

    lines.extend(
        [
            "# HELP guardrail_circuit_state Circuit breaker state (1 if active)",
            "# TYPE guardrail_circuit_state gauge",
        ]
    )
    if circuits:
        for name, cb in circuits.items():
            c = cb.metrics()
            lines.append(
                f'guardrail_circuit_state{{provider="{name}",state="closed"}} '
                f"{1 if c['state'] == 'closed' else 0}"
            )
            lines.append(
                f'guardrail_circuit_state{{provider="{name}",state="open"}} '
                f"{1 if c['state'] == 'open' else 0}"
            )
            lines.append(
                f'guardrail_circuit_state{{provider="{name}",state="half_open"}} '
                f"{1 if c['state'] == 'half_open' else 0}"
            )
    else:
        for name in _KNOWN_PROVIDERS:
            lines.append(f'guardrail_circuit_state{{provider="{name}",state="closed"}} 1')
            lines.append(f'guardrail_circuit_state{{provider="{name}",state="open"}} 0')
            lines.append(f'guardrail_circuit_state{{provider="{name}",state="half_open"}} 0')
    lines.append("")

    lines.extend(
        [
            "# HELP guardrail_circuit_trips_total Times a circuit opened",
            "# TYPE guardrail_circuit_trips_total counter",
        ]
    )
    if circuits:
        for name, cb in circuits.items():
            c = cb.metrics()
            lines.append(f'guardrail_circuit_trips_total{{provider="{name}"}} {c["trips_total"]}')
    else:
        for name in _KNOWN_PROVIDERS:
            lines.append(f'guardrail_circuit_trips_total{{provider="{name}"}} 0')
    lines.append("")

    lines.extend(
        [
            "# HELP guardrail_circuit_rejections_total Requests short-circuited",
            "# TYPE guardrail_circuit_rejections_total counter",
        ]
    )
    if circuits:
        for name, cb in circuits.items():
            c = cb.metrics()
            lines.append(
                f'guardrail_circuit_rejections_total{{provider="{name}"}} {c["rejections_total"]}'
            )
    else:
        for name in _KNOWN_PROVIDERS:
            lines.append(f'guardrail_circuit_rejections_total{{provider="{name}"}} 0')
    lines.append("")

    return Response(
        "\n".join(lines),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
