"""Integration tests for GET /metrics."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
from app.observability import counters

client = TestClient(app)
AUTH = {"Authorization": f"Bearer {settings.guardrail_api_token}"}


@pytest.fixture(autouse=True)
def _reset_counters():
    counters.reset()
    yield
    counters.reset()


def test_metrics_returns_prometheus_text():
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "text/plain" in r.headers["content-type"]
    assert "guardrail_requests_total" in r.text
    assert "guardrail_latency_p95_ms" in r.text
    assert "guardrail_circuit_state" in r.text


def test_deterministic_block_updates_counters():
    client.post(
        "/api/v1/validate-message",
        headers=AUTH,
        json={"message": "Call me at 9876543210"},
    )
    text = client.get("/metrics").text
    assert "guardrail_requests_total 1" in text
    assert "guardrail_blocks_total 1" in text
    assert 'guardrail_blocks_by_category{category="PHONE_NUMBER"} 1' in text
    assert 'guardrail_sources{source="deterministic"} 1' in text


def test_latency_p95_recorded():
    for _ in range(5):
        client.post(
            "/api/v1/validate-message",
            headers=AUTH,
            json={"message": "Call me at 9876543210"},
        )
    text = client.get("/metrics").text
    for line in text.splitlines():
        if line.startswith("guardrail_latency_p95_ms "):
            value = float(line.split()[1])
            assert value > 0
            break
    else:
        pytest.fail("guardrail_latency_p95_ms not found in metrics output")


def test_circuit_metrics_emitted_even_when_unused():
    text = client.get("/metrics").text
    # Both known providers should be present with closed state by default
    for provider in ("groq", "gemini"):
        assert f'provider="{provider}"' in text
