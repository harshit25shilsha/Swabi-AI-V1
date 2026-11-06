"""Middleware-order verification tests.

These assert observable effects of the intended order:
    request_id -> auth_context -> observability -> route
"""

from __future__ import annotations

import logging

from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app

client = TestClient(app)
AUTH = {"Authorization": f"Bearer {settings.guardrail_api_token}"}


def test_client_supplied_request_id_is_preserved():
    """Outer request_id middleware preserves incoming X-Request-ID."""
    r = client.get("/health", headers={"X-Request-ID": "custom-id-abc"})
    assert r.status_code == 200
    assert r.headers["X-Request-ID"] == "custom-id-abc"


def test_request_id_generated_with_gr_prefix():
    """Default generated IDs use the gr_ prefix, matching GDv1 behavior."""
    r = client.get("/health")
    assert r.status_code == 200
    assert r.headers["X-Request-ID"].startswith("gr_")


def test_observability_log_event_carries_outer_request_id(caplog):
    """Observability middleware reads request_id from the outer contextvar."""
    with caplog.at_level(logging.INFO, logger="guardrail.request"):
        client.post(
            "/api/v1/validate-message",
            headers={**AUTH, "X-Request-ID": "observed-rid-xyz"},
            json={"message": "Call me at 9876543210"},
        )

    moderation_records = [r for r in caplog.records if r.message == "moderation"]
    assert moderation_records, "no 'moderation' log event was emitted"
    assert getattr(moderation_records[-1], "request_id", None) == "observed-rid-xyz"


def test_header_not_overwritten_by_inner_middleware():
    """Even after the observability middleware runs, outer value wins."""
    r = client.get("/health", headers={"X-Request-ID": "outer-wins"})
    assert r.headers["X-Request-ID"] == "outer-wins"
