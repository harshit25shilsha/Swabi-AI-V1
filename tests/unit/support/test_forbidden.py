"""Forbidden filter tests."""
from __future__ import annotations

import pytest

from app.capabilities.support.forbidden import (
    has_financial_promise,
    strip_forbidden_paths,
)


@pytest.mark.parametrize("answer", [
    "See /admin/escrows for details",
    "Go to /payments/verify",
    "The escrow is at /escrow/123",
    "Track via /devices/abc",
])
def test_internal_paths_detected(answer):
    clean, removed = strip_forbidden_paths(answer)
    assert removed
    assert "redacted" in clean
    assert "/admin/" not in clean
    assert "/payments/" not in clean
    assert "/escrow/" not in clean
    assert "/devices/" not in clean


@pytest.mark.parametrize("answer", [
    "The cancellation policy gives 100% refund.",
    "Customers get 75% back if they cancel 15 days early.",
    "Commission is 8% of base amount.",
])
def test_clean_answers_pass_unchanged(answer):
    clean, removed = strip_forbidden_paths(answer)
    assert removed == []
    assert clean == answer


@pytest.mark.parametrize("answer", [
    "You will get a refund of ₹5000.",
    "You'll be credited within 5 days.",
    "You will receive your money back.",
])
def test_financial_promises_flagged(answer):
    assert has_financial_promise(answer) is True


@pytest.mark.parametrize("answer", [
    "The policy is 100% refund for cancellations 30 days or more.",
    "Refunds are processed within 5-7 business days.",
    "Customers can request a refund from the booking page.",
])
def test_policy_descriptions_not_flagged(answer):
    assert has_financial_promise(answer) is False
