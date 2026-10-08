"""Shared pytest fixtures."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("SWABI_API_KEY", "test-swabi-key")
os.environ.setdefault("REDIS_URL", "")  # disable Redis by default in tests


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"
