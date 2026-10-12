"""Shared pytest fixtures and environment setup.

Loads the developer's .env into os.environ *before* any app module is
imported, so real credentials from .env are visible to tests. Falls back
to dummy values only for variables that are still missing.

This ordering matters: pydantic-settings resolves env vars before the
.env file, so setting a dummy env var while the real value sits in .env
would silently shadow the real value.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# 1. Load .env (if present) into os.environ so integration tests see the
#    developer's real credentials. Existing shell env values are preserved.
# ---------------------------------------------------------------------------
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path, override=False)


# ---------------------------------------------------------------------------
# 2. Fill in dummy values for required secrets only if not already set.
#    Case-insensitive check because pydantic-settings matches keys that way.
# ---------------------------------------------------------------------------
def _is_set(name: str) -> bool:
    upper = name.upper()
    return any(k.upper() == upper for k in os.environ)


_DEFAULTS = {
    "GROQ_API_KEY": "test-groq-key",
    "GUARDRAIL_API_TOKEN": "test-guardrail-token",
    "SWABI_API_KEY": "test-swabi-key",
    "REDIS_URL": "",
}

for _name, _value in _DEFAULTS.items():
    if not _is_set(_name):
        os.environ[_name] = _value


# ---------------------------------------------------------------------------
# 3. Import pytest after env setup so nothing observes a partially
#    configured environment.
# ---------------------------------------------------------------------------
import pytest  # noqa: E402


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"