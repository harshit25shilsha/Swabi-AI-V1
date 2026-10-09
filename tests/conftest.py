import os

# Required secrets — set before any app module is imported so the
# module-level Settings() in app.core.config does not raise.
os.environ.setdefault("GROQ_API_KEY", "test-groq-key")
os.environ.setdefault("GUARDRAIL_API_TOKEN", "test-guardrail-token")

# Optional env — keep tests deterministic regardless of the developer's shell
os.environ.setdefault("SWABI_API_KEY", "test-swabi-key")
os.environ.setdefault("REDIS_URL", "")  # disable Redis by default in tests


import pytest  # (import after env setup is intentional)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"
