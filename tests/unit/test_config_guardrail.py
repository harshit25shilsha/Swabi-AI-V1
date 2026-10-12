"""Tests for the extended config, focused on guardrail-related fields."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.core.config import Settings

# Env variables we clear between tests so results are deterministic.
_TRACKED_ENV = (
    "LLM_MODEL",
    "LLM_TIMEOUT",
    "LLM_TEMPERATURE",
    "LLM_MAX_TOKENS",
    "LLM_CONSENSUS_ATTEMPTS",
    "LLM_FALLBACK_ENABLED",
    "GEMINI_API_KEY",
    "GEMINI_MODEL",
    "GEMINI_TIMEOUT_SECONDS",
    "LLM_CIRCUIT_FAILURE_THRESHOLD",
    "LLM_CIRCUIT_OPEN_SECONDS",
    "SERVICE_VERSION",
    "LOG_LEVEL",
    "LOG_FORMAT",
)


@pytest.fixture
def env(monkeypatch):
    """Set required secrets and clear optional env for each test."""
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.setenv("GUARDRAIL_API_TOKEN", "test-guardrail-token")
    for name in _TRACKED_ENV:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def _load() -> Settings:
    """Instantiate Settings without reading the developer's .env file."""
    return Settings(_env_file=None)


# ---------------------------------------------------------------------------
# Required-with-no-default fields
# ---------------------------------------------------------------------------


def test_required_secrets_have_no_default():
    fields = Settings.model_fields
    assert fields["groq_api_key"].is_required()
    assert fields["guardrail_api_token"].is_required()


def test_missing_guardrail_token_raises(env):
    env.delenv("GUARDRAIL_API_TOKEN", raising=False)
    with pytest.raises(ValidationError):
        _load()


def test_missing_groq_key_raises(env):
    env.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ValidationError):
        _load()


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_development_safe_defaults(env):
    s = _load()
    assert s.llm_consensus_attempts == 1
    assert s.llm_fallback_enabled is True
    assert s.llm_circuit_failure_threshold == 5
    assert s.llm_circuit_open_seconds == 60
    assert s.llm_temperature == 0.0
    assert s.llm_max_tokens == 1024
    assert s.gemini_api_key == ""
    assert s.log_format == "json"


def test_consensus_defaults_to_one(env):
    """Integration plan §5.2: consensus must default to a single attempt."""
    s = _load()
    assert s.llm_consensus_attempts == 1
    assert s.LLM_CONSENSUS_ATTEMPTS == 1


# ---------------------------------------------------------------------------
# Upper-case property aliases
# ---------------------------------------------------------------------------


def test_uppercase_aliases_mirror_lowercase_fields(env):
    s = _load()
    assert s.GROQ_API_KEY == s.groq_api_key
    assert s.GROQ_MODEL == s.llm_model
    assert s.GROQ_TIMEOUT_SECONDS == s.llm_timeout
    assert s.LLM_TEMPERATURE == s.llm_temperature
    assert s.LLM_MAX_TOKENS == s.llm_max_tokens
    assert s.LLM_CONSENSUS_ATTEMPTS == s.llm_consensus_attempts
    assert s.LLM_FALLBACK_ENABLED == s.llm_fallback_enabled
    assert s.GEMINI_API_KEY == s.gemini_api_key
    assert s.GEMINI_MODEL == s.gemini_model
    assert s.GEMINI_TIMEOUT_SECONDS == s.gemini_timeout_seconds
    assert s.LLM_CIRCUIT_FAILURE_THRESHOLD == s.llm_circuit_failure_threshold
    assert s.LLM_CIRCUIT_OPEN_SECONDS == s.llm_circuit_open_seconds
    assert s.GUARDRAIL_API_TOKEN == s.guardrail_api_token
    assert s.SERVICE_VERSION == s.service_version
    assert s.LOG_LEVEL == s.log_level
    assert s.LOG_FORMAT == s.log_format


def test_uppercase_env_var_populates_lowercase_field(env):
    env.setenv("GEMINI_API_KEY", "gem-key")
    env.setenv("LLM_CONSENSUS_ATTEMPTS", "1")
    s = _load()
    assert s.gemini_api_key == "gem-key"
    assert s.LLM_CONSENSUS_ATTEMPTS == 1


def test_lowercase_env_var_also_works(env):
    """pydantic-settings matches env var names case-insensitively."""
    env.setenv("guardrail_api_token", "lowercase-value")
    s = _load()
    assert s.guardrail_api_token == "lowercase-value"
