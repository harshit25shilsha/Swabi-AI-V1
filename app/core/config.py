"""Centralized application configuration.

Field names use the lower-case Phase 0 convention. Upper-case properties
are exposed as read-only aliases so guardrail code that reads
``settings.GROQ_API_KEY`` (etc.) continues to work without a rename sweep.

Precedence (highest first):
    init kwargs > environment variables > .env file > field defaults
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # App
    app_name: str = "Swabi AI"
    app_env: str = "development"
    app_version: str = "0.1.0"
    service_version: str = "0.2.0"
    debug: bool = True
    log_level: str = "INFO"
    log_format: str = "json"  # "json" | "text"

    # LLM

    llm_provider: str = "groq"
    llm_model: str = "openai/gpt-oss-120b"
    groq_api_key: str
    groq_chat_url: str = "https://api.groq.com/openai/v1/chat/completions"
    llm_timeout: int = 60
    llm_max_retries: int = 2

    llm_temperature: float = 0.0
    llm_max_tokens: int = 1024
    llm_consensus_attempts: int = 1  # integration plan 5.2
    llm_fallback_enabled: bool = True

    # LLM Gemini fallback

    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.8-flash"
    gemini_timeout_seconds: int = 30

    # LLM - circuit breaker

    llm_circuit_failure_threshold: int = 5
    llm_circuit_open_seconds: int = 60

    # Swabi Backend
    swabi_api_base_url: str = ""
    swabi_api_key: str = ""
    swabi_api_timeout: int = 30

    # Redis
    redis_url: str = ""

    # Guardrail service (inbound auth for /api/v1/validate-message)
    guardrail_api_token: str  # required

    # Feature flags

    enable_rag: bool = False
    enable_tools: bool = False
    enable_evaluation: bool = False

    # Upper-case aliases — for guardrail code (GDv1) that reads
    # settings.GROQ_API_KEY etc. Lower-case field names remain canonical.

    @property
    def GROQ_API_KEY(self) -> str:
        return self.groq_api_key

    @property
    def GROQ_MODEL(self) -> str:
        return self.llm_model

    @property
    def GROQ_TIMEOUT_SECONDS(self) -> int:
        return self.llm_timeout

    @property
    def LLM_TEMPERATURE(self) -> float:
        return self.llm_temperature

    @property
    def LLM_MAX_TOKENS(self) -> int:
        return self.llm_max_tokens

    @property
    def LLM_CONSENSUS_ATTEMPTS(self) -> int:
        return self.llm_consensus_attempts

    @property
    def LLM_FALLBACK_ENABLED(self) -> bool:
        return self.llm_fallback_enabled

    @property
    def GEMINI_API_KEY(self) -> str:
        return self.gemini_api_key

    @property
    def GEMINI_MODEL(self) -> str:
        return self.gemini_model

    @property
    def GEMINI_TIMEOUT_SECONDS(self) -> int:
        return self.gemini_timeout_seconds

    @property
    def LLM_CIRCUIT_FAILURE_THRESHOLD(self) -> int:
        return self.llm_circuit_failure_threshold

    @property
    def LLM_CIRCUIT_OPEN_SECONDS(self) -> int:
        return self.llm_circuit_open_seconds

    @property
    def GUARDRAIL_API_TOKEN(self) -> str:
        return self.guardrail_api_token

    @property
    def SERVICE_VERSION(self) -> str:
        return self.service_version

    @property
    def LOG_LEVEL(self) -> str:
        return self.log_level

    @property
    def LOG_FORMAT(self) -> str:
        return self.log_format


settings = Settings()
