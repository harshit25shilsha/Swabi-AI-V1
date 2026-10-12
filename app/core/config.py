"""Centralized application configuration."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    # App
    app_name: str = "Swabi AI"
    app_env: str = "development"
    app_version: str = "0.1.0"
    debug: bool
    log_level: str

    # LLM

    llm_provider: str
    llm_model: str
    groq_api_key: str
    groq_chat_url: str
    llm_timeout: int
    llm_max_retries: int

    # Swabi Backend
    swabi_api_base_url: str
    swabi_api_key: str
    swabi_api_timeout: int

    # Redis
    redis_url: str

    # Feature flags

    enable_rag: bool
    enable_tools: bool
    enable_evaluation: bool


settings = Settings()
