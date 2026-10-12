"""Chat Safety request, response, and internal LLM schemas."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class Action(StrEnum):
    ALLOW = "ALLOW"
    BLOCK = "BLOCK"


class ModerationCategory(StrEnum):
    ALLOW = "ALLOW"
    PHONE_NUMBER = "PHONE_NUMBER"
    EMAIL = "EMAIL"
    SOCIAL_CONTACT = "SOCIAL_CONTACT"
    EXTERNAL_URL = "EXTERNAL_URL"
    PAYMENT_INFORMATION = "PAYMENT_INFORMATION"
    OFF_PLATFORM_COMMUNICATION = "OFF_PLATFORM_COMMUNICATION"
    OFF_PLATFORM_TRANSACTION = "OFF_PLATFORM_TRANSACTION"
    OTHER_PROHIBITED_CONTENT = "OTHER_PROHIBITED_CONTENT"


class SenderRole(StrEnum):
    user = "user"
    vendor = "vendor"


# ---- API request ----------------------------------------------------------
class ModerationRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)

    # Optional metadata — never required by the integration
    message_id: str | None = None
    conversation_id: str | None = None
    sender_id: str | None = None
    sender_role: SenderRole | None = None


# ---- API response ---------------------------------------------------------
class ModerationResponse(BaseModel):
    allowed: bool
    action: Action
    category: ModerationCategory | None = None
    confidence: float = Field(..., ge=0.0, le=1.0)

    # for backend side logging
    source: str = "unknown"  # deterministic | llm | fail_open
    request_id: str = ""
    prompt_version: str = ""
    provider: str = ""  # groq | gemini


# ---- LLM structured output (internal only, never exposed) -----------------
class LLMResult(BaseModel):
    decision: Action
    category: ModerationCategory | None = None
    confidence: float = Field(..., ge=0.0, le=1.0)


class GuardrailDecision(StrEnum):
    """Internal decision values for guardrail evaluation"""

    ALLOW = "ALLOW"
    BLOCK = "BLOCK"
    REVIEW = "REVIEW"  # internal only
    UNAVAILABLE = "UNAVAILABLE"  # internal only


class GuardrailResult(BaseModel):
    """Structured outcome of a guardrail check."""

    decision: GuardrailDecision
    reason_codes: list[str] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    source: str = "unknown"
    message: str | None = None
