"""Schemas for Swabi backend responses (Phase 0 minimal)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SwabiError(BaseModel):
    status_code: int
    detail: str
    raw: dict[str, Any] = Field(default_factory=dict)


class CustomerProfile(BaseModel):
    user_id: str
    first_name: str | None = None
    last_name: str | None = None
    email: str | None = None
    currency: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)
