"""Lightweight base class for Swabi API tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from app.core.exceptions import ToolExecutionError
from app.core.logging import get_logger
from app.observability.logging import log_tool_call
from app.swabi.client import SwabiClient

log = get_logger(__name__)


class ToolResult(BaseModel):
    success: bool
    data: dict[str, Any] | None = None
    error: str | None = None


class BaseSwabiTool(ABC):
    """Base class for every Swabi tool.

    Each tool declares a name, description, input/output schemas, and an
    authorization requirement. Actual tools are added in later phases.
    """

    name: str = ""
    description: str = ""
    input_schema: type[BaseModel] = BaseModel
    output_schema: type[BaseModel] = BaseModel
    requires_auth: bool = True

    def __init__(self, client: SwabiClient | None = None) -> None:
        self.client = client or SwabiClient()

    @abstractmethod
    async def execute(self, payload: BaseModel) -> ToolResult:
        """Execute the tool with a validated payload."""

    async def __call__(self, raw_input: dict[str, Any]) -> ToolResult:
        try:
            payload = self.input_schema.model_validate(raw_input)
        except Exception as exc:
            raise ToolExecutionError(f"{self.name} invalid input: {exc}") from exc

        result = await self.execute(payload)
        log_tool_call(
            tool_name=self.name,
            success=result.success,
            error=result.error,
        )
        return result


class ToolMetadata(BaseModel):
    """Descriptive metadata used when registering tools with an LLM."""

    name: str
    description: str
    requires_auth: bool = True
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)
