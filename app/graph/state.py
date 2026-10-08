"""LangGraph shared state for the AI service."""

from __future__ import annotations

from typing import Any, TypedDict


class GraphState(TypedDict, total=False):
    request_id: str
    session_id: str | None
    user_id: str | None
    messages: list[dict[str, Any]]
    context: dict[str, Any]
    structured_data: dict[str, Any] | None
    retrieved_documents: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    errors: list[str]
    final_response: str | None
    metadata: dict[str, Any]
