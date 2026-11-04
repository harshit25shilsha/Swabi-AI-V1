"""Request, response, and internal LLM schemas."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

from app.rag.schemas import Audience, KnowledgeArea

MAX_SUPPORT_LIST_ITEMS = 20
DEFAULT_SUPPORT_LIST_ITEMS = 5


class AnswerType(StrEnum):
    ANSWERED = "answered"
    INSUFFICIENT_CONTEXT = "insufficient_context"
    NEEDS_LIVE_DATA = "needs_live_data"
    OUT_OF_SCOPE = "out_of_scope"


class ConversationTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=2000)


class SupportAskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    audience: Audience
    knowledge_area: KnowledgeArea | None = None
    conversation_id: str | None = None
    conversation_history: list[ConversationTurn] = Field(
        default_factory=list,
        max_length=10,
    )
    user_id: str | None = None


class Citation(BaseModel):
    chunk_id: str
    section_id: str
    section_path: str
    snippet: str = Field(..., max_length=300)


class SupportAskResponse(BaseModel):
    answer: str
    answer_type: AnswerType
    citations: list[Citation] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    retrieval_mode_used: str = Field(
        default="not_applicable",
        description=(
            "Knowledge retrieval mode, or not_applicable when no knowledge retrieval ran."
        ),
    )
    knowledge_version: str = ""
    request_id: str = ""
    provider: str = ""
    prompt_version: str = ""
    live_data_used: bool = False
    live_data_sources: list[str] = Field(default_factory=list)
    live_data_warning: str | None = None


class LLMAnswerOutput(BaseModel):
    """What the model must produce. Never exposed directly to the caller.

    The model may return only ANSWERED, INSUFFICIENT_CONTEXT, or
    OUT_OF_SCOPE. NEEDS_LIVE_DATA is deterministic and never requested
    from the LLM.
    """

    answer_type: AnswerType
    answer: str
    citation_chunk_ids: list[str] = Field(default_factory=list, max_length=3)
    confidence: float | None = None
