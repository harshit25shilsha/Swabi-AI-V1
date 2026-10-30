"""Request, response, and internal LLM schemas."""
from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from app.rag.schemas import Audience, KnowledgeArea


class AnswerType(StrEnum):
    ANSWERED = "answered"
    INSUFFICIENT_CONTEXT = "insufficient_context"
    NEEDS_LIVE_DATA = "needs_live_data"
    OUT_OF_SCOPE = "out_of_scope"


class SupportAskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    audience: Audience
    knowledge_area: KnowledgeArea | None = None
    conversation_id: str | None = None
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
            "Knowledge retrieval mode, or not_applicable when no knowledge "
            "retrieval ran."
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
