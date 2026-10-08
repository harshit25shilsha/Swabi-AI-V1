"""RAG schemas (interface only in Phase 0)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    source: str | None = None
    title: str | None = None
    url: str | None = None
    tags: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)


class RetrievedDocument(BaseModel):
    content: str
    score: float = 0.0
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)


class RetrievalResult(BaseModel):
    query: str
    documents: list[RetrievedDocument] = Field(default_factory=list)
