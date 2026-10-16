"""RAG schemas.

Phase 0 read-side types (DocumentMetadata, RetrievedDocument,
RetrievalResult) are preserved for the existing Retriever protocol.

Phase 2 adds ingestion-side types used by the LangChain-based pipeline.
"""
from __future__ import annotations

from enum import StrEnum
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



# Phase 2 — ingestion types

class KnowledgeArea(StrEnum):
    SWABI = "swabi"
    TRAVEL = "travel"
    BOTH = "both"


class Audience(StrEnum):
    CUSTOMER = "customer"
    VENDOR = "vendor"
    BOTH = "both"


class ReviewFlag(BaseModel):
    id: str | None = None
    text: str
    confirmed: bool = False


class ChunkMetadata(BaseModel):
    """Metadata for one indexed chunk."""
    chunk_id: str
    document_id: str
    category: str
    section_id: str
    section_title: str
    section_path: str
    area: KnowledgeArea
    audience: Audience
    source: str
    source_version: str
    last_updated: str
    destination: str | None = None
    activity: str | None = None
    review_flags: list[ReviewFlag] = Field(default_factory=list)


class Chunk(BaseModel):
    text: str
    metadata: ChunkMetadata


class IndexVersion(BaseModel):
    """Manifest. Byte-stable across runs of identical inputs."""
    index_version: str
    pipeline_version: str
    source_path: str
    source_hash: str
    source_version: str
    config_hash: str
    chunk_size_cap: int
    release: bool = False
    chunk_count: int


class IngestionReport(BaseModel):
    """Run log. Contains built_at; not byte-stable."""
    built_at: str
    source_path: str
    source_hash: str
    source_version: str
    config_hash: str
    config_path: str
    pipeline_version: str
    total_chunks: int
    per_area: dict[str, int] = Field(default_factory=dict)
    per_audience: dict[str, int] = Field(default_factory=dict)
    per_category: dict[str, int] = Field(default_factory=dict)
    over_cap: list[str] = Field(default_factory=list)
    skipped: list[str] = Field(default_factory=list)
    excluded: list[str] = Field(default_factory=list)
    known_gaps: dict[str, list[str]] = Field(default_factory=dict)
    review_flags: list[ReviewFlag] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
