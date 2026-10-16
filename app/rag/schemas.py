"""RAG schemas.

Phase 0 types (`DocumentMetadata`, `RetrievedDocument`, `RetrievalResult`)
are unchanged.

Phase 2 (RAG) adds ingestion-side types:

    KnowledgeArea, Audience     — controlled vocabularies
    ReviewFlag                  — human-confirmation flags from the config
    ChunkMetadata, Chunk        — one indexed section
    IndexVersion                — the on-disk index manifest
    IngestionReport             — the ingestion audit report

Only additive changes. Nothing here changes the Phase 0 contract.
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


# Phase 2 (RAG) types
# ---------------------------------------------------------------------------
# Phase 2 (RAG) additions — ingestion + retrieval types
# ---------------------------------------------------------------------------


class KnowledgeArea(StrEnum):
    SWABI = "swabi"
    TRAVEL = "travel"
    BOTH = "both"


class Audience(StrEnum):
    CUSTOMER = "customer"
    VENDOR = "vendor"
    BOTH = "both"


class ReviewFlag(BaseModel):
    """Human-confirmation item from the reviewed config.

    `id` is optional in the YAML; the chunker derives one deterministically
    from `(section_id, position)` when not supplied. Flags sharing an id
    are confirmed together. `--release` refuses to build while any flag has
    `confirmed=False`.
    """

    id: str | None = None
    text: str
    confirmed: bool = False


class ChunkMetadata(BaseModel):
    """Metadata for one indexed chunk. Populated deterministically from the
    source document structure plus the reviewed `knowledge_sections.yaml`."""

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
    """Manifest written alongside the index. Byte-stable across runs of
    identical inputs — no timestamp field."""

    index_version: str
    pipeline_version: str
    source_path: str
    source_hash: str
    source_version: str
    config_hash: str
    embedding_model: str | None = None
    release: bool = False
    chunk_count: int
    has_embeddings: bool = False


class IngestionReport(BaseModel):
    """Audit report for a single ingestion run. Contains `built_at` because
    it is a run log, not a stable artifact."""

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
    skipped_subsections: list[str] = Field(default_factory=list)
    known_gaps: dict[str, list[str]] = Field(default_factory=dict)
    review_flags: list[ReviewFlag] = Field(default_factory=list)
    coverage: dict[str, list[str]] = Field(default_factory=dict)
    excluded_sections: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
