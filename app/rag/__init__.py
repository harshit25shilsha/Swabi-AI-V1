"""RAG package.

Phase 0: read-side types + Retriever protocol.
Phase 2: LangChain-based ingestion pipeline.
"""
from app.rag.schemas import (
    Audience,
    Chunk,
    ChunkMetadata,
    DocumentMetadata,
    IndexVersion,
    IngestionReport,
    KnowledgeArea,
    RetrievalResult,
    RetrievedDocument,
    ReviewFlag,
)

__all__ = [
    "Audience", "Chunk", "ChunkMetadata",
    "DocumentMetadata", "IndexVersion", "IngestionReport",
    "KnowledgeArea", "RetrievalResult", "RetrievedDocument", "ReviewFlag",
]
