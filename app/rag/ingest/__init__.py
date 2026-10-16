"""RAG knowledge ingestion package.

Modules:
    source  — parse markdown, normalise text, hash
    config  — load and validate knowledge_sections.yaml  (batch 2)
    chunker — walk config + tree, produce chunks         (batch 2)
    writer  — write artifacts to disk                    (batch 2)

The thin CLI lives at scripts/ingest_knowledge.py.
"""

from app.rag.ingest.source import (
    SourceError,
    SourceNode,
    find_by_section_id,
    find_duplicate_section_ids,
    find_subheading,
    has_preamble,
    iter_descendants,
    normalise_text,
    parse_markdown,
    sha256_of,
    validate_unique_section_ids,
)

# Bump when the chunking or parsing logic changes in a way that
# affects produced artifacts. Included in index_version.
PIPELINE_VERSION = "1"

__all__ = [
    "PIPELINE_VERSION",
    "SourceError",
    "SourceNode",
    "find_by_section_id",
    "find_duplicate_section_ids",
    "find_subheading",
    "has_preamble",
    "iter_descendants",
    "normalise_text",
    "parse_markdown",
    "sha256_of",
    "validate_unique_section_ids",
]
