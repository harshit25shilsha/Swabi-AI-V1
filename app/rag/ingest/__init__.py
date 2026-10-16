"""RAG knowledge ingestion package."""
from app.rag.ingest.chunker import ChunkBuildResult, build_chunks
from app.rag.ingest.config import (
    ConfigError,
    collect_review_flags,
    excluded_section_ids,
    load_config,
    validate_config,
)
from app.rag.ingest.render import (
    RenderError,
    render_claim,
    render_leaf_body,
    split_body,
)
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
    validate_unique_titles_within_parents,
)
from app.rag.ingest.writer import (
    CURRENT_NAME,
    PublishError,
    compute_index_version,
    current_version,
    publish,
)

PIPELINE_VERSION = "1"

__all__ = [
    "PIPELINE_VERSION",
    "ChunkBuildResult", "build_chunks",
    "ConfigError", "collect_review_flags", "excluded_section_ids",
    "load_config", "validate_config",
    "RenderError", "render_claim", "render_leaf_body", "split_body",
    "SourceError", "SourceNode",
    "find_by_section_id", "find_duplicate_section_ids", "find_subheading",
    "has_preamble", "iter_descendants", "normalise_text", "parse_markdown",
    "sha256_of", "validate_unique_section_ids",
    "CURRENT_NAME", "PublishError", "compute_index_version",
    "current_version", "publish",
    "validate_unique_titles_within_parents"
]
