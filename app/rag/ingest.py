"""LangChain-based ingestion pipeline.

Markdown text
    -> LangChain MarkdownHeaderTextSplitter  (strip_headers=True)
    -> heading-scoped Documents
    -> resolve each Document to a leaf config section
    -> bucket Documents by leaf, merge content (subheadings preserved)
    -> RecursiveCharacterTextSplitter for oversized leaves only
    -> final chunks with stable IDs

Public entry:
    ingest(config_path, source_path=..., release=..., chunk_size_cap=...)

Returns:
    IngestResult containing chunks, manifest, report, and known gaps.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.rag.config import (
    ConfigError,
    collect_review_flags,
    excluded_section_ids,
    excluded_title_substrings,
    load_config,
    validate_config,
)
from app.rag.schemas import (
    Audience,
    Chunk,
    ChunkMetadata,
    IndexVersion,
    IngestionReport,
    KnowledgeArea,
    ReviewFlag,
)

PIPELINE_VERSION = "2A2"

_HEADER_LEVELS = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
    ("####", "h4"),
]

_ID_RE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)*)\s*[\s\-—–:.]?\s*(.*?)\s*$")


class IngestError(Exception):
    """Deterministic ingestion failure."""



# Text helpers


def normalise_text(text: str) -> str:
    """Strip BOM, normalize line endings, and trailing whitespace."""
    text = text.replace("\ufeff", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in text.split("\n")).rstrip() + "\n"


def sha256_of(text: str) -> str:
    return hashlib.sha256(normalise_text(text).encode("utf-8")).hexdigest()


def _extract_section_id(heading: str) -> str | None:
    match = _ID_RE.match(heading)
    return match.group(1) if match else None


def _heading_title(heading: str) -> str:
    match = _ID_RE.match(heading)
    return match.group(2).strip() if match else heading.strip()


def _document_id(source_document: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", source_document.lower()).strip("-")
    return slug or "doc"


def _chunk_id(document_id: str, section_id: str, text: str) -> str:
    digest = hashlib.sha256()
    for part in (document_id, section_id, text):
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()[:16]



# LangChain loading and splitting

def _load_langchain_documents(text: str):
    """Split Markdown into heading-scoped LangChain Documents.

    strip_headers=True so page_content contains only body text; the
    heading path is carried in metadata and reattached by the pipeline.
    """
    try:
        from langchain_text_splitters import MarkdownHeaderTextSplitter
    except ImportError as exc:
        raise IngestError(
            "langchain-text-splitters is not installed. "
            "Install it with: pip install -r requirements-rag.txt"
        ) from exc

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=_HEADER_LEVELS,
        strip_headers=True,
    )
    return splitter.split_text(normalise_text(text))


def _split_oversized(text: str, chunk_size_cap: int) -> list[str]:
    """Split an oversized leaf. Never called unless text > cap."""
    if len(text) <= chunk_size_cap:
        return [text]

    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError as exc:
        raise IngestError(
            "langchain-text-splitters is required for oversized chunks"
        ) from exc

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size_cap,
        chunk_overlap=max(0, chunk_size_cap // 20),
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    return [doc.page_content for doc in splitter.create_documents([text])]


# Config flattening

def _flatten_config(
    cfg: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Return (leaves_by_id, split_parents_by_id).

    A leaf is any config section that produces chunks directly (top-level
    leaf OR a split parent's child). A split parent stores its children.
    """
    leaves: dict[str, dict[str, Any]] = {}
    parents: dict[str, list[dict[str, Any]]] = {}

    def walk(spec: dict[str, Any]) -> None:
        children = spec.get("split_into")
        if children is not None:
            parents[spec["id"]] = list(children)
            for child in children:
                walk(child)
        else:
            leaves[spec["id"]] = spec

    for section in cfg["sections"]:
        walk(section)

    return leaves, parents



# Heading path helpers

def _heading_path(document: Any) -> list[str]:
    metadata = document.metadata
    return [metadata[key] for key in ("h1", "h2", "h3", "h4")
            if metadata.get(key)]


def _numbered_path(path: list[str]) -> list[str]:
    return [sid for h in path if (sid := _extract_section_id(h)) is not None]


def _resolve_section_id_from_path(path: list[str]) -> str | None:
    for heading in reversed(path):
        section_id = _extract_section_id(heading)
        if section_id:
            return section_id
    return None


def _subheading_matches(subheading: str, claims: list[str]) -> bool:
    """Match a subheading against a claim list, tolerating case and
    leading section-id differences."""
    sub_full = subheading.strip().lower()
    sub_title = _heading_title(subheading).strip().lower()
    for claim in claims:
        c_full = claim.strip().lower()
        c_title = _heading_title(claim).strip().lower()
        if sub_full == c_full or sub_title == c_title:
            return True
    return False


def _resolve_leaf(
    path: list[str],
    leaves: dict[str, dict[str, Any]],
    parents: dict[str, list[dict[str, Any]]],
) -> tuple[str | None, dict[str, Any] | None, str | None]:
    """Map a heading path to a leaf config section.

    Returns (leaf_id, spec, skip_reason). If skip_reason is not None, the
    document should be skipped (parent body, unclaimed subheading, or
    explicit skip_subsections match).
    """
    nums = _numbered_path(path)
    if not nums:
        return None, None, "no_numeric_id"

    deepest = nums[-1]
    deepest_idx = next(
        (i for i, h in enumerate(path) if _extract_section_id(h) == deepest),
        None,
    )
    if deepest_idx is None:
        return None, None, "internal_error"

    # Case A — leaf section.
    if deepest in leaves:
        spec = leaves[deepest]
        skip_titles = [s.strip().lower()
                       for s in (spec.get("skip_subsections") or [])]
        for sub in path[deepest_idx + 1:]:
            if sub.strip().lower() in skip_titles:
                return deepest, None, f"skip:{sub}"
        return deepest, spec, None

    # Case B — under a split parent.
    if deepest in parents:
        if deepest_idx + 1 >= len(path):
            return deepest, None, "parent_body"
        subheading = path[deepest_idx + 1]
        for child in parents[deepest]:
            if _subheading_matches(
                subheading, child.get("includes_subsections") or []
            ):
                return child["id"], child, None
        return deepest, None, f"unclaimed:{subheading}"

    # Case C — try shallower ancestors.
    for sid in reversed(nums[:-1]):
        if sid in leaves:
            return sid, leaves[sid], None
        if sid in parents:
            idx = next(
                (i for i, h in enumerate(path)
                 if _extract_section_id(h) == sid),
                None,
            )
            if idx is not None and idx + 1 < len(path):
                subheading = path[idx + 1]
                for child in parents[sid]:
                    if _subheading_matches(
                        subheading, child.get("includes_subsections") or []
                    ):
                        return child["id"], child, None

    return deepest, None, "no_match"


# Merging helpers

def _merge_bucket(
    items: list[tuple[list[str], str]],
    *,
    leaf_heading: str,
    leaf_depth: int,
) -> str:
    """Merge documents belonging to one leaf.

    Subheadings deeper than the leaf are re-emitted so the hierarchy is
    preserved. The leaf's own heading is NOT re-emitted here — the caller
    prepends the full heading path once.
    """
    parts: list[str] = []
    for path, content in items:
        content = content.strip()
        if not content:
            continue
        # Headings below the leaf heading (excluding the leaf itself).
        try:
            leaf_idx = next(
                i for i, h in enumerate(path)
                if _extract_section_id(h) == leaf_heading
                or _heading_title(h).strip().lower()
                   == _heading_title(leaf_heading).strip().lower()
            )
        except StopIteration:
            leaf_idx = len(path) - 1

        extra = path[leaf_idx + 1:]
        if extra:
            # Re-emit as subheadings with their original relative level.
            sub_prefix = "\n\n".join(
                f"{'#' * (leaf_depth + 1 + i)} {h}"
                for i, h in enumerate(extra)
            )
            parts.append(f"{sub_prefix}\n\n{content}")
        else:
            parts.append(content)
    return "\n\n".join(parts)



# Gaps + version

def _collect_gaps(cfg: dict[str, Any]) -> dict[str, list[str]]:
    gaps: dict[str, list[str]] = {}

    def walk(section: dict[str, Any]) -> None:
        if section.get("known_gaps"):
            gaps[section["id"]] = [str(x).strip()
                                   for x in section["known_gaps"]]
        for child in section.get("split_into") or []:
            walk(child)

    for section in cfg["sections"]:
        walk(section)
    return gaps


def _compute_version(
    *,
    source_hash: str,
    config_hash: str,
    chunk_size_cap: int,
    release: bool,
    embedding_model: str,
) -> str:
    digest = hashlib.sha256()
    for part in (
        "swabi-rag-index/v2a1",
        source_hash,
        config_hash,
        str(chunk_size_cap),
        "1" if release else "0",
        embedding_model,
    ):
        digest.update(part.encode("utf-8"))
        digest.update(b"|")
    return digest.hexdigest()[:12]



# Pipeline result


@dataclass
class IngestResult:
    chunks: list[Chunk]
    manifest: IndexVersion
    report: IngestionReport
    known_gaps: dict[str, list[str]]



# Ingestion

def ingest(
    config_path: Path,
    *,
    source_path: Path | None = None,
    release: bool = False,
    chunk_size_cap: int = 4000,
) -> IngestResult:
    if chunk_size_cap <= 0:
        raise ConfigError(
            f"chunk_size_cap must be positive, got {chunk_size_cap}"
        )

    cfg = load_config(config_path)
    validate_config(cfg, release=release)

    source_path = source_path or Path(cfg["source_path"])
    if not source_path.exists():
        raise ConfigError(f"source not found: {source_path}")

    raw = source_path.read_text(encoding="utf-8")
    source_hash = sha256_of(raw)
    expected_hash = cfg.get("source_hash_expected")
    if expected_hash and expected_hash != source_hash:
        raise ConfigError(
            "source hash mismatch\n"
            f"  expected: {expected_hash}\n"
            f"  actual:   {source_hash}"
        )

    config_hash = sha256_of(config_path.read_text(encoding="utf-8"))
    documents = _load_langchain_documents(raw)
    if not documents:
        raise IngestError("no documents after LangChain split")

    leaves, parents = _flatten_config(cfg)
    excluded_ids = set(excluded_section_ids(cfg))
    excluded_titles = excluded_title_substrings(cfg)
    document_id = _document_id(cfg["source_document"])

    chunks: list[Chunk] = []
    skipped: list[str] = []
    excluded: list[str] = []
    over_cap: list[str] = []
    warnings: list[str] = []


    # Bucket documents by resolved leaf section id.

    buckets: dict[str, list[tuple[list[str], str]]] = defaultdict(list)
    bucket_path: dict[str, list[str]] = {}

    for document in documents:
        path = _heading_path(document)
        if not path:
            warnings.append("document without heading metadata skipped")
            continue

        nums = _numbered_path(path)

        if any(sid in excluded_ids for sid in nums):
            excluded.append(" > ".join(path))
            continue
        if excluded_titles and any(
            sub in " ".join(path).lower() for sub in excluded_titles
        ):
            excluded.append(" > ".join(path))
            continue

        leaf_id, spec, reason = _resolve_leaf(path, leaves, parents)

        if reason is not None:
            skipped.append(f"{leaf_id or '?'}: {reason}")
            continue
        if leaf_id is None or spec is None:
            warnings.append(
                f"unmapped heading path: {' > '.join(path)}"
            )
            continue
        lifecycle = spec.get("policy_lifecycle") or {}
        if lifecycle.get("status") == "withdrawn":
            excluded.append(
                f"{' > '.join(path)}: withdrawn policy "
                f"{lifecycle.get('policy_id', leaf_id)}"
            )
            continue

        buckets[leaf_id].append((path, document.page_content))
        bucket_path.setdefault(leaf_id, path)


    # One chunk per leaf; split only when oversized.

    for leaf_id, items in buckets.items():
        spec = leaves[leaf_id]
        path = bucket_path[leaf_id]
        leaf_depth = len(path)

        merged = _merge_bucket(
            items,
            leaf_heading=path[-1],
            leaf_depth=leaf_depth,
        )

        heading_prefix = "\n\n".join(
            f"{'#' * (i + 1)} {h}" for i, h in enumerate(path)
        )
        text = f"{heading_prefix}\n\n{merged}".rstrip() + "\n"

        pieces = _split_oversized(text, chunk_size_cap)

        flags: list[ReviewFlag] = []
        for index, flag in enumerate(spec.get("review_flags") or []):
            if not isinstance(flag, dict) or not flag.get("text"):
                raise IngestError(
                    f"section {leaf_id}: invalid review flag at index {index}"
                )
            flag_id = flag.get("id") or f"{leaf_id}#{index}"
            if release and not flag.get("id"):
                raise IngestError(
                    f"section {leaf_id}: review flag #{index} requires an "
                    "explicit id for release"
                )
            flags.append(ReviewFlag(
                id=str(flag_id),
                text=flag["text"].strip(),
                confirmed=flag.get("confirmed", False),
            ))

        for piece_index, piece in enumerate(pieces, start=1):
            chunk_section_id = (
                leaf_id if len(pieces) == 1
                else f"{leaf_id}__p{piece_index}"
            )
            if len(piece) > chunk_size_cap:
                over_cap.append(chunk_section_id)

            chunks.append(Chunk(
                text=piece,
                metadata=ChunkMetadata(
                    chunk_id=_chunk_id(document_id, chunk_section_id, piece),
                    document_id=document_id,
                    category=spec["category"],
                    section_id=chunk_section_id,
                    section_title=spec["title"],
                    section_path=" > ".join(path),
                    area=KnowledgeArea(spec["area"]),
                    audience=Audience(spec["audience"]),
                    source=cfg["default_source"],
                    source_version=cfg["source_version"],
                    last_updated=cfg["last_updated"],
                    policy_id=(spec.get("policy_lifecycle") or {}).get(
                        "policy_id"
                    ),
                    policy_status=(
                        (spec.get("policy_lifecycle") or {}).get("status")
                        or (
                            "not_applicable"
                            if spec["category"] not in cfg["policy_categories"]
                            else "unclassified"
                        )
                    ),
                    effective_from=(spec.get("policy_lifecycle") or {}).get(
                        "effective_from"
                    ),
                    expires_on=(spec.get("policy_lifecycle") or {}).get(
                        "expires_on"
                    ),
                    supersedes=(spec.get("policy_lifecycle") or {}).get(
                        "supersedes", []
                    ),
                    destination=spec.get("destination"),
                    activity=spec.get("activity"),
                    review_flags=flags,
                ),
            ))

    # -------------------------------------------------------------------
    # Coverage: every leaf must have produced at least one chunk.
    # -------------------------------------------------------------------
    produced_ids = {
        chunk.metadata.section_id.split("__p")[0] for chunk in chunks
    }
    missing_leaves = [
        sid for sid in leaves
        if sid not in produced_ids
        and sid not in excluded_ids
        and (leaves[sid].get("policy_lifecycle") or {}).get("status")
        != "withdrawn"
    ]
    if missing_leaves:
        raise IngestError(
            "config leaves not covered by any chunk: "
            f"{sorted(missing_leaves)}"
        )


    # Release checks

    all_flags = collect_review_flags(cfg)
    if release:
        unconfirmed = [f.id for f in all_flags if not f.confirmed]
        if unconfirmed:
            raise IngestError(
                f"--release blocked by unconfirmed review flags: {unconfirmed}"
            )
        if over_cap:
            raise IngestError(
                f"--release blocked by {len(over_cap)} over-cap chunks"
            )


    # Manifest + report

    version = _compute_version(
        source_hash=source_hash,
        config_hash=config_hash,
        chunk_size_cap=chunk_size_cap,
        release=release,
        embedding_model=settings.rag_embedding_model,
    )

    manifest = IndexVersion(
        index_version=version,
        pipeline_version=PIPELINE_VERSION,
        source_path=str(source_path),
        source_hash=source_hash,
        source_version=cfg["source_version"],
        config_hash=config_hash,
        chunk_size_cap=chunk_size_cap,
        embedding_model=settings.rag_embedding_model,
        release=release,
        chunk_count=len(chunks),
    )

    per_area: dict[str, int] = {}
    per_audience: dict[str, int] = {}
    per_category: dict[str, int] = {}
    for chunk in chunks:
        area = chunk.metadata.area.value
        audience = chunk.metadata.audience.value
        category = chunk.metadata.category
        per_area[area] = per_area.get(area, 0) + 1
        per_audience[audience] = per_audience.get(audience, 0) + 1
        per_category[category] = per_category.get(category, 0) + 1

    known_gaps = _collect_gaps(cfg)

    report = IngestionReport(
        built_at=datetime.now(UTC).isoformat(),
        source_path=str(source_path),
        source_hash=source_hash,
        source_version=cfg["source_version"],
        config_hash=config_hash,
        config_path=str(config_path),
        pipeline_version=PIPELINE_VERSION,
        total_chunks=len(chunks),
        per_area=per_area,
        per_audience=per_audience,
        per_category=per_category,
        over_cap=over_cap,
        skipped=skipped,
        excluded=excluded,
        known_gaps=known_gaps,
        review_flags=all_flags,
        warnings=warnings,
    )

    return IngestResult(
        chunks=chunks,
        manifest=manifest,
        report=report,
        known_gaps=known_gaps,
    )
