"""Walk the reviewed config and the parsed source tree → chunks.

Coverage rule: every source node with non-empty body must appear in some
chunk, be skipped, be excluded, or be a structural container (no body).
Claims are matched by node identity via `find_subheading`; the CLI
guarantees unique titles within a parent before calling this module.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

from app.rag.ingest.config import ConfigError, excluded_section_ids
from app.rag.ingest.render import (
    RenderError,
    render_claim,
    render_leaf_body,
    split_body,
)
from app.rag.ingest.source import (
    SourceNode,
    find_by_section_id,
    find_subheading,
    iter_descendants,
)
from app.rag.schemas import (
    Audience,
    Chunk,
    ChunkMetadata,
    KnowledgeArea,
    ReviewFlag,
)


@dataclass
class ChunkBuildResult:
    chunks: list[Chunk] = field(default_factory=list)
    coverage: dict[str, list[str]] = field(default_factory=dict)
    skipped_subsections: list[str] = field(default_factory=list)
    over_cap: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _document_id(source_document: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", source_document.lower()).strip("-")
    return slug or "doc"


def _chunk_id(document_id: str, section_id: str, text: str) -> str:
    h = hashlib.sha256()
    h.update(document_id.encode("utf-8"))
    h.update(b"\x00")
    h.update(section_id.encode("utf-8"))
    h.update(b"\x00")
    h.update(text.encode("utf-8"))
    return h.hexdigest()[:16]


def _normalise_flags(raw: list[Any] | None, section_id: str) -> list[ReviewFlag]:
    if not raw:
        return []
    return [
        ReviewFlag(
            id=str(flag.get("id") or f"{section_id}#{i}"),
            text=flag["text"].strip(),
            confirmed=flag.get("confirmed", False),
        )
        for i, flag in enumerate(raw)
    ]

def _excluded_matchers(cfg: dict[str, Any]) -> tuple[set[str], list[str]]:
    """Return (excluded_ids, excluded_title_substrings_lowercased).

    Reads `excluded_sections`. Each entry may be a bare id string, or a
    dict with `id` and/or `title`. Titles are matched case-insensitively
    as substrings, so `SECTION 0 — PURPOSE ...` matches `PURPOSE ...`.
    """
    ids: set[str] = set()
    titles: list[str] = []
    raw = cfg.get("excluded_sections") or []
    for item in raw:
        if isinstance(item, str):
            ids.add(item)
        elif isinstance(item, dict):
            if item.get("id"):
                ids.add(str(item["id"]))
            if "title" in item and item["title"] is not None:
                t = str(item["title"]).strip().lower()
                if t:
                    titles.append(t)
    return ids, titles

def _validate_no_nested_claims(
    root: SourceNode, claimed_nodes: set[int]
) -> None:
    """Reject if any claimed node is an ancestor of another claimed node.

    Claims must be disjoint subtrees. Nested claims would duplicate the
    descendant content across two split children.
    """
    def _walk(node: SourceNode, ancestor_path: list[SourceNode]) -> None:
        if id(node) in claimed_nodes and ancestor_path:
            ancestor = ancestor_path[-1]
            raise ConfigError(
                f"nested split-parent claims: {node.title!r} is inside "
                f"{ancestor.title!r}; claims must not overlap"
            )
        if id(node) in claimed_nodes:
            ancestor_path = ancestor_path + [node]
        for child in node.children:
            _walk(child, ancestor_path)

    _walk(root, ancestor_path=[])

# Source coverage validation — by node identity

def _validate_source_coverage(
    cfg: dict[str, Any],
    root: SourceNode,
    excluded_ids: set[str],
) -> None:
    covered: set[int] = set()
    excluded: set[int] = set()
    skipped: set[int] = set()
    claimed_nodes: set[int] = set()

    # Split parents — descendants of these are validated by the
    # split-parent check in build_chunks._walk, not here.
    split_parent_ids: set[str] = set()

    def _collect_split_parents(spec: dict) -> None:
        if "split_into" in spec:
            split_parent_ids.add(spec["id"])
        for child in spec.get("split_into") or []:
            _collect_split_parents(child)

    for spec in cfg["sections"]:
        _collect_split_parents(spec)

    def _mark_covered(node: SourceNode) -> None:
        covered.add(id(node))
        for d in iter_descendants(node):
            covered.add(id(d))

    excluded_title_substrings: list[str] = []
    excluded_ids_from_cfg, excluded_title_substrings = _excluded_matchers(cfg)
    # `excluded_ids` is the caller-supplied set; union with config ids.
    all_excluded_ids = set(excluded_ids) | excluded_ids_from_cfg

    def _is_excluded(node: SourceNode) -> bool:
        if node.section_id and node.section_id in all_excluded_ids:
            return True
        title_lc = node.title.lower()
        return any(sub in title_lc for sub in excluded_title_substrings)

    def _mark_excluded(node: SourceNode) -> None:
        if _is_excluded(node):
            excluded.add(id(node))
            for d in iter_descendants(node):
                excluded.add(id(d))
            return
        for child in node.children:
            _mark_excluded(child)


    def _resolve_claim(parent_node: SourceNode, title: str, sid: str) -> SourceNode:
        claimed = find_subheading(parent_node, title)
        if claimed is None:
            raise ConfigError(
                f"section {sid}: subheading {title!r} not found under "
                f"{parent_node.title!r}"
            )
        node_id = id(claimed)
        if node_id in claimed_nodes:
            raise ConfigError(
                f"subheading {title!r} under {parent_node.title!r} "
                "resolved to a node already claimed"
            )
        claimed_nodes.add(node_id)
        return claimed

    for top in root.children:
        _mark_excluded(top)

    for spec in cfg["sections"]:
        node = find_by_section_id(root, spec["id"])
        if node is None:
            raise ConfigError(
                f"section {spec['id']} from config not found in source"
            )

        skip_titles = set(spec.get("skip_subsections") or [])
        for title in skip_titles:
            sub = find_subheading(node, title)
            if sub is not None:
                skipped.add(id(sub))
                for d in iter_descendants(sub):
                    skipped.add(id(d))

        if "split_into" in spec:
            covered.add(id(node))
            for child in spec["split_into"]:
                for title in child["includes_subsections"]:
                    claimed = _resolve_claim(node, title, child["id"])
                    _mark_covered(claimed)
        else:
            _mark_covered(node)

    _validate_no_nested_claims(root, claimed_nodes)

    def _walk(node: SourceNode, *, is_first_top_level: bool,
              under_split_parent: bool) -> None:
        accounted = (id(node) in covered or id(node) in excluded
                     or id(node) in skipped)

        if not accounted:
            # Doc-title exemption: first level-1 heading with no section id.
            is_doc_title = is_first_top_level and node.section_id is None
            # Descendants of split parents are validated by _walk in
            # build_chunks, not here.
            if not is_doc_title and not under_split_parent:
                has_body = bool(node.body.strip())
                children_accounted = all(
                    id(c) in covered or id(c) in excluded or id(c) in skipped
                    for c in node.children
                ) if node.children else True

                if has_body or not children_accounted:
                    label = node.section_id or f'"{node.title}"'
                    raise ConfigError(
                        f"source heading {label} is not referenced by any "
                        "config section, split-child claim, skip_subsections, "
                        "or excluded_sections"
                    )

        next_under_split = under_split_parent or (
            node.section_id is not None and node.section_id in split_parent_ids
        )
        for child in node.children:
            _walk(child, is_first_top_level=False,
                  under_split_parent=next_under_split)

    for i, top in enumerate(root.children):
        _walk(top, is_first_top_level=(i == 0), under_split_parent=False)

# Build

def build_chunks(
    cfg: dict[str, Any],
    root: SourceNode,
    *,
    chunk_size_cap: int,
    excluded: list[str] | None = None,
) -> ChunkBuildResult:
    if chunk_size_cap <= 0:
        raise ConfigError(
            f"chunk_size_cap must be positive, got {chunk_size_cap}"
        )

    if excluded is None:
        excluded = excluded_section_ids(cfg)
    excluded_set = set(excluded)

    _validate_source_coverage(cfg, root, excluded_set)

    result = ChunkBuildResult()
    doc_id = _document_id(cfg["source_document"])

    def _emit(
        spec: dict[str, Any],
        parent_title: str,
        *,
        heading: str,
        body: str,
        coverage_entry: list[str],
    ) -> None:
        sid = spec["id"]
        result.coverage[sid] = coverage_entry

        flags = _normalise_flags(spec.get("review_flags"), sid)
        try:
            parts = split_body(body, sid, chunk_size_cap, heading)
        except RenderError as exc:
            raise ConfigError(str(exc)) from exc

        for chunk_sid, text in parts:
            # Final check on the fully rendered chunk.
            if len(text) > chunk_size_cap:
                result.over_cap.append(chunk_sid)
            result.chunks.append(Chunk(
                text=text,
                metadata=ChunkMetadata(
                    chunk_id=_chunk_id(doc_id, chunk_sid, text),
                    document_id=doc_id,
                    category=spec["category"],
                    section_id=chunk_sid,
                    section_title=spec["title"],
                    section_path=(
                        f"{parent_title} → {spec['title']}"
                        if parent_title != spec["title"] else spec["title"]
                    ),
                    area=KnowledgeArea(spec["area"]),
                    audience=Audience(spec["audience"]),
                    source=cfg["default_source"],
                    source_version=cfg["source_version"],
                    last_updated=cfg["last_updated"],
                    destination=spec.get("destination"),
                    activity=spec.get("activity"),
                    review_flags=flags,
                ),
            ))

    def _walk(spec: dict[str, Any]) -> None:
        sid = spec["id"]
        node = find_by_section_id(root, sid)
        if node is None:
            raise ConfigError(f"section {sid} from config not found in source")

        skip_titles = set(spec.get("skip_subsections") or [])
        for title in skip_titles:
            result.skipped_subsections.append(f"{sid}: {title}")

        if "split_into" in spec:
            claimed_all: list[str] = []
            for child in spec["split_into"]:
                try:
                    text = render_claim(node, child["includes_subsections"])
                except RenderError as exc:
                    raise ConfigError(f"child {child['id']}: {exc}") from exc
                heading = f"{'#' * node.level} {node.title}"
                body = text[len(heading):].lstrip("\n")
                _emit(
                    child,
                    parent_title=node.title,
                    heading=heading,
                    body=body,
                    coverage_entry=list(child["includes_subsections"]),
                )
                claimed_all.extend(child["includes_subsections"])

            descendant_titles = [d.title for d in iter_descendants(node)]
            claimed = set(claimed_all)
            unclaimed = [t for t in descendant_titles
                         if t not in claimed and t not in skip_titles]
            if unclaimed:
                raise ConfigError(
                    f"section {sid}: subheadings not claimed and not skipped: "
                    f"{unclaimed}"
                )
            return

        try:
            body = render_leaf_body(node, skip_titles)
        except RenderError as exc:
            raise ConfigError(f"section {sid}: {exc}") from exc
        heading = f"{'#' * node.level} {node.title}"
        _emit(
            spec,
            parent_title=node.title,
            heading=heading,
            body=body,
            coverage_entry=[node.title],
        )

    for spec in cfg["sections"]:
        _walk(spec)

    return result
