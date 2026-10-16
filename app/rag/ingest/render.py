"""Render a section subtree into chunk text.

Guarantees:
  * Final chunk text length <= cap (unless cap too small — ConfigError).
  * Parent heading is prepended to every part.
  * Parent body is included in every split child (no content dropped).
  * No infinite recursion: if no splitter makes progress, RenderError.
"""
from __future__ import annotations

import re

from app.rag.ingest.source import SourceNode, find_subheading


class RenderError(Exception):
    pass


def _heading_line(node: SourceNode) -> str:
    return f"{'#' * node.level} {node.title}"


def _render_subtree(node: SourceNode, skip_titles: set[str], *,
                    is_root: bool) -> str:
    if not is_root and node.title in skip_titles:
        return ""
    parts = [_heading_line(node)]
    if node.body:
        parts.append(node.body)
    for child in node.children:
        sub = _render_subtree(child, skip_titles, is_root=False)
        if sub:
            parts.append(sub)
    return "\n\n".join(parts)


def render_leaf_body(node: SourceNode, skip_titles: set[str]) -> str:
    full = _render_subtree(node, skip_titles, is_root=True)
    top = _heading_line(node)
    if not full.startswith(top):
        raise RenderError(
            f"section {node.title!r}: rendered subtree does not start "
            "with the section heading"
        )
    return full[len(top):].lstrip("\n")


def render_claim(parent: SourceNode, titles: list[str]) -> str:
    """Parent heading + parent body (if any) + each claimed subtree."""
    parts = [_heading_line(parent)]
    if parent.body:
        parts.append(parent.body)
    for title in titles:
        node = find_subheading(parent, title)
        if node is None:
            raise RenderError(
                f"subheading {title!r} not found under {parent.title!r}"
            )
        parts.append(_render_subtree(node, set(), is_root=True))
    return "\n\n".join(parts)


# Splitting

def _split_at_subheadings(body: str) -> list[str]:
    lines = body.split("\n")
    groups: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        if line.startswith(("## ", "### ", "#### ")):
            if any(x.strip() for x in current):
                groups.append(current)
            current = [line]
        else:
            current.append(line)
    if any(x.strip() for x in current):
        groups.append(current)
    return ["\n".join(g).strip() for g in groups if any(x.strip() for x in g)]


def _split_at_paragraphs(body: str, budget: int) -> list[str]:
    paragraphs = [p for p in body.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) > budget and current:
            chunks.append(current)
            current = para
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _split_at_sentences(body: str, budget: int) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", body)
    chunks: list[str] = []
    current = ""
    for part in parts:
        candidate = f"{current} {part}".strip() if current else part
        if len(candidate) > budget and current:
            chunks.append(current)
            current = part
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks

def _split_at_whitespace(body: str, budget: int) -> list[str]:
    """Last-resort splitter: cut on any whitespace. Used when a single
    atomic sentence exceeds the budget."""
    tokens = body.split()
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for tok in tokens:
        add = len(tok) + (1 if current else 0)
        if current_len + add > budget and current:
            chunks.append(" ".join(current))
            current = [tok]
            current_len = len(tok)
        else:
            current.append(tok)
            current_len += add
    if current:
        chunks.append(" ".join(current))
    return chunks


def _render_chunk(parent_heading: str, body: str) -> str:
    return f"{parent_heading}\n\n{body}".rstrip() + "\n"


def split_body(
    body: str,
    section_id: str,
    cap: int,
    parent_heading: str,
) -> list[tuple[str, str]]:
    """Return [(chunk_section_id, text)]. Every text has len <= cap."""
    if cap <= 0:
        raise RenderError(
            f"section {section_id}: chunk_size_cap must be positive, got {cap}"
        )

    overhead = len(parent_heading) + 3  # "\n\n" + trailing "\n"
    budget = cap - overhead
    if budget <= 0:
        raise RenderError(
            f"section {section_id}: cap {cap} is smaller than the parent "
            f"heading overhead ({overhead})"
        )

    if len(body) <= budget:
        text = _render_chunk(parent_heading, body)
        if len(text) > cap:  # safety net
            raise RenderError(
                f"section {section_id}: rendered chunk length {len(text)} "
                f"exceeds cap {cap}"
            )
        return [(section_id, text)]

    for splitter in (
        _split_at_subheadings,
        lambda b: _split_at_paragraphs(b, budget),
        lambda b: _split_at_sentences(b, budget),
        lambda b: _split_at_whitespace(b, budget),
    ):
        parts = splitter(body)
        if len(parts) <= 1:
            continue
        out: list[tuple[str, str]] = []
        for i, part in enumerate(parts, 1):
            out.extend(split_body(part, f"{section_id}__p{i}", cap, parent_heading))
        return out

    raise RenderError(
        f"section {section_id}: body cannot be split under cap {cap}; "
        "a single atomic unit exceeds the budget"
    )
