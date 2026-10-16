"""Markdown source parsing, normalisation, and hashing.

Headings are lines matching `^#{1,6}\\s+.+$`, ignored when inside a
fenced code block. Fence rules follow CommonMark:

  * Opening fence: 0-3 spaces, 3+ backticks or 3+ tildes, optional info
    string. Backtick info strings cannot contain backticks.
  * Closing fence: 0-3 spaces, same marker character, length >= opener,
    only trailing whitespace after the marker (no info string).

Text before the first heading is preserved as the root node's `body`.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator
from dataclasses import dataclass, field


class SourceError(Exception):
    """Raised on parse failures or empty source."""


_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(\S.*?)[ \t]*$")
_SECTION_ID_RE = re.compile(r"^([0-9]+(?:\.[0-9]+)*)\s*[\s\-—–:.]\s*(.+?)\s*$")
# Opening fence: 0-3 spaces, 3+ backticks or tildes, then anything (info string).
_FENCE_OPEN_RE = re.compile(r"^( {0,3})(`{3,}|~{3,})(.*)$")
# Closing fence: 0-3 spaces, 3+ backticks or tildes, only trailing whitespace.
_FENCE_CLOSE_RE = re.compile(r"^( {0,3})(`{3,}|~{3,})[ \t]*$")


@dataclass
class SourceNode:
    level: int
    title: str
    section_id: str | None
    section_title: str
    body: str
    children: list[SourceNode] = field(default_factory=list)


# Normalisation + hashing


def normalise_text(text: str) -> str:
    text = text.replace("\ufeff", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in text.split("\n")).rstrip() + "\n"


def sha256_of(text: str) -> str:
    """SHA-256 of the normalised UTF-8 content."""
    return hashlib.sha256(normalise_text(text).encode("utf-8")).hexdigest()


# Fenced-code detection



def _opening_fence(line: str) -> str | None:
    """Return the fence marker ('```' or '~~~' family) if this line opens
    a fence, else None.

    CommonMark: indent 0-3 spaces. Backtick fence info strings cannot
    contain a backtick; tilde fence info strings cannot contain a tilde.
    """
    m = _FENCE_OPEN_RE.match(line)
    if not m:
        return None
    marker = m.group(2)
    info = m.group(3)
    if marker[0] == "`" and "`" in info:
        return None
    if marker[0] == "~" and "~" in info:
        return None
    return marker


def _closes_fence(line: str, opening_marker: str) -> bool:
    """True if this line closes a fence opened with `opening_marker`.

    CommonMark: same marker character, length >= opener, only trailing
    whitespace after the marker (no info string).
    """
    m = _FENCE_CLOSE_RE.match(line)
    if not m:
        return False
    marker = m.group(2)
    return marker[0] == opening_marker[0] and len(marker) >= len(opening_marker)


# Parsing


def parse_markdown(text: str) -> SourceNode:
    """Parse markdown into a tree of headings with their body text."""
    text = normalise_text(text)
    lines = text.split("\n")

    in_fence: bool = False
    fence_marker: str = ""
    headings: list[tuple[int, int, str]] = []

    for i, line in enumerate(lines):
        if in_fence:
            if _closes_fence(line, fence_marker):
                in_fence = False
                fence_marker = ""
            continue

        opener = _opening_fence(line)
        if opener is not None:
            in_fence = True
            fence_marker = opener
            continue

        m = _HEADING_RE.match(line)
        if m:
            headings.append((i, len(m.group(1)), m.group(2).strip()))

    if not headings:
        raise SourceError("source markdown contains no headings")

    preamble = "\n".join(lines[: headings[0][0]]).strip("\n")

    root = SourceNode(
        level=0,
        title="",
        section_id=None,
        section_title="",
        body=preamble,
    )
    stack: list[SourceNode] = [root]

    for idx, (line_idx, level, title) in enumerate(headings):
        next_start = headings[idx + 1][0] if idx + 1 < len(headings) else len(lines)
        body = "\n".join(lines[line_idx + 1 : next_start]).strip("\n")

        section_id: str | None = None
        section_title = title
        m = _SECTION_ID_RE.match(title)
        if m:
            section_id = m.group(1)
            section_title = m.group(2)

        node = SourceNode(
            level=level,
            title=title,
            section_id=section_id,
            section_title=section_title,
            body=body,
        )
        while len(stack) > 1 and stack[-1].level >= level:
            stack.pop()
        stack[-1].children.append(node)
        stack.append(node)

    return root


# Tree queries


def find_by_section_id(node: SourceNode, section_id: str) -> SourceNode | None:
    if node.section_id == section_id:
        return node
    for child in node.children:
        found = find_by_section_id(child, section_id)
        if found is not None:
            return found
    return None


def iter_descendants(node: SourceNode) -> Iterator[SourceNode]:
    for child in node.children:
        yield child
        yield from iter_descendants(child)


def find_subheading(parent: SourceNode, title: str) -> SourceNode | None:
    for desc in iter_descendants(parent):
        if desc.title == title:
            return desc
    normalised = title.strip().lower()
    for desc in iter_descendants(parent):
        if desc.title.strip().lower() == normalised:
            return desc
    return None


def has_preamble(root: SourceNode) -> bool:
    """True if the source had text before the first heading."""
    return bool(root.body.strip())


def find_duplicate_section_ids(root: SourceNode) -> dict[str, list[int]]:
    """Return {section_id: [heading levels]} for every duplicate id."""
    seen: dict[str, list[int]] = {}

    def _walk(node: SourceNode) -> None:
        if node.section_id:
            seen.setdefault(node.section_id, []).append(node.level)
        for child in node.children:
            _walk(child)

    _walk(root)
    return {sid: levels for sid, levels in seen.items() if len(levels) > 1}


def validate_unique_section_ids(root: SourceNode) -> None:
    """Raise SourceError if the tree contains duplicate section ids."""
    dups = find_duplicate_section_ids(root)
    if dups:
        parts = [f"{sid} (levels {lvl})" for sid, lvl in sorted(dups.items())]
        raise SourceError("source markdown contains duplicate section ids: " + ", ".join(parts))

def validate_unique_titles_within_parents(
    root: SourceNode,
    split_parent_ids: set[str],
) -> None:
    """Raise SourceError if any split parent has two descendant subheadings
    with the same title. Only split parents are checked — for other parents
    title collisions are harmless because resolution is by section_id.

    `split_parent_ids` is the set of section ids in the config that have
    a `split_into` field.
    """
    problems: list[str] = []
    for sid in sorted(split_parent_ids):
        node = find_by_section_id(root, sid)
        if node is None:
            continue  # reported elsewhere
        seen: dict[str, int] = {}
        for desc in iter_descendants(node):
            seen[desc.title] = seen.get(desc.title, 0) + 1
        dupes = {t: c for t, c in seen.items() if c > 1}
        if dupes:
            problems.append(f"{sid}: duplicate subheading titles {sorted(dupes)}")

    if problems:
        raise SourceError(
            "source markdown has duplicate subheading titles under a split "
            "parent: " + "; ".join(problems)
        )
