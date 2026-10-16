"""Load and validate config/knowledge_sections.yaml."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from app.rag.schemas import Audience, KnowledgeArea, ReviewFlag

VALID_AREAS = {a.value for a in KnowledgeArea}
VALID_AUDIENCES = {a.value for a in Audience}
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")

_REQUIRED_TOP = (
    "version", "source_document", "source_version", "source_format",
    "source_path", "default_source", "last_updated", "categories", "sections",
)
_CHILD_REQUIRED = ("id", "title", "category", "area", "audience",
                   "includes_subsections")


class ConfigError(Exception):
    """Deterministic config failure. Caller exits non-zero."""


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ConfigError(msg)


def _nonempty_str(value: Any, field: str) -> str:
    _require(isinstance(value, str) and value.strip() != "",
             f"{field}: must be a non-empty string")
    return value


def _bool(value: Any, field: str) -> bool:
    _require(isinstance(value, bool),
             f"{field}: must be a boolean, got {type(value).__name__}")
    return value


def _str_list(value: Any, field: str) -> list[str]:
    _require(isinstance(value, list), f"{field}: must be a list")
    for i, item in enumerate(value):
        _require(isinstance(item, str) and item.strip() != "",
                 f"{field}[{i}]: must be a non-empty string")
    return value


def _mapping(value: Any, field: str) -> dict:
    _require(isinstance(value, dict), f"{field}: must be a mapping")
    return value


def _list(value: Any, field: str) -> list:
    _require(isinstance(value, list), f"{field}: must be a list")
    return value


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"config not found: {path}")
    try:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"YAML parse error in {path}: {exc}") from exc
    if not isinstance(cfg, dict):
        raise ConfigError("config root must be a mapping")
    return cfg


def excluded_section_ids(cfg: dict[str, Any]) -> list[str]:
    """Return ids from `excluded_sections`.

    Entries may be:
      * a bare id string -> its value
      * a dict with `id` (and possibly `title`) -> the id
      * a dict with only `title` -> no id; skip (title matching is handled
        separately by the chunker)

    Entries with neither id nor title raise ConfigError.
    """
    raw = cfg.get("excluded_sections") or []
    _list(raw, "excluded_sections")
    out: list[str] = []
    for i, item in enumerate(raw):
        ctx = f"excluded_sections[{i}]"
        if isinstance(item, str):
            out.append(_nonempty_str(item, ctx))
            continue
        if isinstance(item, dict):
            if item.get("id"):
                out.append(_nonempty_str(item["id"], f"{ctx}.id"))
                continue
            if item.get("title"):
                continue  # title-only exclusion; handled by _excluded_matchers
            raise ConfigError(
                f"{ctx}: must have at least one of `id` or `title`"
            )
        raise ConfigError(
            f"{ctx}: expected string or mapping, got {type(item).__name__}"
        )
    return out


def validate_excluded_sections(cfg: dict[str, Any]) -> None:
    """Structural validation of `excluded_sections`, including titles.

    An entry may be a bare id string or a dict with `id` and/or `title`.
    Empty or whitespace-only values are rejected — a blank title would
    match every heading under substring matching and silently exclude the
    whole source.
    """
    raw = cfg.get("excluded_sections") or []
    _list(raw, "excluded_sections")
    for i, item in enumerate(raw):
        ctx = f"excluded_sections[{i}]"
        if isinstance(item, str):
            _nonempty_str(item, ctx)
            continue
        if isinstance(item, dict):
            if "id" in item:
                _nonempty_str(item["id"], f"{ctx}.id")
            if "title" in item:
                _nonempty_str(item["title"], f"{ctx}.title")
            if "id" not in item and "title" not in item:
                raise ConfigError(
                    f"{ctx}: must have at least one of `id` or `title`"
                )
            continue
        raise ConfigError(
            f"{ctx}: expected string or mapping, got {type(item).__name__}"
        )

def collect_review_flags(cfg: dict[str, Any]) -> list[ReviewFlag]:
    out: list[ReviewFlag] = []

    def _walk(section: dict, sid: str) -> None:
        for i, flag in enumerate(section.get("review_flags") or []):
            fid = flag.get("id") or f"{sid}#{i}"
            out.append(ReviewFlag(
                id=str(fid),
                text=flag["text"].strip(),
                confirmed=flag.get("confirmed", False),
            ))
        for child in section.get("split_into") or []:
            _walk(child, child["id"])

    for s in cfg["sections"]:
        _walk(s, s["id"])
    return out


def _validate_review_flags(raw: Any, field: str) -> None:
    _list(raw, field)
    for i, flag in enumerate(raw):
        ctx = f"{field}[{i}]"
        _mapping(flag, ctx)
        _require("text" in flag, f"{ctx}: missing text")
        _nonempty_str(flag["text"], f"{ctx}.text")
        if "confirmed" in flag:
            _bool(flag["confirmed"], f"{ctx}.confirmed")
        if "id" in flag and flag["id"] is not None:
            _nonempty_str(flag["id"], f"{ctx}.id")


def _validate_known_gaps(raw: Any, field: str) -> None:
    _list(raw, field)
    for i, gap in enumerate(raw):
        _nonempty_str(gap, f"{field}[{i}]")


def validate_config(cfg: dict[str, Any], *, release: bool) -> None:
    for key in _REQUIRED_TOP:
        _require(key in cfg, f"config missing required key: {key}")

    for key in ("version", "source_document", "source_version",
                "default_source", "last_updated", "source_path"):
        _nonempty_str(cfg[key], key)

    _require(cfg["source_format"] == "markdown",
             f"source_format must be 'markdown', got {cfg['source_format']!r}")

    categories = _str_list(cfg["categories"], "categories")
    _require(len(categories) > 0, "categories must be non-empty")

    sections = _list(cfg["sections"], "sections")
    _require(len(sections) > 0, "sections must be non-empty")

    expected = cfg.get("source_hash_expected")
    if expected is None:
        _require(not release,
                 "source_hash_expected is null; --release requires a real hash")
    else:
        _require(isinstance(expected, str) and bool(_HEX64_RE.match(expected)),
                 "source_hash_expected must be null or 64-char lowercase hex")

    excluded_section_ids(cfg)
    validate_excluded_sections(cfg)

    cat_set = set(categories)
    ids_seen: set[str] = set()

    def _check_leaf_fields(section: dict, sid: str) -> None:
        """Fields shared by top-level leaves and split children."""
        _nonempty_str(section["title"], f"section {sid}.title")
        _require(section["category"] in cat_set,
                 f"section {sid}: category {section['category']!r} not in categories")
        _require(section["area"] in VALID_AREAS,
                 f"section {sid}: invalid area {section['area']!r}")
        _require(section["audience"] in VALID_AUDIENCES,
                 f"section {sid}: invalid audience {section['audience']!r}")
        for opt in ("destination", "activity"):
            if opt in section and section[opt] is not None:
                _nonempty_str(section[opt], f"section {sid}.{opt}")
        if "known_gaps" in section:
            _validate_known_gaps(section["known_gaps"],
                                 f"section {sid}.known_gaps")
        if "review_flags" in section:
            _validate_review_flags(section["review_flags"],
                                   f"section {sid}.review_flags")

    def _check(section: Any, ctx: str) -> None:
        _mapping(section, ctx)
        for req in ("id", "title", "category", "area"):
            _require(req in section, f"{ctx}: missing {req}")

        sid = _nonempty_str(section["id"], f"{ctx}.id")
        _require(sid not in ids_seen, f"duplicate section id: {sid}")
        ids_seen.add(sid)

        is_parent = "split_into" in section
        _require(not (is_parent and "audience" in section),
                 f"section {sid}: cannot be both split parent and leaf")

        if is_parent:
            _nonempty_str(section["title"], f"section {sid}.title")
            _require(section["category"] in cat_set,
                     f"section {sid}: category {section['category']!r} not in categories")
            _require(section["area"] in VALID_AREAS,
                     f"section {sid}: invalid area {section['area']!r}")
            if "known_gaps" in section:
                _validate_known_gaps(section["known_gaps"],
                                     f"section {sid}.known_gaps")
            if "review_flags" in section:
                _validate_review_flags(section["review_flags"],
                                       f"section {sid}.review_flags")
        else:
            _require("audience" in section, f"section {sid}: missing audience")
            _check_leaf_fields(section, sid)

        skip_titles: set[str] = set()
        if "skip_subsections" in section:
            skip_titles = set(_str_list(section["skip_subsections"],
                                        f"section {sid}.skip_subsections"))

        if is_parent:
            children = _list(section["split_into"], f"section {sid}.split_into")
            _require(len(children) > 0,
                     f"section {sid}.split_into: must be non-empty")

            # Pre-validate every child's required child fields before access.
            for i, child in enumerate(children):
                cctx = f"section {sid}.split_into[{i}]"
                _mapping(child, cctx)
                for req in _CHILD_REQUIRED:
                    _require(req in child, f"{cctx}: missing {req}")

            claims_seen: dict[str, str] = {}
            for child in children:
                _check(child, ctx=f"child of {sid}")
                inc = _str_list(child["includes_subsections"],
                                f"child {child['id']}.includes_subsections")
                _require(len(inc) > 0,
                         f"child {child['id']}.includes_subsections: must be non-empty")
                for title in inc:
                    _require(title not in skip_titles,
                             f"child {child['id']} claims subheading "
                             f"{title!r} that parent {sid} skips")
                    _require(title not in claims_seen,
                             f"subheading {title!r} of {sid} claimed by more "
                             f"than one child ({claims_seen.get(title)} and "
                             f"{child['id']})")
                    claims_seen[title] = child["id"]

    for i, section in enumerate(sections):
        _check(section, ctx=f"sections[{i}]")
