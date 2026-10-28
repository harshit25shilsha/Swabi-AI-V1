"""Load and validate config/knowledge_sections.yaml."""
from __future__ import annotations

import re
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import yaml

from app.rag.schemas import Audience, KnowledgeArea, ReviewFlag

VALID_AREAS = {a.value for a in KnowledgeArea}
VALID_AUDIENCES = {a.value for a in Audience}
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")

_REQUIRED_TOP = (
    "version", "source_document", "source_version", "source_format",
    "source_path", "default_source", "last_updated", "categories",
    "policy_categories", "non_policy_categories", "sections",
)
_POLICY_STATUSES = {"approved", "draft", "withdrawn"}
_POLICY_LIFECYCLE_FIELDS = {
    "policy_id", "status", "effective_from", "expires_on", "supersedes",
}


class ConfigError(Exception):
    """Deterministic config failure."""


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ConfigError(msg)


def _str(v: Any, field: str) -> str:
    _require(isinstance(v, str) and v.strip() != "",
             f"{field}: must be a non-empty string")
    return v


def _bool(v: Any, field: str) -> bool:
    _require(isinstance(v, bool),
             f"{field}: must be a boolean, got {type(v).__name__}")
    return v


def _list(v: Any, field: str) -> list:
    _require(isinstance(v, list), f"{field}: must be a list")
    return v


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
    """Return ids from `excluded_sections` (bare string or dict with id)."""
    raw = cfg.get("excluded_sections") or []
    _list(raw, "excluded_sections")
    out: list[str] = []
    for i, item in enumerate(raw):
        if isinstance(item, str):
            out.append(_str(item, f"excluded_sections[{i}]"))
        elif isinstance(item, dict):
            if item.get("id"):
                out.append(_str(item["id"], f"excluded_sections[{i}].id"))
        else:
            raise ConfigError(
                f"excluded_sections[{i}]: must be string or mapping"
            )
    return out


def excluded_title_substrings(cfg: dict[str, Any]) -> list[str]:
    """Title substrings from `excluded_sections`. Blank titles rejected."""
    raw = cfg.get("excluded_sections") or []
    out: list[str] = []
    for i, item in enumerate(raw):
        if isinstance(item, dict) and "title" in item:
            t = _str(item["title"], f"excluded_sections[{i}].title")
            out.append(t.strip().lower())
    return out


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
        _require(isinstance(flag, dict), f"{ctx}: must be a mapping")
        _require("text" in flag, f"{ctx}: missing text")
        _str(flag["text"], f"{ctx}.text")
        if "confirmed" in flag:
            _bool(flag["confirmed"], f"{ctx}.confirmed")
        if "id" in flag and flag["id"] is not None:
            _str(flag["id"], f"{ctx}.id")


def _validate_known_gaps(raw: Any, field: str) -> None:
    _list(raw, field)
    for i, gap in enumerate(raw):
        _str(gap, f"{field}[{i}]")


def _iso_date(value: Any, field: str) -> date:
    _require(isinstance(value, str), f"{field}: must be an ISO date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ConfigError(f"{field}: must be an ISO date (YYYY-MM-DD)") from exc
    _require(parsed.isoformat() == value,
             f"{field}: must use YYYY-MM-DD format")
    return parsed


def _validate_policy_lifecycle(
    sections: list[dict[str, Any]],
    policy_categories: set[str],
    *,
    release: bool,
) -> None:
    policies: dict[str, dict[str, Any]] = {}

    def walk(section: dict[str, Any], ctx: str) -> None:
        children = section.get("split_into") or []
        if children:
            _require("policy_lifecycle" not in section,
                     f"{ctx}.policy_lifecycle: set lifecycle on leaf sections")
        for i, child in enumerate(children):
            walk(child, f"{ctx}.split_into[{i}]")
        if children:
            return

        category = section["category"]
        lifecycle = section.get("policy_lifecycle")
        if category not in policy_categories:
            _require(lifecycle is None,
                     f"{ctx}.policy_lifecycle: only valid for a policy category")
            return

        if lifecycle is None:
            _require(not release,
                     f"{ctx}: release requires policy_lifecycle metadata")
            return
        _require(isinstance(lifecycle, dict),
                 f"{ctx}.policy_lifecycle: must be a mapping")
        unknown = set(lifecycle) - _POLICY_LIFECYCLE_FIELDS
        _require(not unknown,
                 f"{ctx}.policy_lifecycle: unknown fields {sorted(unknown)}")
        for field in ("policy_id", "status", "effective_from"):
            _require(field in lifecycle,
                     f"{ctx}.policy_lifecycle: missing {field}")

        policy_id = _str(lifecycle["policy_id"],
                         f"{ctx}.policy_lifecycle.policy_id")
        status = _str(lifecycle["status"],
                      f"{ctx}.policy_lifecycle.status")
        _require(status in _POLICY_STATUSES,
                 f"{ctx}.policy_lifecycle.status: invalid status {status!r}")
        _require(policy_id not in policies,
                 f"duplicate policy_id: {policy_id}")

        effective_from = _iso_date(
            lifecycle["effective_from"],
            f"{ctx}.policy_lifecycle.effective_from",
        )
        expires_raw = lifecycle.get("expires_on")
        expires_on = (
            _iso_date(expires_raw, f"{ctx}.policy_lifecycle.expires_on")
            if expires_raw is not None else None
        )
        if expires_on is not None:
            _require(expires_on > effective_from,
                     f"{ctx}.policy_lifecycle.expires_on must be after "
                     "effective_from")

        supersedes = lifecycle.get("supersedes", [])
        _require(isinstance(supersedes, list),
                 f"{ctx}.policy_lifecycle.supersedes: must be a list")
        for i, superseded_id in enumerate(supersedes):
            _str(superseded_id,
                 f"{ctx}.policy_lifecycle.supersedes[{i}]")
        _require(len(supersedes) == len(set(supersedes)),
                 f"{ctx}.policy_lifecycle.supersedes: duplicate policy id")
        _require(policy_id not in supersedes,
                 f"{ctx}.policy_lifecycle cannot supersede itself")

        if release:
            _require(status in {"approved", "withdrawn"},
                     f"{ctx}: release blocked; policy {policy_id} is "
                     f"{status}, not approved")
            if status == "approved":
                today = datetime.now(UTC).date()
                _require(effective_from <= today,
                         f"{ctx}: policy {policy_id} is not effective yet")
                _require(expires_on is None or expires_on >= today,
                         f"{ctx}: policy {policy_id} has expired")

        policies[policy_id] = {
            "status": status,
            "supersedes": supersedes,
        }

    for i, section in enumerate(sections):
        walk(section, f"sections[{i}]")

    for policy_id, metadata in policies.items():
        for superseded_id in metadata["supersedes"]:
            _require(superseded_id in policies,
                     f"policy {policy_id} supersedes unknown policy "
                     f"{superseded_id}")
            _require(policies[superseded_id]["status"] == "withdrawn",
                     f"superseded policy {superseded_id} must be withdrawn")

    visited: set[str] = set()
    active: set[str] = set()

    def visit(policy_id: str) -> None:
        _require(policy_id not in active,
                 f"policy supersession cycle includes {policy_id}")
        if policy_id in visited:
            return
        active.add(policy_id)
        for predecessor in policies[policy_id]["supersedes"]:
            visit(predecessor)
        active.remove(policy_id)
        visited.add(policy_id)

    for policy_id in policies:
        visit(policy_id)


def validate_config(cfg: dict[str, Any], *, release: bool) -> None:
    for key in _REQUIRED_TOP:
        _require(key in cfg, f"config missing required key: {key}")

    for key in ("version", "source_document", "source_version",
                "default_source", "last_updated", "source_path"):
        _str(cfg[key], key)

    _require(cfg["source_format"] == "markdown",
             f"source_format must be 'markdown', got {cfg['source_format']!r}")

    cats = _list(cfg["categories"], "categories")
    _require(len(cats) > 0, "categories must be non-empty")
    cat_set = set()
    for i, c in enumerate(cats):
        cat_set.add(_str(c, f"categories[{i}]"))
    policy_categories = _list(cfg["policy_categories"], "policy_categories")
    policy_category_set = {
        _str(category, f"policy_categories[{i}]")
        for i, category in enumerate(policy_categories)
    }
    non_policy_categories = _list(
        cfg["non_policy_categories"], "non_policy_categories",
    )
    non_policy_category_set = {
        _str(category, f"non_policy_categories[{i}]")
        for i, category in enumerate(non_policy_categories)
    }
    _require(not (policy_category_set & non_policy_category_set),
             "policy_categories and non_policy_categories must not overlap")
    _require(policy_category_set | non_policy_category_set == cat_set,
             "every category must be classified as policy or non-policy")

    sections = _list(cfg["sections"], "sections")
    _require(len(sections) > 0, "sections must be non-empty")

    expected = cfg.get("source_hash_expected")
    if expected is None:
        _require(not release,
                 "source_hash_expected is null; --release requires a real hash")
    else:
        _require(isinstance(expected, str) and bool(_HEX64_RE.match(expected)),
                 "source_hash_expected must be null or 64-char lowercase hex")

    # excluded_sections structure — also triggers blank-title rejection
    excluded_section_ids(cfg)
    excluded_title_substrings(cfg)

    ids_seen: set[str] = set()

    def _check(section: Any, ctx: str) -> None:
        _require(isinstance(section, dict), f"{ctx}: must be a mapping")
        for req in ("id", "title", "category", "area"):
            _require(req in section, f"{ctx}: missing {req}")

        sid = _str(section["id"], f"{ctx}.id")
        _require(sid not in ids_seen, f"duplicate section id: {sid}")
        ids_seen.add(sid)

        _str(section["title"], f"section {sid}.title")
        _require(section["category"] in cat_set,
                 f"section {sid}: category not in categories")
        _require(section["area"] in VALID_AREAS,
                 f"section {sid}: invalid area {section['area']!r}")

        for opt in ("destination", "activity"):
            if opt in section and section[opt] is not None:
                _str(section[opt], f"section {sid}.{opt}")

        if "known_gaps" in section:
            _validate_known_gaps(section["known_gaps"], f"section {sid}.known_gaps")
        if "review_flags" in section:
            _validate_review_flags(section["review_flags"],
                                   f"section {sid}.review_flags")

        skip_titles: set[str] = set()
        if "skip_subsections" in section:
            for i, t in enumerate(_list(section["skip_subsections"],
                                        f"section {sid}.skip_subsections")):
                skip_titles.add(_str(t, f"section {sid}.skip_subsections[{i}]"))

        is_parent = "split_into" in section
        _require(not (is_parent and "audience" in section),
                 f"section {sid}: cannot be both split parent and leaf")

        if is_parent:
            children = _list(section["split_into"], f"section {sid}.split_into")
            _require(len(children) > 0,
                     f"section {sid}.split_into: must be non-empty")
            claims_seen: dict[str, str] = {}
            for child in children:
                _check(child, ctx=f"child of {sid}")
                inc = _list(child.get("includes_subsections") or [],
                            f"child {child.get('id')}.includes_subsections")
                _require(len(inc) > 0,
                         f"child {child['id']}.includes_subsections: non-empty")
                for title in inc:
                    t = _str(title, f"child {child['id']}.includes_subsections")
                    _require(t not in skip_titles,
                             f"child {child['id']} claims {t!r}, parent skips it")
                    _require(t not in claims_seen,
                             f"subheading {t!r} claimed by more than one child")
                    claims_seen[t] = child["id"]
        else:
            _require("audience" in section, f"section {sid}: missing audience")
            _require(section["audience"] in VALID_AUDIENCES,
                     f"section {sid}: invalid audience {section['audience']!r}")

    for i, s in enumerate(sections):
        _check(s, ctx=f"sections[{i}]")

    _validate_policy_lifecycle(
        sections, policy_category_set, release=release,
    )
