"""Config validation — structural checks and clear error messages."""
from __future__ import annotations

import copy

import pytest

from app.rag.ingest.config import (
    ConfigError,
    collect_review_flags,
    excluded_section_ids,
    validate_config,
)
from tests.unit.rag.ingest.conftest import BASE_CFG


def _cfg():
    return copy.deepcopy(BASE_CFG)


# Top-level


def test_base_config_passes():
    validate_config(_cfg(), release=False)


def test_missing_required_key_rejected():
    cfg = _cfg()
    del cfg["version"]
    with pytest.raises(ConfigError, match="version"):
        validate_config(cfg, release=False)


def test_source_format_must_be_markdown():
    cfg = _cfg()
    cfg["source_format"] = "pdf"
    with pytest.raises(ConfigError, match="source_format"):
        validate_config(cfg, release=False)


def test_null_hash_rejected_for_release():
    with pytest.raises(ConfigError, match="source_hash_expected"):
        validate_config(_cfg(), release=True)


def test_bad_hash_format_rejected():
    cfg = _cfg()
    cfg["source_hash_expected"] = "ABC123"
    with pytest.raises(ConfigError, match="64-char lowercase hex"):
        validate_config(cfg, release=False)


def test_duplicate_ids_rejected():
    cfg = _cfg()
    cfg["sections"].append(copy.deepcopy(cfg["sections"][0]))
    with pytest.raises(ConfigError, match="duplicate section id"):
        validate_config(cfg, release=False)


def test_bad_area_rejected():
    cfg = _cfg()
    cfg["sections"][0]["area"] = "unknown"
    with pytest.raises(ConfigError, match="invalid area"):
        validate_config(cfg, release=False)


def test_admin_audience_rejected():
    cfg = _cfg()
    cfg["sections"][0]["audience"] = "admin"
    with pytest.raises(ConfigError, match="invalid audience"):
        validate_config(cfg, release=False)


def test_category_not_in_list_rejected():
    cfg = _cfg()
    cfg["sections"][0]["category"] = "Z"
    with pytest.raises(ConfigError, match="not in categories"):
        validate_config(cfg, release=False)


# known_gaps


def test_known_gaps_must_be_list():
    cfg = _cfg()
    cfg["sections"][0]["known_gaps"] = "not a list"
    with pytest.raises(ConfigError, match="known_gaps"):
        validate_config(cfg, release=False)


def test_known_gaps_entries_must_be_strings():
    cfg = _cfg()
    cfg["sections"][0]["known_gaps"] = ["ok", 123]
    with pytest.raises(ConfigError, match="known_gaps"):
        validate_config(cfg, release=False)



# review_flags

def test_review_flag_confirmed_must_be_bool():
    cfg = _cfg()
    cfg["sections"][0]["review_flags"] = [
        {"text": "check", "confirmed": "false"},
    ]
    with pytest.raises(ConfigError, match="confirmed"):
        validate_config(cfg, release=False)


def test_review_flag_text_required():
    cfg = _cfg()
    cfg["sections"][0]["review_flags"] = [{"confirmed": False}]
    with pytest.raises(ConfigError, match="text"):
        validate_config(cfg, release=False)


def test_review_flag_accepts_valid():
    cfg = _cfg()
    cfg["sections"][0]["review_flags"] = [
        {"text": "check", "confirmed": False},
        {"id": "shared", "text": "another", "confirmed": True},
    ]
    validate_config(cfg, release=False)


# split_into

def test_split_parent_and_leaf_cannot_coexist():
    cfg = _cfg()
    cfg["sections"].append({
        "id": "9.9", "title": "Parent", "category": "A", "area": "swabi",
        "audience": "both",
        "split_into": [
            {"id": "9.9.a", "title": "A", "category": "A", "area": "swabi",
             "audience": "both", "includes_subsections": ["Sub A"]},
        ],
    })
    with pytest.raises(ConfigError, match="cannot be both"):
        validate_config(cfg, release=False)


def test_split_child_claiming_skipped_subheading_rejected():
    cfg = _cfg()
    cfg["sections"].append({
        "id": "9.9", "title": "Parent", "category": "A", "area": "swabi",
        "skip_subsections": ["Skipped"],
        "split_into": [
            {"id": "9.9.a", "title": "A", "category": "A", "area": "swabi",
             "audience": "both", "includes_subsections": ["Skipped"]},
        ],
    })
    with pytest.raises(ConfigError, match="skips"):
        validate_config(cfg, release=False)


def test_sibling_children_cannot_claim_same_subheading():
    cfg = _cfg()
    cfg["sections"].append({
        "id": "9.9", "title": "Parent", "category": "A", "area": "swabi",
        "split_into": [
            {"id": "9.9.a", "title": "A", "category": "A", "area": "swabi",
             "audience": "both", "includes_subsections": ["Shared"]},
            {"id": "9.9.b", "title": "B", "category": "A", "area": "swabi",
             "audience": "both", "includes_subsections": ["Shared"]},
        ],
    })
    with pytest.raises(ConfigError, match="more than one child"):
        validate_config(cfg, release=False)



# excluded_section_ids

def test_excluded_section_ids_accepts_string_list():
    cfg = _cfg()
    cfg["excluded_sections"] = ["0", "8.4"]
    assert excluded_section_ids(cfg) == ["0", "8.4"]


def test_excluded_section_ids_accepts_dict_list():
    cfg = _cfg()
    cfg["excluded_sections"] = [{"id": "0"}, {"id": "8.4"}]
    assert excluded_section_ids(cfg) == ["0", "8.4"]


def test_excluded_section_ids_rejects_mixed_garbage():
    cfg = _cfg()
    cfg["excluded_sections"] = [123]
    with pytest.raises(ConfigError, match="excluded_sections"):
        excluded_section_ids(cfg)


# collect_review_flags

def test_collect_review_flags_includes_parent_flags():
    cfg = _cfg()
    cfg["sections"].append({
        "id": "9.9", "title": "Parent", "category": "A", "area": "swabi",
        "review_flags": [{"text": "parent check"}],
        "split_into": [
            {"id": "9.9.a", "title": "A", "category": "A", "area": "swabi",
             "audience": "both", "includes_subsections": ["X"],
             "review_flags": [{"text": "child check"}]},
        ],
    })
    flags = collect_review_flags(cfg)
    texts = sorted(f.text for f in flags)
    assert texts == ["child check", "parent check"]
