"""Config loader and validator tests."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from app.rag.ingest.config import ConfigError, load_config, validate_config

FIXTURES = Path(__file__).parent / "fixtures"


def _load(**overrides):
    base = yaml.safe_load((FIXTURES / "minimal.yaml").read_text(encoding="utf-8"))
    base.update(overrides)
    return base


def test_fixture_config_is_valid():
    validate_config(load_config(FIXTURES / "minimal.yaml"), release=False)


def test_missing_top_key_rejected():
    cfg = _load()
    del cfg["source_version"]
    with pytest.raises(ConfigError, match="source_version"):
        validate_config(cfg, release=False)


def test_non_markdown_source_rejected():
    with pytest.raises(ConfigError, match="source_format"):
        validate_config(_load(source_format="pdf"), release=False)


def test_null_hash_rejected_for_release():
    with pytest.raises(ConfigError, match="source_hash_expected"):
        validate_config(_load(source_hash_expected=None), release=True)


def test_malformed_hash_rejected():
    with pytest.raises(ConfigError, match="source_hash_expected"):
        validate_config(_load(source_hash_expected="not-a-hash"), release=False)


def test_duplicate_section_ids_rejected():
    cfg = _load()
    cfg["sections"].append({**cfg["sections"][0]})
    with pytest.raises(ConfigError, match="duplicate section id"):
        validate_config(cfg, release=False)


def test_bad_area_rejected():
    cfg = _load()
    cfg["sections"][0]["area"] = "unknown"
    with pytest.raises(ConfigError, match="invalid area"):
        validate_config(cfg, release=False)


def test_admin_audience_rejected():
    cfg = _load()
    cfg["sections"][0]["audience"] = "admin"
    with pytest.raises(ConfigError, match="invalid audience"):
        validate_config(cfg, release=False)


def test_category_not_in_list_rejected():
    cfg = _load()
    cfg["sections"][0]["category"] = "NOPE"
    with pytest.raises(ConfigError, match="not in categories"):
        validate_config(cfg, release=False)


def test_split_child_claiming_skipped_subheading_rejected():
    cfg = _load()
    cfg["sections"].append({
        "id": "9.9", "title": "Parent", "category": "FIRST", "area": "swabi",
        "skip_subsections": ["Skipped subsection"],
        "split_into": [{
            "id": "9.9.a", "title": "A", "category": "FIRST", "area": "swabi",
            "audience": "both",
            "includes_subsections": ["Skipped subsection"],
        }],
    })
    with pytest.raises(ConfigError, match="skips"):
        validate_config(cfg, release=False)


def test_sibling_split_children_cannot_claim_same_subheading():
    cfg = _load()
    cfg["sections"].append({
        "id": "9.9", "title": "Parent", "category": "FIRST", "area": "swabi",
        "split_into": [
            {"id": "9.9.a", "title": "A", "category": "FIRST", "area": "swabi",
             "audience": "both", "includes_subsections": ["Shared"]},
            {"id": "9.9.b", "title": "B", "category": "FIRST", "area": "swabi",
             "audience": "both", "includes_subsections": ["Shared"]},
        ],
    })
    with pytest.raises(ConfigError, match="claimed by more than one"):
        validate_config(cfg, release=False)
