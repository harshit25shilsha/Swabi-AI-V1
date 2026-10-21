"""Config loader + validator tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.rag.config import ConfigError, load_config, validate_config

FIXTURES = Path(__file__).parent / "fixtures"


def test_fixture_config_valid():
    validate_config(load_config(FIXTURES / "minimal.yaml"), release=False)


def test_missing_required_key():
    cfg = load_config(FIXTURES / "minimal.yaml")
    del cfg["version"]
    with pytest.raises(ConfigError, match="version"):
        validate_config(cfg, release=False)


def test_null_hash_rejected_for_release():
    cfg = load_config(FIXTURES / "minimal.yaml")
    with pytest.raises(ConfigError, match="source_hash_expected"):
        validate_config(cfg, release=True)


def test_bad_area():
    cfg = load_config(FIXTURES / "minimal.yaml")
    cfg["sections"][0]["area"] = "wrong"
    with pytest.raises(ConfigError, match="invalid area"):
        validate_config(cfg, release=False)


def test_admin_audience():
    cfg = load_config(FIXTURES / "minimal.yaml")
    cfg["sections"][0]["audience"] = "admin"
    with pytest.raises(ConfigError, match="invalid audience"):
        validate_config(cfg, release=False)


def test_blank_exclusion_title():
    cfg = load_config(FIXTURES / "minimal.yaml")
    cfg["excluded_sections"] = [{"id": "9.9", "title": "   "}]
    with pytest.raises(ConfigError, match="title"):
        validate_config(cfg, release=False)


def test_duplicate_section_ids():
    cfg = load_config(FIXTURES / "minimal.yaml")
    cfg["sections"].append(dict(cfg["sections"][0]))
    with pytest.raises(ConfigError, match="duplicate section id"):
        validate_config(cfg, release=False)
