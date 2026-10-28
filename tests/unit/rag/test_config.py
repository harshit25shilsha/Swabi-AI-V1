"""Config loader + validator tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.rag.config import ConfigError, load_config, validate_config
from app.rag.ingest import sha256_of

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


def test_policy_metadata_required_for_release():
    cfg = load_config(FIXTURES / "minimal.yaml")
    cfg["source_hash_expected"] = sha256_of(
        (FIXTURES / "minimal.md").read_text(encoding="utf-8")
    )
    del cfg["sections"][0]["policy_lifecycle"]
    with pytest.raises(ConfigError, match="requires policy_lifecycle"):
        validate_config(cfg, release=True)


def test_policy_dates_must_be_iso_dates():
    cfg = load_config(FIXTURES / "minimal.yaml")
    cfg["sections"][0]["policy_lifecycle"]["effective_from"] = "2026-1-1"
    with pytest.raises(ConfigError, match="YYYY-MM-DD"):
        validate_config(cfg, release=False)


def test_policy_lifecycle_must_be_on_leaf_section():
    cfg = load_config(FIXTURES / "minimal.yaml")
    cfg["sections"][2]["policy_lifecycle"] = {
        "policy_id": "destinations",
        "status": "approved",
        "effective_from": "2026-01-01",
    }
    with pytest.raises(ConfigError, match="set lifecycle on leaf sections"):
        validate_config(cfg, release=False)


def test_policy_expiry_must_follow_effective_date():
    cfg = load_config(FIXTURES / "minimal.yaml")
    lifecycle = cfg["sections"][0]["policy_lifecycle"]
    lifecycle["expires_on"] = "2025-12-31"
    with pytest.raises(ConfigError, match="must be after"):
        validate_config(cfg, release=False)


def test_superseded_policy_must_be_withdrawn():
    cfg = load_config(FIXTURES / "minimal.yaml")
    cfg["sections"][0]["policy_lifecycle"]["supersedes"] = ["beta-policy"]
    with pytest.raises(ConfigError, match="must be withdrawn"):
        validate_config(cfg, release=False)


def test_unknown_superseded_policy_is_rejected():
    cfg = load_config(FIXTURES / "minimal.yaml")
    cfg["sections"][0]["policy_lifecycle"]["supersedes"] = ["unknown-policy"]
    with pytest.raises(ConfigError, match="supersedes unknown policy"):
        validate_config(cfg, release=False)


def test_policy_category_must_be_declared_category():
    cfg = load_config(FIXTURES / "minimal.yaml")
    cfg["policy_categories"].append("Missing")
    with pytest.raises(ConfigError, match="every category"):
        validate_config(cfg, release=False)


def test_categories_must_be_classified_exactly_once():
    cfg = load_config(FIXTURES / "minimal.yaml")
    cfg["non_policy_categories"].append("A")
    with pytest.raises(ConfigError, match="must not overlap"):
        validate_config(cfg, release=False)


def test_draft_policy_blocks_release(tmp_path):
    cfg = load_config(FIXTURES / "minimal.yaml")
    cfg["source_hash_expected"] = sha256_of(
        (FIXTURES / "minimal.md").read_text(encoding="utf-8")
    )
    cfg["sections"][0]["policy_lifecycle"]["status"] = "draft"
    with pytest.raises(ConfigError, match="not approved"):
        validate_config(cfg, release=True)


def test_expired_policy_blocks_release():
    from datetime import UTC, datetime, timedelta

    cfg = load_config(FIXTURES / "minimal.yaml")
    cfg["source_hash_expected"] = sha256_of(
        (FIXTURES / "minimal.md").read_text(encoding="utf-8")
    )
    today = datetime.now(UTC).date()
    lifecycle = cfg["sections"][0]["policy_lifecycle"]
    lifecycle["effective_from"] = (today - timedelta(days=3)).isoformat()
    lifecycle["expires_on"] = (today - timedelta(days=1)).isoformat()
    with pytest.raises(ConfigError, match="has expired"):
        validate_config(cfg, release=True)


def test_future_policy_blocks_release():
    from datetime import UTC, datetime, timedelta

    cfg = load_config(FIXTURES / "minimal.yaml")
    cfg["source_hash_expected"] = sha256_of(
        (FIXTURES / "minimal.md").read_text(encoding="utf-8")
    )
    tomorrow = (datetime.now(UTC).date() + timedelta(days=1)).isoformat()
    cfg["sections"][0]["policy_lifecycle"]["effective_from"] = tomorrow
    with pytest.raises(ConfigError, match="not effective yet"):
        validate_config(cfg, release=True)


def test_release_accepts_explicit_supersession():
    cfg = load_config(FIXTURES / "minimal.yaml")
    cfg["source_hash_expected"] = sha256_of(
        (FIXTURES / "minimal.md").read_text(encoding="utf-8")
    )
    cfg["sections"][0]["policy_lifecycle"]["status"] = "withdrawn"
    cfg["sections"][1]["policy_lifecycle"]["supersedes"] = ["alpha-policy"]
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
