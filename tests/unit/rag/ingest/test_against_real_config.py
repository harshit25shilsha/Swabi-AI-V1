"""End-to-end validation against the real project config and source.

Skips if either file is missing.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.rag.ingest.chunker import build_chunks
from app.rag.ingest.config import (
    collect_review_flags,
    excluded_section_ids,
    load_config,
    validate_config,
)
from app.rag.ingest.source import (
    parse_markdown,
    validate_unique_section_ids,
    validate_unique_titles_within_parents,
)

REPO = Path(__file__).resolve().parents[4]
CONFIG = REPO / "config" / "knowledge_sections.yaml"


@pytest.fixture(scope="module")
def real_cfg():
    if not CONFIG.exists():
        pytest.skip(f"real config not present at {CONFIG}")
    cfg = load_config(CONFIG)
    return cfg


def test_real_config_validates(real_cfg):
    validate_config(real_cfg, release=False)

def _split_parent_ids(cfg) -> set[str]:
    out: set[str] = set()

    def _walk(spec: dict) -> None:
        if "split_into" in spec:
            out.add(spec["id"])
        for child in spec.get("split_into") or []:
            _walk(child)

    for s in cfg["sections"]:
        _walk(s)
    return out


def test_real_source_parses_and_validates(real_cfg):
    source_path = REPO / real_cfg["source_path"]
    if not source_path.exists():
        pytest.skip(f"real source not present at {source_path}")
    root = parse_markdown(source_path.read_text(encoding="utf-8"))
    validate_unique_section_ids(root)
    validate_unique_titles_within_parents(root, _split_parent_ids(real_cfg))


def test_real_config_chunks(real_cfg):
    source_path = REPO / real_cfg["source_path"]
    if not source_path.exists():
        pytest.skip(f"real source not present at {source_path}")
    root = parse_markdown(source_path.read_text(encoding="utf-8"))
    validate_unique_section_ids(root)
    validate_unique_titles_within_parents(root, _split_parent_ids(real_cfg))

    result = build_chunks(
        real_cfg, root,
        chunk_size_cap=4000,
        excluded=excluded_section_ids(real_cfg),
    )
    assert result.chunks, "no chunks produced"
    for c in result.chunks:
        assert len(c.text) <= 4000, (
            f"{c.metadata.section_id}: {len(c.text)} chars"
        )

def test_real_config_release_flags_collected(real_cfg):
    """Sanity check: release safety sees flags regardless of emitting status."""
    flags = collect_review_flags(real_cfg)
    print(f"collected {len(flags)} review flags")
