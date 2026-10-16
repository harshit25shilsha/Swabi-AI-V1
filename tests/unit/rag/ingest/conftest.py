"""Shared fixtures for ingestion tests.

All fixtures use inline data — no external files needed.
"""
from __future__ import annotations

import pytest

BASE_CFG = {
    "version": "1.0.0",
    "source_document": "Test Document",
    "source_version": "1.0",
    "source_format": "markdown",
    "source_path": "unused.md",
    "source_hash_expected": None,
    "default_source": "Test KB",
    "last_updated": "2026-10",
    "categories": ["A", "B"],
    "sections": [
        {
            "id": "1.1", "title": "Alpha", "category": "A",
            "area": "swabi", "audience": "both",
        },
        {
            "id": "1.2", "title": "Beta", "category": "A",
            "area": "swabi", "audience": "both",
        },
    ],
}

BASE_MD = """# Doc

## 1.1 Alpha
alpha body

## 1.2 Beta
beta body
"""


@pytest.fixture
def base_cfg():
    """Return a deep copy so tests can mutate freely."""
    import copy
    return copy.deepcopy(BASE_CFG)


@pytest.fixture
def base_md():
    return BASE_MD
