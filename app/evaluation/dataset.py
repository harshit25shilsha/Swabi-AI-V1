"""Simple evaluation dataset loader."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


class EvalCase(BaseModel):
    id: str
    input: str
    expected: dict = Field(default_factory=dict)
    metadata: dict = Field(default_factory=dict)


class EvalDataset(BaseModel):
    name: str
    cases: list[EvalCase]

    @classmethod
    def load(cls, path: str | Path) -> EvalDataset:
        path = Path(path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls.model_validate(raw)
