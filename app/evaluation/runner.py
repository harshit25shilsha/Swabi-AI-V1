"""Minimal evaluation runner.

Executes each case through a supplied async callable and records the output,
latency and any error. Designed to be extended in later phases.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from app.core.logging import get_logger
from app.evaluation.dataset import EvalCase, EvalDataset

log = get_logger(__name__)

CaseFn = Callable[[str], Awaitable[Any]]


@dataclass
class CaseResult:
    case_id: str
    input: str
    output: Any = None
    expected: dict = field(default_factory=dict)
    latency_ms: float = 0.0
    error: str | None = None
    passed: bool = False


@dataclass
class EvalReport:
    dataset: str
    results: list[CaseResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        return self.total - self.passed


class EvaluationRunner:
    def __init__(self, case_fn: CaseFn) -> None:
        self.case_fn = case_fn

    async def run(self, dataset: EvalDataset) -> EvalReport:
        report = EvalReport(dataset=dataset.name)
        for case in dataset.cases:
            report.results.append(await self._run_case(case))
        log.info(
            "evaluation_complete dataset=%s passed=%d failed=%d",
            report.dataset,
            report.passed,
            report.failed,
        )
        return report

    async def _run_case(self, case: EvalCase) -> CaseResult:
        start = time.perf_counter()
        try:
            output = await self.case_fn(case.input)
            latency = (time.perf_counter() - start) * 1000
            return CaseResult(
                case_id=case.id,
                input=case.input,
                output=output,
                expected=case.expected,
                latency_ms=latency,
                passed=True,
            )
        except Exception as exc:  # noqa: BLE001
            latency = (time.perf_counter() - start) * 1000
            return CaseResult(
                case_id=case.id,
                input=case.input,
                expected=case.expected,
                latency_ms=latency,
                error=str(exc),
                passed=False,
            )
