"""Grounding and citation evaluation for generated RAG answers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

from pydantic import BaseModel, Field

from app.rag.schemas import RetrievedChunk


class GroundingJudgeOutput(BaseModel):
    answer_supported: bool
    answer_support_score: float = Field(ge=0.0, le=1.0)
    citation_support_score: float = Field(ge=0.0, le=1.0)
    unsupported_claims: list[str] = Field(default_factory=list, max_length=5)
    rationale: str = Field(default="", max_length=300)


class GroundingJudgeError(Exception):
    """A semantic judge call failed but deterministic metrics remain usable."""


@dataclass
class GroundingObservation:
    case_id: str
    query: str
    answer: str
    answer_type: str
    expected_evidence_chunk_ids: list[str]
    citation_chunk_ids: list[str]
    retrieved_chunks: list[RetrievedChunk]


@dataclass
class GroundingCaseResult:
    case_id: str
    answer_type: str
    retrieved_chunk_ids: list[str]
    expected_evidence_chunk_ids: list[str]
    citation_chunk_ids: list[str]
    invalid_citation_ids: list[str]
    citation_integrity: float
    citation_precision: float
    citation_recall: float
    retrieval_evidence_recall: float
    answer_supported: bool | None = None
    answer_support_score: float | None = None
    citation_support_score: float | None = None
    unsupported_claims: list[str] | None = None
    judge_error: str | None = None


@dataclass
class GroundingEvalReport:
    cases: list[GroundingCaseResult]
    citation_integrity: float
    citation_precision: float
    citation_recall: float
    retrieval_evidence_recall: float
    judge_coverage: float
    judge_error_count: int
    answer_support_rate: float | None
    mean_answer_support_score: float | None
    mean_citation_support_score: float | None

    def to_dict(self) -> dict:
        return {
            "case_count": len(self.cases),
            "metrics": {
                "citation_integrity": self.citation_integrity,
                "citation_precision": self.citation_precision,
                "citation_recall": self.citation_recall,
                "retrieval_evidence_recall": self.retrieval_evidence_recall,
                "judge_coverage": self.judge_coverage,
                "judge_error_count": self.judge_error_count,
                "answer_support_rate": self.answer_support_rate,
                "mean_answer_support_score": self.mean_answer_support_score,
                "mean_citation_support_score": self.mean_citation_support_score,
            },
            "cases": [asdict(case) for case in self.cases],
        }


GroundingJudge = Callable[[GroundingObservation], GroundingJudgeOutput]


def evaluate_grounding(
    observations: Sequence[GroundingObservation],
    *,
    judge: GroundingJudge | None = None,
) -> GroundingEvalReport:
    """Measure evidence retrieval, citation integrity, and optional semantic support."""
    if not observations:
        raise ValueError("at least one grounding observation is required")

    results: list[GroundingCaseResult] = []
    for observation in observations:
        retrieved_ids = [chunk.chunk_id for chunk in observation.retrieved_chunks]
        retrieved_set = set(retrieved_ids)
        expected_set = set(observation.expected_evidence_chunk_ids)
        cited_ids = list(dict.fromkeys(observation.citation_chunk_ids))
        cited_set = set(cited_ids)
        invalid_citations = [chunk_id for chunk_id in cited_ids if chunk_id not in retrieved_set]
        cited_evidence = cited_set.intersection(expected_set, retrieved_set)
        judge_result = None
        judge_error = None
        if judge:
            try:
                judge_result = judge(observation)
            except GroundingJudgeError as exc:
                judge_error = str(exc)
        results.append(
            GroundingCaseResult(
                case_id=observation.case_id,
                answer_type=observation.answer_type,
                retrieved_chunk_ids=retrieved_ids,
                expected_evidence_chunk_ids=observation.expected_evidence_chunk_ids,
                citation_chunk_ids=cited_ids,
                invalid_citation_ids=invalid_citations,
                citation_integrity=(
                    (len(cited_ids) - len(invalid_citations)) / len(cited_ids) if cited_ids else 0.0
                ),
                citation_precision=(len(cited_evidence) / len(cited_ids) if cited_ids else 0.0),
                citation_recall=(len(cited_evidence) / len(expected_set) if expected_set else 0.0),
                retrieval_evidence_recall=(
                    len(retrieved_set.intersection(expected_set)) / len(expected_set)
                    if expected_set
                    else 0.0
                ),
                answer_supported=(judge_result.answer_supported if judge_result else None),
                answer_support_score=(judge_result.answer_support_score if judge_result else None),
                citation_support_score=(
                    judge_result.citation_support_score if judge_result else None
                ),
                unsupported_claims=(judge_result.unsupported_claims if judge_result else None),
                judge_error=judge_error,
            )
        )

    count = len(results)
    judged = [result for result in results if result.answer_supported is not None]
    return GroundingEvalReport(
        cases=results,
        citation_integrity=sum(result.citation_integrity for result in results) / count,
        citation_precision=sum(result.citation_precision for result in results) / count,
        citation_recall=sum(result.citation_recall for result in results) / count,
        retrieval_evidence_recall=(
            sum(result.retrieval_evidence_recall for result in results) / count
        ),
        judge_coverage=len(judged) / count,
        judge_error_count=sum(result.judge_error is not None for result in results),
        answer_support_rate=(
            sum(result.answer_supported is True for result in judged) / len(judged)
            if judged
            else None
        ),
        mean_answer_support_score=(
            sum(result.answer_support_score or 0.0 for result in judged) / len(judged)
            if judged
            else None
        ),
        mean_citation_support_score=(
            sum(result.citation_support_score or 0.0 for result in judged) / len(judged)
            if judged
            else None
        ),
    )
