"""Tests for deterministic and LLM-judged grounding metrics."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.core.exceptions import AIProviderError
from app.evaluation.grounding import (
    GroundingEvalReport,
    GroundingJudgeError,
    GroundingJudgeOutput,
    GroundingObservation,
    evaluate_grounding,
)
from app.rag.schemas import RetrievedChunk


def _chunk(chunk_id: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        section_id="section",
        section_path="Policies",
        text=f"Evidence for {chunk_id}",
        score=0.5,
    )


def _observation() -> GroundingObservation:
    return GroundingObservation(
        case_id="case-1",
        query="What is the policy?",
        answer="The policy says X.",
        answer_type="answered",
        expected_evidence_chunk_ids=["evidence-1", "evidence-2"],
        citation_chunk_ids=["evidence-1", "not-retrieved"],
        retrieved_chunks=[_chunk("evidence-1"), _chunk("evidence-2")],
    )


def test_evaluate_grounding_reports_retrieval_and_citation_quality():
    report = evaluate_grounding([_observation()])

    assert isinstance(report, GroundingEvalReport)
    result = report.cases[0]
    assert result.invalid_citation_ids == ["not-retrieved"]
    assert result.citation_integrity == 0.5
    assert result.citation_precision == 0.5
    assert result.citation_recall == 0.5
    assert result.retrieval_evidence_recall == 1.0
    assert report.answer_support_rate is None
    assert report.judge_coverage == 0.0
    assert report.judge_error_count == 0
    assert report.to_dict()["metrics"]["mean_answer_support_score"] is None


def test_evaluate_grounding_includes_optional_judge_scores():
    observed = _observation()
    judge_calls = []

    def judge(observation):
        judge_calls.append(observation)
        return GroundingJudgeOutput(
            answer_supported=False,
            answer_support_score=0.25,
            citation_support_score=0.5,
            unsupported_claims=["The answer claims X without evidence."],
        )

    report = evaluate_grounding([observed], judge=judge)

    assert judge_calls == [observed]
    assert report.answer_support_rate == 0.0
    assert report.mean_answer_support_score == 0.25
    assert report.mean_citation_support_score == 0.5
    assert report.cases[0].unsupported_claims == ["The answer claims X without evidence."]


def test_judge_failure_preserves_deterministic_metrics_and_reports_error():
    def failing_judge(observation):
        raise GroundingJudgeError("LLM judge failed (AIProviderError)")

    report = evaluate_grounding([_observation()], judge=failing_judge)

    assert report.citation_precision == 0.5
    assert report.answer_support_rate is None
    assert report.judge_coverage == 0.0
    assert report.judge_error_count == 1
    assert report.cases[0].judge_error == "LLM judge failed (AIProviderError)"


def test_evaluate_grounding_deduplicates_citations():
    observation = _observation()
    observation.citation_chunk_ids = ["evidence-1", "evidence-1"]

    report = evaluate_grounding([observation])

    assert report.cases[0].citation_chunk_ids == ["evidence-1"]
    assert report.cases[0].citation_precision == 1.0
    assert report.cases[0].citation_recall == 0.5


def test_evaluate_grounding_requires_observations():
    with pytest.raises(ValueError, match="at least one"):
        evaluate_grounding([])


def test_grounding_cli_loads_labeled_retrieval_dataset_and_judge_option():
    from scripts.evaluate_grounding import DEFAULT_DATASET, _args, _load_dataset

    dataset = _load_dataset(DEFAULT_DATASET)
    args = _args(["--judge"])

    assert len(dataset.cases) == 15
    booking_case = next(
        case for case in dataset.cases
        if case.id == "package-booking-through-enquiry"
    )
    assert booking_case.relevant_chunk_ids == [
        "47c29ba71be7e11c",
        "c3a5eb066f5f711e",
    ]
    assert dataset.cases[0].relevant_chunk_ids
    assert args.judge is True


def test_grounding_cli_retries_judging_after_provider_failure():
    from scripts.evaluate_grounding import _make_judge

    gateway = MagicMock()
    gateway.generate_structured_sync.side_effect = AIProviderError("rate limited")
    judge = _make_judge(gateway)

    with pytest.raises(GroundingJudgeError, match="AIProviderError"):
        judge(_observation())
    with pytest.raises(GroundingJudgeError, match="AIProviderError"):
        judge(_observation())

    assert gateway.generate_structured_sync.call_count == 2
    assert gateway.generate_structured_sync.call_args.kwargs == {
        "temperature": 0.0,
        "max_tokens": 1024,
    }
